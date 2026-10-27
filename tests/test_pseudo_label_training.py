"""Real FCOS training must be identical with audit labels enabled or changed."""

import copy
import json
from pathlib import Path
import random
import tempfile
from unittest import mock

import torch

from config import CurriculumConfig, EMAConfig, PseudoLabelEvalConfig, TeacherSchedule
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_wrapper import D3TWrapper
from trainer import CurriculumDomainAdaptationTrainer
from test_curriculum_wrapper_integration import build_config
from test_torchvision_fcos_adapter import build_adapter


def make_trainer(base, directory, *, enabled, interval=2):
    config = build_config(
        total_iters=9, output_dir=str(directory), ema=EMAConfig(alpha=.9, start_steps=1),
        curriculum=CurriculumConfig(phase1_end=1, phase2_end=3, phase3_end=7,
                                    phase2_rgb_sampling_ratio=.5, phase3_rgb_sampling_ratio=.5)
    )
    config.data.root = str(directory)
    config.pseudo_label_eval = PseudoLabelEvalConfig(enabled=enabled, interval=interval)
    schedule = TeacherSchedule(phase1=(.1, 0.), phase2=(.1, 0.),
                               phase3=(.1, 0.), phase4=(.1, 0.))
    config.distill.rgb_teacher = config.distill.ir_teacher = schedule
    # Exercise both teacher records on one physical IR batch.
    config.mid_routing.p3_ir_flow.teacher_names = ("rgb", "ir")
    student, rgb, ir = (copy.deepcopy(base) for _ in range(3))
    optimizer = torch.optim.SGD(student.parameters(), lr=.001, momentum=.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.8)
    images = torch.full((1, 3, 32, 48), .3)
    target = {"boxes": torch.tensor([[4., 3., 14., 15.]]),
              "labels": torch.tensor([0]), "stem": "rgb-a"}
    return CurriculumDomainAdaptationTrainer(
        student, rgb, ir, optimizer, config,
        [(images, [target], ("rgb-a",))], [(images, ("a",))], lr_scheduler=lr_scheduler
    )


def write_annotation(directory, label="person"):
    annotations = Path(directory) / "Annotations"
    annotations.mkdir(exist_ok=True)
    (annotations / "a.xml").write_text(
        f"<annotation><object><name>{label}</name><bndbox>"
        "<xmin>4</xmin><ymin>3</ymin><xmax>14</xmax><ymax>15</ymax>"
        "</bndbox></object></annotation>"
    )


def test_trainer_audits_existing_ir_outputs_and_resumes_without_duplicate_window():
    base = D3TWrapper(build_adapter(), D3TLossCriterion())
    with tempfile.TemporaryDirectory() as directory:
        write_annotation(directory)
        trainer = make_trainer(base, directory, enabled=True)
        with mock.patch.object(trainer.student, "raw", wraps=trainer.student.raw) as student_raw, \
             mock.patch.object(trainer.ir_teacher, "raw", wraps=trainer.ir_teacher.raw) as ir_raw, \
             mock.patch.object(trainer.rgb_teacher, "raw", wraps=trainer.rgb_teacher.raw) as rgb_raw:
            logs = [trainer.train_one_iteration() for _ in range(9)]
        path = Path(directory) / "logs" / "pseudo_label_eval.jsonl"
        assert path.exists(), "eligible IR KD batches must produce audit records"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [(r["global_step"], r["teacher"]) for r in records] == [
            (5, "rgb"), (5, "ir"), (7, "rgb"), (7, "ir"), (8, "ir")
        ]
        assert all(r["metrics"]["available_images"] == 1 for r in records)
        assert student_raw.call_count == 9
        assert rgb_raw.call_count == 5
        assert ir_raw.call_count == 5
        for call in student_raw.call_args_list:
            if call.kwargs["sample_ids"] == ("a",):
                assert call.args[1] is None
        assert all(not k.startswith("sup_") for log in logs if "ir" in log["step_type"]
                   and log["phase"] != "PHASE2_TRANSITION" for k in log)
        assert trainer.best_map == 0.
        trainer.save_checkpoint("resume.pth")
        restored = make_trainer(base, directory, enabled=True)
        restored.load_checkpoint(str(Path(directory) / "resume.pth"))
        assert restored.last_pseudo_label_eval_step == 8
        assert restored.global_step == 9
        restored.train_one_iteration()  # Step 10 enters the next global window.
        assert restored.last_pseudo_label_eval_step == 10
        restored.save_checkpoint("resume.pth")
        resumed_again = make_trainer(base, directory, enabled=True)
        resumed_again.load_checkpoint(str(Path(directory) / "resume.pth"))
        resumed_again.train_one_iteration()  # Step 11 must not duplicate step 10.
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["global_step"] for r in records] == [5, 5, 7, 7, 8, 10]


def test_audit_enabled_changed_or_missing_gt_preserves_losses_updates_and_rng():
    torch.manual_seed(31)
    base = D3TWrapper(build_adapter(min_size=(64, 96)), D3TLossCriterion())
    runs = []
    with tempfile.TemporaryDirectory() as directory:
        for name, enabled, label in [("off", False, "person"), ("on", True, "person"),
                                      ("changed", True, "car"), ("missing", True, None)]:
            output = Path(directory) / name
            output.mkdir()
            if label:
                write_annotation(output, label)
            trainer = make_trainer(base, output, enabled=enabled, interval=1)
            trainer.config.aug.hflip_prob = .5
            trainer.config.aug.brightness_prob = .5
            random.seed(99)
            torch.manual_seed(99)
            logs = []
            for _ in range(9):
                log = trainer.train_one_iteration()
                log.pop("pseudo_label_evaluations", None)
                logs.append(log)
            assert (output / "logs" / "pseudo_label_eval.jsonl").exists() == enabled
            runs.append((logs, copy.deepcopy(trainer.student.state_dict()),
                         copy.deepcopy(trainer.rgb_teacher.state_dict()),
                         copy.deepcopy(trainer.ir_teacher.state_dict()),
                         copy.deepcopy(trainer.optimizer.state_dict()),
                         copy.deepcopy(trainer.lr_scheduler.state_dict()),
                         {n: p.grad.clone() for n, p in trainer.student.named_parameters() if p.grad is not None},
                         random.getstate(), torch.get_rng_state().clone()))
        def assert_equal(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    assert_equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    assert_equal(a, b)
            else:
                assert left == right
        for run in runs[1:]:
            assert_equal(runs[0], run)


def test_empty_kd_mask_is_reported_and_legacy_checkpoint_remains_loadable():
    base = D3TWrapper(build_adapter(), D3TLossCriterion())
    with tempfile.TemporaryDirectory() as directory:
        write_annotation(directory)
        trainer = make_trainer(base, directory, enabled=True)
        trainer.global_step = 7  # First phase-4 batch; next completed step is 8.
        trainer.config.distill.ir_teacher.phase4 = (.1, 1.)
        trainer.train_one_iteration()
        path = Path(directory) / "logs" / "pseudo_label_eval.jsonl"
        record = json.loads(path.read_text())
        assert record["global_step"] == 8
        assert record["metrics"]["selected_rows_all"] == 0
        assert record["metrics"]["valid_fraction"] is None
        assert record["metrics"]["gt_coverage"] == 0.
        trainer.save_checkpoint("legacy.pth")
        checkpoint_path = Path(directory) / "legacy.pth"
        payload = torch.load(checkpoint_path, weights_only=False)
        payload.pop("last_pseudo_label_eval_step")
        # An older baseline config had no observer section either.
        payload["metadata"]["config"].pop("pseudo_label_eval")
        torch.save(payload, checkpoint_path)
        restored = make_trainer(base, directory, enabled=True)
        restored.load_checkpoint(str(checkpoint_path))
        assert restored.global_step == 8
        assert restored.last_pseudo_label_eval_step == 0
        restored.train_one_iteration()
        assert restored.last_pseudo_label_eval_step == 9

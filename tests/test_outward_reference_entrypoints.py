import copy
from pathlib import Path
import sys
import tempfile
from unittest import mock

from PIL import Image
import pytest
import torch

from config import CurriculumConfig, FCOSModelConfig, TrainingConfig
from evaluate_coco import _load_checkpoint_model
from example_flir import main, make_training_config, parse_args
from models.fcos_factory import build_checkpoint_metadata, build_fcos_d3t_model, build_fcos_d3t_trio
from test_outward_reference_training import make_outref_trainer
from test_pseudo_label_training import write_annotation
from test_torchvision_fcos_adapter import TinyBackbone
from torchvision.models.detection import FCOS
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights


def test_cli_maps_outref_architecture_and_loss_options():
    with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "synthetic",
        "--domain-neck", "--out-ref", "--out-ref-channels", "8", "--no-out-ref-batch-norm",
        "--out-ref-loss-weight", ".03", "--out-ref-teacher-score-thresh", ".6"]):
        config = make_training_config(parse_args())
    assert config.model.out_ref_enabled and config.model.out_ref_channels == 8
    assert not config.model.out_ref_batch_norm
    assert config.loss.weight_out_ref == .03
    assert config.loss.out_ref_teacher_score_thresh == .6


def test_cli_keeps_outref_opt_in_and_rejects_missing_neck():
    with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "synthetic"]):
        assert not make_training_config(parse_args()).model.out_ref_enabled
    with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "synthetic", "--out-ref"]):
        with pytest.raises(ValueError, match="neck"):
            make_training_config(parse_args())


def test_outref_factory_keeps_pretrained_loading_and_existing_detector_weights():
    config = TrainingConfig(model=FCOSModelConfig(
        weights="DEFAULT", domain_neck_enabled=True, out_ref_enabled=True,
        min_sizes=(64,), max_size=128,
    ), device="cpu")
    detector = FCOS(TinyBackbone(), num_classes=91, min_size=64, max_size=128)
    backbone_before = copy.deepcopy(detector.backbone.state_dict())
    tower_before = copy.deepcopy(detector.head.classification_head.conv.state_dict())
    calls = {}
    def builder(**kwargs):
        calls.update(kwargs)
        return detector
    model = build_fcos_d3t_model(config, detector_builder=builder)
    assert calls["weights"] is FCOS_ResNet50_FPN_Weights.DEFAULT
    assert calls["weights_backbone"] is None
    for name, value in backbone_before.items():
        torch.testing.assert_close(value, model.adapter.detector.backbone.state_dict()[name])
    for name, value in tower_before.items():
        torch.testing.assert_close(value, model.adapter.detector.head.classification_head.conv.state_dict()[name])
    assert model.adapter.domain_neck.blocks[0].out_ref is not None


@pytest.mark.parametrize("batch_norm", [False, True])
def test_coco_checkpoint_restores_outref_and_predictions(batch_norm):
    config = TrainingConfig(model=FCOSModelConfig(
        weights=None, pretrained_backbone=False, min_sizes=(64,), max_size=64,
        domain_neck_enabled=True, domain_neck_bottleneck_channels=8,
        out_ref_enabled=True, out_ref_channels=4, out_ref_batch_norm=batch_norm,
    ), device="cpu")
    model = build_fcos_d3t_model(config).eval()
    # Exercise a learned non-neutral attention, not merely its initial gain of one.
    with torch.no_grad():
        model.adapter.domain_neck.blocks[0].out_ref.gdt_convs_attn.bias.fill_(.4)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "outref.pth")
        torch.save({"metadata": build_checkpoint_metadata(config), "student": model.state_dict()}, path)
        restored, _ = _load_checkpoint_model(path, torch.device("cpu"), .05)
    image = torch.rand(3, 64, 64)
    with torch.no_grad():
        before, after = model.raw([image], domain="ir"), restored.raw([image], domain="ir")
    assert after.context.out_ref_logits == ()
    for name in ("class_logits", "quality_logits", "boxes"):
        torch.testing.assert_close(getattr(before.predictions[0], name), getattr(after.predictions[0], name))


def test_phase_transition_ema_and_resume_keep_outref_state_without_reinitializing():
    curriculum = CurriculumConfig(phase1_end=1, phase2_end=2, phase3_end=4,
                                  phase2_rgb_sampling_ratio=0., phase3_rgb_sampling_ratio=0.)
    with tempfile.TemporaryDirectory() as directory:
        trainer = make_outref_trainer(curriculum, directory)
        assert trainer.train_one_iteration()["step_type"] == "p1_rgb_supervised"
        assert trainer.train_one_iteration()["step_type"] == "p2_ir_flow"
        assert trainer.ir_neck_initialized
        before_student = copy.deepcopy(trainer.student.state_dict())
        before_teacher = copy.deepcopy(trainer.ir_teacher.state_dict())
        step = trainer.global_step
        assert trainer.train_one_iteration()["step_type"] == "p3_ir_flow"
        alpha = min(trainer.config.ema.alpha, (1 + step) / (10 + step))
        for name, actual in trainer.ir_teacher.state_dict().items():
            if ".out_ref." in name:
                expected = trainer.student.state_dict()[name]
                if actual.is_floating_point():
                    expected = before_teacher[name] * alpha + expected * (1 - alpha)
                torch.testing.assert_close(actual, expected)
        assert any(not torch.equal(value, trainer.student.state_dict()[name])
                   for name, value in before_student.items() if ".out_ref." in name)
        trainer.save_checkpoint("outref.pth")
        resumed = make_outref_trainer(curriculum, directory)
        resumed.load_checkpoint(str(Path(directory) / "outref.pth"))
        assert resumed.ir_neck_initialized and resumed.ema_initialized
        assert resumed.global_step == 3
        for name in trainer.student.state_dict():
            torch.testing.assert_close(trainer.student.state_dict()[name], resumed.student.state_dict()[name])
        for name in trainer.ir_teacher.state_dict():
            torch.testing.assert_close(trainer.ir_teacher.state_dict()[name], resumed.ir_teacher.state_dict()[name])
        for key, state in trainer.optimizer.state_dict()["state"].items():
            torch.testing.assert_close(state["momentum_buffer"], resumed.optimizer.state_dict()["state"][key]["momentum_buffer"])
        state_before = copy.deepcopy(resumed.student.state_dict())
        resumed._ensure_ema_initialized()
        for name, value in resumed.student.state_dict().items():
            torch.testing.assert_close(value, state_before[name])
        original_log, resumed_log = trainer.train_one_iteration(), resumed.train_one_iteration()
        assert original_log["loss_out_ref"] == resumed_log["loss_out_ref"]
        for name, value in resumed.student.state_dict().items():
            torch.testing.assert_close(value, trainer.student.state_dict()[name])


def test_cli_smoke_trains_outref_all_phases_and_saves_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_annotation(root)
        (root / "align_train.txt").write_text("a\nb\nc\nd\ne\n")
        for stem in "abcde":
            Image.new("RGB", (48, 32), color=(80, 60, 40)).save(root / "JPEGImages" / f"{stem}_RGB.jpg")
        args = ["example_flir.py", "--data_root", str(root), "--output_dir", str(root / "out"),
                "--device", "cpu", "--weights", "none", "--no-pretrained-backbone",
                "--domain-neck", "--domain-neck-bottleneck-channels", "8", "--out-ref",
                "--out-ref-channels", "4", "--out-ref-teacher-score-thresh", "0",
                "--total_iters", "4", "--phase1-end", "1", "--phase2-end", "2", "--phase3-end", "3",
                "--phase2-rgb-ratio", "0", "--phase3-rgb-ratio", "0", "--phase4-rgb-ratio", "0",
                "--batch_size", "1", "--eval_batch_size", "2", "--workers", "0",
                "--min-sizes", "64", "--max_size", "128"]
        built = []
        def tiny_factory(config):
            detector = FCOS(TinyBackbone(), num_classes=91, min_size=64, max_size=128)
            trio = build_fcos_d3t_trio(config, base_detector=detector)
            built.append(trio)
            return trio
        with mock.patch.object(sys, "argv", args), \
             mock.patch("training.bootstrap.build_fcos_d3t_trio", side_effect=tiny_factory), \
             mock.patch("example_flir.configure_training_logging", return_value=root / "train.log"):
            main(parse_args())
        final = torch.load(root / "out" / "final_model.pth", weights_only=False)
        assert final["global_step"] == 4 and final["ir_neck_initialized"]
        assert final["metadata"]["config"]["model"]["out_ref_enabled"]
        assert final["metadata"]["config"]["loss"]["weight_out_ref"] == .1
        assert final["student"]["adapter.domain_neck.blocks.0.out_ref.gdt_convs.1.num_batches_tracked"] == 4
        assert all(not p.requires_grad for p in built[0][2].parameters())
        assert not (root / "out" / "logs" / "pseudo_label_eval.jsonl").exists()

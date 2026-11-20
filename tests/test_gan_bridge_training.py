"""Exercise four-domain training, initialization, CLI and durable resume."""

import copy
from collections import Counter
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

import config as configs
from models.d3t_wrapper import D3TWrapper
from trainer import CurriculumDomainAdaptationTrainer
from test_curriculum_wrapper_integration import (
    TinyAdapter, TinyCriterion, RGBDataset, IRDataset, collate_rgb, collate_ir,
    StatefulLRScheduler,
)


def bridge_config(output_dir="outputs", **kwargs):
    return configs.TrainingConfig(
        model=configs.FCOSModelConfig(weights=None, pretrained_backbone=False),
        gan_bridge=configs.GANBridgeConfig(checkpoint="fixture.pth"),
        aug=configs.AugConfig(hflip_prob=0, blur_prob=0, brightness_prob=0, contrast_prob=0),
        workflow="gan_bridge", total_iters=100, device="cpu", output_dir=output_dir,
        **kwargs,
    )


def make_trainer(config, translator=None):
    from training.gan_bridge import FrozenGANTranslator
    from test_gan_bridge import FirstChannelGenerator

    if translator is None:
        translator = FrozenGANTranslator(FirstChannelGenerator(), identity={"sha256": "fixture", "architecture": "fixture"})
    student = D3TWrapper(TinyAdapter(), TinyCriterion())
    rgb_teacher, ir_teacher = copy.deepcopy(student), copy.deepcopy(student)
    return CurriculumDomainAdaptationTrainer(
        student, rgb_teacher, ir_teacher, torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9), config,
        DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb),
        DataLoader(IRDataset(), batch_size=1, collate_fn=collate_ir),
        gan_translator=translator,
    )


class BridgeTrainingTests(unittest.TestCase):
    def test_real_fcos_dense_distillation_runs_on_each_domain_with_shared_resize(self):
        from loss.d3t_criterion import D3TLossCriterion
        from test_torchvision_fcos_adapter import build_adapter

        config = bridge_config()
        schedule = configs.TeacherSchedule(
            phase1=(1, 0), phase2=(1, 0), phase3=(1, 0), phase4=(1, 0)
        )
        config.distill = configs.DistillConfig(rgb_teacher=schedule, ir_teacher=copy.deepcopy(schedule))
        trainer = make_trainer(config)
        student = D3TWrapper(build_adapter(min_size=(64, 96), max_size=128), D3TLossCriterion())
        trainer.student = student
        trainer.rgb_teacher, trainer.ir_teacher = copy.deepcopy(student), copy.deepcopy(student)
        trainer._setup_models()
        trainer.optimizer = torch.optim.SGD(student.parameters(), lr=0.001)
        # Jump to stage 3 via real scheduler state; the lifecycle flags represent
        # an already warmed student, allowing a short real-FCOS smoke test.
        for step in range(40):
            trainer.scheduler.get_next_domain(step)
        trainer.global_step = 40
        trainer.ema_initialized = True
        seen = set()
        for _ in range(10):
            item = trainer.train_one_iteration()
            seen.add(item["domain"])
            self.assertTrue(torch.isfinite(torch.tensor(item["total_loss"])))
            self.assertGreater(item["kd_selected_count_" + item["ema_target"]], 0)
        self.assertEqual(seen, {"rgb", "saga", "ir_gan", "ir_real"})

    def test_failed_iteration_does_not_advance_the_resumable_domain_schedule(self):
        for failure in (RuntimeError("forward failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                config = bridge_config(directory)
                trainer = make_trainer(config)
                for _ in range(23):
                    trainer.train_one_iteration()
                before = copy.deepcopy(trainer.scheduler.state_dict())
                with mock.patch.object(trainer.student, "raw", side_effect=failure):
                    with self.assertRaises(type(failure)):
                        trainer.train_one_iteration()
                self.assertEqual(trainer.global_step, 23)
                self.assertEqual(trainer.scheduler.state_dict(), before)
                trainer.save_checkpoint("interrupted.pth")
                resumed = make_trainer(config)
                resumed.load_checkpoint(str(Path(directory) / "interrupted.pth"))
                self.assertEqual(resumed.train_one_iteration()["domain"], trainer.train_one_iteration()["domain"])

    def test_checkpoint_restores_teachers_optimizer_lr_and_next_domain_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = bridge_config(directory)
            trainer = make_trainer(config)
            trainer.lr_scheduler = StatefulLRScheduler()
            for _ in range(53):
                trainer.train_one_iteration()
            trainer.save_checkpoint("bridge.pth")
            expected_domains = [trainer.train_one_iteration()["domain"] for _ in range(47)]
            resumed = make_trainer(config)
            resumed.lr_scheduler = StatefulLRScheduler()
            resumed.load_checkpoint(str(Path(directory) / "bridge.pth"))
            self.assertEqual(resumed.global_step, 53)
            self.assertTrue(resumed.ema_initialized)
            self.assertEqual(resumed.lr_scheduler.steps, 53)
            self.assertIn("momentum_buffer", next(iter(resumed.optimizer.state.values())))
            checkpoint = torch.load(Path(directory) / "bridge.pth", weights_only=False)
            for key, model in (("student", resumed.student), ("rgb_teacher", resumed.rgb_teacher), ("ir_teacher", resumed.ir_teacher)):
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, checkpoint[key][name]))
            with mock.patch("training.teachers.initialize_teachers", side_effect=AssertionError("resumed teacher was reset")):
                self.assertEqual([resumed.train_one_iteration()["domain"] for _ in range(47)], expected_domains)

    def test_resume_rejects_changed_bridge_policy_before_mutating_model(self):
        with tempfile.TemporaryDirectory() as directory:
            original = make_trainer(bridge_config(directory))
            for _ in range(23):
                original.train_one_iteration()
            original.save_checkpoint("bridge.pth")
            changed_configs = []
            for field, value in (
                ("source_sup_weight", 0.25),
                ("saga_student_alphas", (1, 0.5, 0.5, 0.5, 0.25)),
                ("saga_teachers", ("none", "ir", "ir", "ir", "ir")),
            ):
                config = bridge_config(directory)
                setattr(config.gan_bridge, field, value)
                changed_configs.append(config)
            schedule_config = bridge_config(directory, bridge_schedule=configs.BridgeScheduleConfig(
                phase_fractions=(0.1, 0.3, 0.5, 0.7, 1)
            ))
            changed_configs.append(schedule_config)
            for config in changed_configs:
                with self.subTest(settings=config.gan_bridge):
                    resumed = make_trainer(config)
                    before = copy.deepcopy(resumed.student.state_dict())
                    with self.assertRaisesRegex(ValueError, "bridge|schedule"):
                        resumed.load_checkpoint(str(Path(directory) / "bridge.pth"))
                    self.assertEqual(resumed.global_step, 0)
                    for name, value in resumed.student.state_dict().items():
                        self.assertTrue(torch.equal(value, before[name]))

    def test_generator_identity_and_scheduler_progress_are_checked_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(bridge_config(directory))
            for _ in range(23):
                trainer.train_one_iteration()
            trainer.save_checkpoint("bridge.pth")
            path = Path(directory) / "bridge.pth"
            different = make_trainer(bridge_config(directory))
            different.gan_translator.identity["sha256"] = "other-weights"
            with self.assertRaisesRegex(ValueError, "GAN|generator|bridge"):
                different.load_checkpoint(str(path))
            checkpoint = torch.load(path, weights_only=False)
            for mutate in (lambda c: c.pop("scheduler"), lambda c: c.update(global_step=24),
                           lambda c: c.update(ema_initialized=False)):
                broken = copy.deepcopy(checkpoint)
                mutate(broken)
                torch.save(broken, path)
                fresh = make_trainer(bridge_config(directory))
                before = fresh.student.adapter.weight.detach().clone()
                with self.assertRaises(ValueError):
                    fresh.load_checkpoint(str(path))
                self.assertTrue(torch.equal(fresh.student.adapter.weight, before))

    def test_identical_generator_at_new_path_can_resume_but_legacy_workflow_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(bridge_config(directory))
            trainer.train_one_iteration()
            trainer.save_checkpoint("bridge.pth")
            config = bridge_config(directory)
            config.gan_bridge.checkpoint = "moved-identical-weights.pth"
            moved = make_trainer(config)
            moved.load_checkpoint(str(Path(directory) / "bridge.pth"))
            self.assertEqual(moved.global_step, 1)
            legacy = make_trainer(configs.TrainingConfig(device="cpu", output_dir=directory))
            with self.assertRaisesRegex(ValueError, "workflow|bridge"):
                legacy.load_checkpoint(str(Path(directory) / "bridge.pth"))

    def test_log_exposes_domain_stage_ratio_and_ema_target(self):
        trainer = make_trainer(bridge_config())
        for _ in range(25):
            item = trainer.train_one_iteration()
        with self.assertLogs("trainer", level="INFO") as captured:
            trainer._log(item)
        message = captured.output[0]
        self.assertIn("Stage: 2/5", message)
        self.assertIn(f"Domain: {item['domain']}", message)
        self.assertIn("Ratio: 5:3:2:0", message)
        self.assertIn(f"EMA: {item['ema_target']}", message)

    def test_all_five_stages_train_and_keep_gt_on_source_derived_domains(self):
        trainer = make_trainer(bridge_config())
        logs = []
        for iteration in range(100):
            before = trainer.student.adapter.weight.detach().clone()
            item = trainer.train_one_iteration()
            self.assertTrue(torch.isfinite(torch.tensor(item["total_loss"])))
            if iteration < 20:
                self.assertFalse(trainer.ema_initialized)
                self.assertEqual(item["ema_target"], "none")
                self.assertFalse(any(key.startswith("loss_kd") for key in item))
            else:
                self.assertTrue(trainer.ema_initialized)
                expected_teacher = "rgb" if item["domain"] == "rgb" or (item["domain"] == "saga" and iteration < 40) else "ir"
                self.assertEqual(item["ema_target"], expected_teacher)
                self.assertTrue(any(key.endswith("_" + expected_teacher) and "loss_kd" in key for key in item))
            if item["domain"] == "ir_real":
                self.assertFalse(any(key.startswith("sup_") for key in item))
                self.assertEqual(item["source_domain"], "ir")
            else:
                self.assertGreater(item["sup_loss_cls"], 0)
                self.assertFalse(torch.equal(before, trainer.student.adapter.weight))
                self.assertEqual(item["source_domain"], "rgb")
            logs.append(item)
        expected = [(20, 0, 0, 0), (10, 6, 4, 0), (6, 6, 6, 2), (4, 4, 8, 4), (2, 2, 4, 12)]
        for stage in range(5):
            counts = Counter(item["domain"] for item in logs[stage * 20:(stage + 1) * 20])
            self.assertEqual(tuple(counts[d] for d in ("rgb", "saga", "ir_gan", "ir_real")), expected[stage])
        self.assertEqual(len(trainer.gan_translator.generator.inputs), 22)
        self.assertTrue(all(parameter.grad is None for parameter in trainer.gan_translator.parameters()))
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        self.assertTrue(optimized.isdisjoint(id(p) for p in trainer.gan_translator.parameters()))

    def test_phase_mapping_and_ema_start_follow_bridge_durations(self):
        config = bridge_config(bridge_schedule=configs.BridgeScheduleConfig(phase_fractions=(0.1, 0.3, 0.5, 0.7, 1.0)))
        self.assertEqual(config.ema.start_steps, 10)
        self.assertEqual(config.optimizer.warmup_iters, 10)
        self.assertEqual([config.get_phase(i) for i in (0, 10, 30, 50, 70, 100)], [
            configs.Phase.PHASE1_RGB_WARMUP, configs.Phase.PHASE2_TRANSITION,
            configs.Phase.PHASE3_ADAPTATION, configs.Phase.PHASE3_ADAPTATION,
            configs.Phase.PHASE4_IR_FOCUS, configs.Phase.PHASE4_IR_FOCUS,
        ])
        with self.assertRaises(ValueError):
            bridge_config(ema=configs.EMAConfig(start_steps=30))
        with self.assertRaises(ValueError):
            bridge_config(teacher_mode="rgb")

    def test_cli_sets_the_bridge_schedule_and_saga_policy(self):
        from example_flir import parse_args, build_training_config

        argv = ["example_flir.py", "--data_root", "synthetic", "--workflow", "gan_bridge",
                "--gan-weights", "G_A.pth", "--total_iters", "100",
                "--bridge-phase-fractions", "0.1", "0.3", "0.5", "0.7", "1",
                "--bridge-ratios", "10:0:0:0,5:3:2:0,3:3:3:1,2:2:4:2,1:1:2:6",
                "--bridge-source-sup-weight", "0.75",
                "--bridge-saga-student-alphas", "1", "0.8", "0.6", "0.4", "0.2"]
        with mock.patch.object(sys, "argv", argv):
            config = build_training_config(parse_args())
        self.assertEqual(config.gan_bridge.checkpoint, "G_A.pth")
        self.assertEqual(config.ema.start_steps, 10)
        self.assertEqual(config.gan_bridge.saga_student_alphas, (1, 0.8, 0.6, 0.4, 0.2))
        self.assertEqual(config.gan_bridge.source_sup_weight, 0.75)

    def test_invalid_bridge_settings_fail_early(self):
        for kwargs in ({"source_sup_weight": 0}, {"saga_student_alphas": (1, 0.5)},
                       {"saga_teacher_alphas": (1, 0.5, float("nan"), 0.5, 0.5)},
                       {"saga_teachers": ("none", "rgb", "bad", "ir", "ir")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                configs.GANBridgeConfig(**kwargs)
        with self.assertRaises(ValueError):
            configs.TrainingConfig(workflow="gan_bridge")

    def test_bootstrap_injects_frozen_gan_without_adding_it_to_optimizer(self):
        from training.bootstrap import build_training
        from training.gan_bridge import FrozenGANTranslator
        from test_gan_bridge import FirstChannelGenerator

        config = bridge_config()
        student = D3TWrapper(TinyAdapter(), TinyCriterion())
        loader = DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb)
        translator = FrozenGANTranslator(FirstChannelGenerator(), identity={"sha256": "fixture"})
        with mock.patch("training.bootstrap.build_dataloaders", return_value=dict.fromkeys(
            ("rgb_train", "ir_train", "rgb_val", "ir_val"), loader
        )), mock.patch("training.bootstrap.build_fcos_d3t_trio", return_value=(student, copy.deepcopy(student), copy.deepcopy(student))), mock.patch(
            "training.bootstrap.load_gan_translator", return_value=translator
        ):
            trainer = build_training(config, logger=logging.getLogger(__name__))
        self.assertIs(trainer.gan_translator, translator)
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        self.assertTrue(optimized.isdisjoint(id(p) for p in translator.parameters()))


if __name__ == "__main__":
    unittest.main()

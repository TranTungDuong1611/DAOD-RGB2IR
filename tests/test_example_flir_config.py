import sys
import unittest
from unittest import mock

import torch
from torch import nn

from example_flir import _move_model_trio_to_device, make_training_config, parse_args
from config import FCOSModelConfig
from models.torchvision_fcos_adapter import ClassificationInitMode


class EntrypointConfigTests(unittest.TestCase):
    def test_default_hm_settings_match_d3t(self):
        with mock.patch.object(
            sys, "argv", ["example_flir.py", "--data_root", "synthetic"]
        ):
            config = make_training_config(parse_args())

        self.assertEqual(config.distill.hm_alpha, 0.5)
        self.assertEqual(config.distill.hm_beta, 0.5)
        self.assertEqual(config.distill.un_regular_alpha, 1.0)

    def test_cli_can_override_hm_settings(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "example_flir.py",
                "--data_root",
                "synthetic",
                "--hm-alpha",
                "0.25",
                "--hm-beta",
                "0.75",
                "--un-regular-alpha",
                "2.5",
            ],
        ):
            config = make_training_config(parse_args())

        self.assertEqual(config.distill.hm_alpha, 0.25)
        self.assertEqual(config.distill.hm_beta, 0.75)
        self.assertEqual(config.distill.un_regular_alpha, 2.5)

    def test_default_ema_start_matches_phase_two_start(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "example_flir.py",
                "--data_root",
                "synthetic",
                "--total_iters",
                "100",
            ],
        ):
            config = make_training_config(parse_args())

        self.assertEqual(config.curriculum.phase1_end, 20)
        self.assertEqual(config.ema.start_steps, config.curriculum.phase1_end)

    def test_explicit_ema_start_overrides_phase_boundary_default(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "example_flir.py",
                "--data_root",
                "synthetic",
                "--total_iters",
                "100",
                "--ema-start",
                "35",
            ],
        ):
            config = make_training_config(parse_args())

        self.assertEqual(config.ema.start_steps, 35)

    def test_default_score_threshold_matches_d3t_fcos_evaluation(self):
        with mock.patch.object(
            sys, "argv", ["example_flir.py", "--data_root", "synthetic"]
        ):
            config = make_training_config(parse_args())

        self.assertEqual(config.model.score_thresh, 0.05)
        self.assertEqual(FCOSModelConfig().score_thresh, 0.05)

    def test_move_model_trio_accepts_a_missing_teacher(self):
        student = nn.Linear(1, 1)
        rgb_teacher = nn.Linear(1, 1)

        moved = _move_model_trio_to_device(
            student, rgb_teacher, None, torch.device("cpu")
        )

        self.assertIs(moved[0], student)
        self.assertIs(moved[1], rgb_teacher)
        self.assertIsNone(moved[2])
        self.assertEqual(next(student.parameters()).device.type, "cpu")

    def test_cli_values_are_reflected_in_effective_config(self):
        argv = [
            "example_flir.py",
            "--data_root", "synthetic",
            "--output_dir", "run-output",
            "--total_iters", "17",
            "--batch_size", "2",
            "--eval_batch_size", "3",
            "--workers", "0",
            "--lr", "0.002",
            "--warmup-iters", "5",
            "--warmup-factor", "0.1",
            "--lr-steps", "8", "12",
            "--lr-gamma", "0.2",
            "--min_size", "96",
            "--max_size", "128",
            "--eval_every", "7",
            "--weights", "none",
            "--classification-init", "random_head",
            "--workflow", "rgb_baseline",
            "--teacher-mode", "rgb",
            "--device", "cpu",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        config = make_training_config(args)

        self.assertEqual(config.total_iters, 17)
        self.assertEqual(config.max_iter, 17)
        self.assertEqual(config.loader.train.batch_size, 2)
        self.assertEqual(config.loader.eval.batch_size, 3)
        self.assertEqual(config.loader.train.num_workers, 0)
        self.assertEqual(config.optimizer.base_lr, 0.002)
        self.assertEqual(config.optimizer.warmup_iters, 5)
        self.assertEqual(config.optimizer.warmup_factor, 0.1)
        self.assertEqual(config.optimizer.milestones, (8, 12))
        self.assertEqual(config.optimizer.gamma, 0.2)
        self.assertEqual(config.eval_period, 7)
        self.assertEqual(config.model.min_size, 96)
        self.assertEqual(config.model.max_size, 128)
        self.assertIsNone(config.model.weights)
        self.assertFalse(config.model.pretrained_backbone)
        self.assertEqual(
            config.model.classification_init_mode,
            ClassificationInitMode.RANDOM_HEAD,
        )
        self.assertEqual(config.workflow, "rgb_baseline")
        self.assertEqual(config.teacher_mode, "rgb")


if __name__ == "__main__":
    unittest.main()

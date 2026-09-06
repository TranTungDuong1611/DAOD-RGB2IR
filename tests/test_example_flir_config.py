import sys
import unittest
from unittest import mock

from example_flir import make_training_config, parse_args
from models.torchvision_fcos_adapter import ClassificationInitMode


class EntrypointConfigTests(unittest.TestCase):
    def test_cli_values_are_reflected_in_effective_config(self):
        argv = [
            "example_flir.py",
            "--data_root", "synthetic",
            "--output_dir", "run-output",
            "--total_iters", "17",
            "--batch_size", "2",
            "--eval_batch_size", "3",
            "--workers", "0",
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

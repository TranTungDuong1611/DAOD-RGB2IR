"""Checkpoint files restore training state without experiment validation."""

from pathlib import Path
import tempfile
import unittest

import torch

from test_curriculum_wrapper_integration import (
    StatefulLRScheduler, StatefulPhaseEvaluator, build_config, build_trainer,
)


class CheckpointingTests(unittest.TestCase):
    def test_save_keeps_only_model_metadata_and_complete_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            config = build_config(output_dir=directory)
            trainer, _, _, _ = build_trainer(config)
            trainer.lr_scheduler = StatefulLRScheduler()
            trainer.phase_evaluator = StatefulPhaseEvaluator({"history": [{"global_step": 3}]})
            for _ in range(3):
                trainer.train_one_iteration()
            trainer.best_map = .61
            trainer.best_rgb_map = .72
            trainer.last_pseudo_label_eval_step = 3
            trainer.last_full_pseudo_label_eval_step = 2
            trainer.save_checkpoint("run.pth")

            payload = torch.load(Path(directory) / "run.pth", weights_only=True)
            self.assertEqual(set(payload["metadata"]), {"model"})
            self.assertEqual(payload["metadata"]["model"]["class_names"], ("person", "car", "bicycle"))
            self.assertEqual(payload["metadata"]["model"]["classification_init_mode"], "coco_tower")
            resumed, _, _, _ = build_trainer(config)
            resumed.lr_scheduler = StatefulLRScheduler()
            resumed.phase_evaluator = StatefulPhaseEvaluator()
            resumed.load_checkpoint(str(Path(directory) / "run.pth"))

            self.assertEqual(resumed.global_step, 3)
            self.assertEqual(resumed.best_map, .61)
            self.assertEqual(resumed.best_rgb_map, .72)
            self.assertTrue(resumed.ema_initialized)
            self.assertEqual(resumed.last_pseudo_label_eval_step, 3)
            self.assertEqual(resumed.last_full_pseudo_label_eval_step, 2)
            self.assertEqual(resumed.lr_scheduler.steps, 3)
            self.assertEqual(resumed.phase_evaluator.state, {"history": [{"global_step": 3}]})
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())
            for name in ("student", "rgb_teacher", "ir_teacher"):
                for key, value in getattr(resumed, name).state_dict().items():
                    self.assertTrue(torch.equal(value, payload[name][key]))
            for actual, expected in zip(resumed.optimizer.param_groups, payload["optimizer"]["param_groups"]):
                self.assertEqual(actual["lr"], expected["lr"])
            self.assertEqual(resumed.train_one_iteration(), trainer.train_one_iteration())

    def test_resume_does_not_require_metadata_or_old_audit_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            config = build_config(output_dir=directory)
            trainer, _, _, _ = build_trainer(config)
            for _ in range(3):
                trainer.train_one_iteration()
            trainer.save_checkpoint("old.pth")
            path = Path(directory) / "old.pth"
            payload = torch.load(path, weights_only=False)
            for name in ("metadata", "last_pseudo_label_eval_step", "last_full_pseudo_label_eval_step", "pseudo_label_evaluator"):
                payload.pop(name)
            torch.save(payload, path)

            resumed, _, _, _ = build_trainer(config)
            resumed.load_checkpoint(str(path))

            self.assertEqual(resumed.global_step, 3)
            self.assertEqual(resumed.last_pseudo_label_eval_step, 0)
            self.assertEqual(resumed.last_full_pseudo_label_eval_step, 0)
            self.assertEqual(resumed.train_one_iteration(), trainer.train_one_iteration())


if __name__ == "__main__":
    unittest.main()

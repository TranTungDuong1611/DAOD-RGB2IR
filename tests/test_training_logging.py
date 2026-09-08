import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from trainer import CurriculumDomainAdaptationTrainer


class TrainingLoggingTests(unittest.TestCase):
    def test_configure_writes_child_logger_to_output_logs(self):
        from training_logging import configure_training_logging

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                log_path = configure_training_logging(output_dir)
                logging.getLogger("trainer.child").info(
                    "persisted training message"
                )
                for handler in logging.getLogger().handlers:
                    handler.flush()

                self.assertEqual(log_path.parent, Path(output_dir) / "logs")
                self.assertTrue(log_path.is_file())
                self.assertIn(
                    "persisted training message",
                    log_path.read_text(encoding="utf-8"),
                )
            finally:
                self._remove_training_handlers()

    def test_iteration_log_contains_training_context_and_timing(self):
        trainer = SimpleNamespace(global_step=50)
        values = {
            "global_step": 50,
            "phase": "phase2_near_rgb",
            "step_type": "rgb_supervised",
            "total_loss": 1.25,
            "sup_loss_cls": 0.75,
            "lr": 0.004,
            "iter_time": 0.125,
        }

        with self.assertLogs("trainer", level=logging.INFO) as captured:
            CurriculumDomainAdaptationTrainer._log(trainer, values)

        message = captured.output[0]
        self.assertIn("[000050]", message)
        self.assertIn("Phase: phase2_near_rgb", message)
        self.assertIn("Route: rgb_supervised", message)
        self.assertIn("Loss: 1.2500", message)
        self.assertIn("LR: 0.004", message)
        self.assertIn("Time: 0.125s", message)
        self.assertIn("sup_loss_cls=0.7500", message)

    def test_reconfigure_closes_the_previous_run_file(self):
        from training_logging import configure_training_logging

        with tempfile.TemporaryDirectory() as output_dir:
            try:
                previous_path = configure_training_logging(output_dir)
                current_path = configure_training_logging(output_dir)

                previous_path.unlink()

                self.assertFalse(previous_path.exists())
                self.assertTrue(current_path.is_file())
                owned_handlers = [
                    handler
                    for handler in logging.getLogger().handlers
                    if getattr(handler, "_d3t_training_handler", False)
                ]
                self.assertEqual(len(owned_handlers), 2)
            finally:
                self._remove_training_handlers()

    @staticmethod
    def _remove_training_handlers():
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_d3t_training_handler", False):
                root.removeHandler(handler)
                handler.close()


if __name__ == "__main__":
    unittest.main()

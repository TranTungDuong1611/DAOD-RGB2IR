"""Exercise trainer-owned best metrics through real evaluation and checkpoint files."""

import logging
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch.utils.data import DataLoader

from config import CurriculumConfig, Phase
from models.detector_wrapper import DetectorWrapper
from training.bootstrap import build_training
from test_curriculum_wrapper_integration import (
    IRDataset, RGBDataset, TinyAdapter, TinyCriterion, build_config, collate_ir, collate_rgb,
)


class EvaluationAdapter(TinyAdapter):
    def postprocess(self, output):
        level = self.weight.item()
        boxes = [[9., 9., 13., 13.]]
        scores = [.9]
        if level >= 2:
            boxes, scores = [[2., 2., 8., 8.]], [.8]
        elif level >= 1:
            boxes.append([2., 2., 8., 8.])
            scores.append(.8)
        return [{
            "boxes": self.weight.new_tensor(boxes),
            "scores": self.weight.new_tensor(scores),
            "labels": torch.zeros(len(boxes), dtype=torch.long, device=self.weight.device),
        } for _ in output.predictions]


def build_evaluation_trainer(directory):
    config = build_config(
        output_dir=str(directory), total_iters=1,
        curriculum=CurriculumConfig(phase1_end=10, phase2_end=20, phase3_end=30),
    )
    config.eval_period = 1
    student = DetectorWrapper(EvaluationAdapter(), TinyCriterion())
    with torch.no_grad():
        student.adapter.weight.fill_(3.)
    rgb = DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb)
    ir = DataLoader(IRDataset(), batch_size=1, collate_fn=collate_ir)
    loaders = {"rgb_train": rgb, "ir_train": ir, "rgb_val": rgb, "ir_val": rgb}
    with (
        mock.patch("training.bootstrap.build_dataloaders", return_value=loaders),
        mock.patch("models.fcos_models.build_fcos_model", return_value=student),
    ):
        return build_training(config, logger=logging.getLogger(__name__))


class TrainingEvaluationTests(unittest.TestCase):
    def test_internal_forward_type_error_is_not_retried(self):
        from evaluator import DetectionEvaluator, PhaseEvaluator

        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images, sample_ids=None):
                self.calls += 1
                raise TypeError("broken forward implementation")

        model = BrokenModel()
        loader = DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb)
        evaluator = PhaseEvaluator(DetectionEvaluator(3), loader, torch.device("cpu"))
        with self.assertRaisesRegex(TypeError, "broken forward implementation"):
            evaluator.evaluate(model, 1, Phase.PHASE1_RGB_WARMUP)
        self.assertEqual(model.calls, 1)
        self.assertEqual(evaluator.history, [])

    def test_scheduled_best_checkpoint_contains_completed_evaluation_and_current_best(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_evaluation_trainer(directory)
            trainer.train()
            payload = torch.load(Path(directory) / "best_model.pth", weights_only=False)

            self.assertEqual(payload["best_map"], 1.)
            self.assertEqual(payload["best_rgb_map"], 1.)
            state = payload["phase_evaluator"]
            self.assertEqual(set(state), {"last_phase", "history"})
            self.assertEqual(state["last_phase"], Phase.PHASE1_RGB_WARMUP)
            self.assertEqual(len(state["history"]), 1)
            result = state["history"][0]
            self.assertEqual(result["rgb_mAP@0.5"], 1.)
            self.assertTrue(result["is_best_ir"])
            self.assertTrue(result["is_best_rgb"])
            self.assertEqual(result["global_step"], payload["global_step"])
            for name, value in trainer.student.state_dict().items():
                self.assertTrue(torch.equal(payload["student"][name], value))
            self.assertTrue(trainer.student.training)

    def test_only_strict_ir_improvement_replaces_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_evaluation_trainer(directory)
            for step, (level, best, saved_history) in enumerate(((1, .5, 1), (1, .5, 1), (0, .5, 1), (2, 1., 4))):
                with self.subTest(step=step):
                    trainer.global_step = step
                    with torch.no_grad():
                        trainer.student.adapter.weight.fill_(level)
                    result = trainer.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP)
                    payload = torch.load(Path(directory) / "best_model.pth", weights_only=False)
                    self.assertEqual(trainer.best_map, best)
                    self.assertEqual(trainer.best_rgb_map, best)
                    self.assertEqual(payload["best_map"], best)
                    self.assertEqual(len(payload["phase_evaluator"]["history"]), saved_history)
                    self.assertEqual(result.get("is_best_ir", False), step in (0, 3))
                    self.assertEqual(result.get("is_best_rgb", False), step in (0, 3))
                    self.assertTrue(trainer.student.training)

    def test_initial_zero_map_is_still_recorded_as_first_best(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = build_evaluation_trainer(directory)
            with torch.no_grad():
                trainer.student.adapter.weight.zero_()
            trainer.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP, trigger_reason="initial")
            payload = torch.load(Path(directory) / "best_model.pth", weights_only=False)
            self.assertEqual(payload["best_map"], 0.)
            self.assertEqual(payload["phase_evaluator"]["history"][0]["trigger"], "initial")

    def test_failed_rgb_evaluation_does_not_publish_partial_best(self):
        class FailingLoader:
            def __iter__(self):
                raise RuntimeError("RGB evaluation failed")

        with tempfile.TemporaryDirectory() as directory:
            trainer = build_evaluation_trainer(directory)
            trainer.phase_evaluator.rgb_val_loader = FailingLoader()
            with self.assertRaisesRegex(RuntimeError, "RGB evaluation failed"):
                trainer.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP)
            self.assertEqual(trainer.best_map, -1.)
            self.assertEqual(trainer.best_rgb_map, -1.)
            self.assertEqual(trainer.phase_evaluator.history, [])
            self.assertFalse((Path(directory) / "best_model.pth").exists())
            self.assertTrue(trainer.student.training)

    def test_resume_preserves_best_metrics_and_evaluation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = build_evaluation_trainer(root / "original")
            with torch.no_grad():
                original.student.adapter.weight.fill_(1.)
            original.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP)

            resumed = build_evaluation_trainer(root / "resumed")
            resumed.load_checkpoint(str(root / "original" / "best_model.pth"))
            self.assertEqual(resumed.phase_evaluator.history, original.phase_evaluator.history)
            with torch.no_grad():
                resumed.student.adapter.weight.zero_()
            resumed.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP)
            self.assertEqual(resumed.best_map, .5)
            self.assertEqual(resumed.best_rgb_map, .5)
            self.assertFalse((root / "resumed" / "best_model.pth").exists())

    def test_legacy_checkpoint_merges_best_metrics_into_trainer(self):
        for old_best, expected in ((.1, .6), (.8, .8)):
            with self.subTest(old_best=old_best), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = build_evaluation_trainer(root / "original")
                original.save_checkpoint("legacy.pth")
                path = root / "original" / "legacy.pth"
                payload = torch.load(path, weights_only=False)
                payload["best_map"] = old_best
                payload.pop("best_rgb_map", None)
                payload["phase_evaluator"].update(best_ir_map=.6, best_rgb_map=.7)
                torch.save(payload, path)

                resumed = build_evaluation_trainer(root / "resumed")
                resumed.load_checkpoint(str(path))
                self.assertEqual(resumed.best_map, expected)
                self.assertEqual(resumed.best_rgb_map, .7)
                self.assertEqual(set(resumed.phase_evaluator.state_dict()), {"last_phase", "history"})
                with torch.no_grad():
                    resumed.student.adapter.weight.fill_(1.)
                result = resumed.evaluate(current_phase=Phase.PHASE1_RGB_WARMUP)
                self.assertNotIn("is_best_ir", result)
                self.assertNotIn("is_best_rgb", result)
                self.assertFalse((root / "resumed" / "best_model.pth").exists())


if __name__ == "__main__":
    unittest.main()

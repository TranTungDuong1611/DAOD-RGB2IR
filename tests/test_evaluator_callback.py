import unittest

import torch
from torch import nn

from evaluator import DetectionEvaluator, PhaseEvaluator
from config import Phase


class EvaluatorCallbackTests(unittest.TestCase):
    def test_loader_domain_is_forwarded_to_the_model(self):
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.domains = []

            def forward(self, images, sample_ids=None, domain="rgb"):
                self.domains.append(domain)
                return [{} for _ in images]

        class EvaluatorStub:
            def reset(self):
                pass

            def update(self, predictions, targets):
                pass

            def compute(self):
                return {"mAP@0.5": 0.0}

        model = RecordingModel()
        phase_evaluator = PhaseEvaluator(
            evaluator=EvaluatorStub(),
            ir_val_loader=[],
            device=torch.device("cpu"),
        )
        batch = (
            torch.zeros(1, 3, 8, 8),
            [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}],
            ("ir-1",),
        )

        phase_evaluator._run_eval_on_loader(model, [batch], "IR")

        self.assertEqual(model.domains, ["ir"])

    def test_best_callback_runs_only_on_strict_ir_improvement(self):
        model = nn.Linear(1, 1)
        model.train()
        phase_evaluator = PhaseEvaluator(
            evaluator=DetectionEvaluator(num_classes=1),
            ir_val_loader=[object()],
            device=torch.device("cpu"),
        )
        values = iter((0.5, 0.5, 0.4, 0.6))
        phase_evaluator._run_eval_on_loader = (
            lambda model, loader, domain: {"mAP@0.5": next(values)}
        )
        calls = []
        phase_evaluator.register_best_fn(lambda result: calls.append(result["mAP@0.5"]))

        for step in range(4):
            result = phase_evaluator.evaluate(
                model, step, Phase.PHASE1_RGB_WARMUP
            )
            self.assertIn("mAP@0.5", result)

        self.assertEqual(calls, [0.5, 0.6])
        self.assertTrue(model.training)
        self.assertEqual(phase_evaluator.best_ir_map, 0.6)

    def test_state_roundtrip_preserves_best_metrics_and_prevents_false_best(self):
        model = nn.Linear(1, 1)
        original = PhaseEvaluator(
            evaluator=DetectionEvaluator(num_classes=1),
            ir_val_loader=[object()],
            device=torch.device("cpu"),
        )
        original.best_ir_map = 0.6
        original.best_rgb_map = 0.7
        original._last_phase = Phase.PHASE2_TRANSITION
        original.history = [{"global_step": 10, "mAP@0.5": 0.6}]

        resumed = PhaseEvaluator(
            evaluator=DetectionEvaluator(num_classes=1),
            ir_val_loader=[object()],
            device=torch.device("cpu"),
        )
        resumed.load_state_dict(original.state_dict())
        resumed._run_eval_on_loader = (
            lambda model, loader, domain: {"mAP@0.5": 0.5}
        )
        calls = []
        resumed.register_best_fn(lambda result: calls.append(result))

        result = resumed.evaluate(model, 11, Phase.PHASE2_TRANSITION)

        self.assertNotIn("is_best_ir", result)
        self.assertEqual(calls, [])
        self.assertEqual(resumed.best_ir_map, 0.6)
        self.assertEqual(resumed.best_rgb_map, 0.7)
        self.assertEqual(resumed._last_phase, Phase.PHASE2_TRANSITION)
        self.assertEqual(resumed.history[0]["global_step"], 10)


if __name__ == "__main__":
    unittest.main()

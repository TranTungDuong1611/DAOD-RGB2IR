import unittest

import torch
from torch import nn

from evaluator import DetectionEvaluator, PhaseEvaluator
from config import Phase


class EvaluatorCallbackTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

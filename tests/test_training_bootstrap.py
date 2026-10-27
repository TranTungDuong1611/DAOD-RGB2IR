"""Verify assembled training dependencies and the best-checkpoint callback."""

import copy
import logging
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import FCOSModelConfig, TrainingConfig
from training.bootstrap import build_training


class TrainingBootstrapTests(unittest.TestCase):
    def _build(self, teacher_mode):
        config = TrainingConfig(
            model=FCOSModelConfig(weights=None, pretrained_backbone=False),
            device="cpu", teacher_mode=teacher_mode,
        )
        student = nn.Linear(1, 1)
        rgb_teacher = copy.deepcopy(student) if teacher_mode == "two_teacher" else None
        ir_teacher = copy.deepcopy(student)
        loader = DataLoader(TensorDataset(torch.zeros(2, 3, 8, 8)))
        loaders = dict.fromkeys(("rgb_train", "ir_train", "rgb_val", "ir_val"), loader)
        # Replace external data/model creation while exercising the real assembly.
        with (
            mock.patch("training.bootstrap.build_dataloaders", return_value=loaders),
            mock.patch(
                "training.bootstrap.build_fcos_d3t_trio",
                return_value=(student, rgb_teacher, ir_teacher),
            ),
        ):
            trainer = build_training(config, logger=logging.getLogger(__name__))
        return trainer, student, rgb_teacher, ir_teacher

    def test_bootstrap_wires_models_optimizer_and_evaluator(self):
        trainer, student, rgb_teacher, ir_teacher = self._build("two_teacher")

        self.assertIs(trainer.student, student)
        self.assertIs(trainer.rgb_teacher, rgb_teacher)
        self.assertIs(trainer.ir_teacher, ir_teacher)
        self.assertIs(trainer.lr_scheduler.optimizer, trainer.optimizer)
        self.assertEqual(
            {id(p) for group in trainer.optimizer.param_groups for p in group["params"]},
            {id(p) for p in student.parameters() if p.requires_grad},
        )
        self.assertFalse(ir_teacher.training)
        self.assertTrue(all(not p.requires_grad for p in ir_teacher.parameters()))
        self.assertEqual(trainer.phase_evaluator.evaluator.interp, "voc11")
        self.assertEqual(trainer.phase_evaluator.evaluator.iou_thresholds, [0.5])

    def test_ir_only_bootstrap_keeps_best_checkpoint_callback(self):
        trainer, _, _, _ = self._build("ir")

        self.assertIsNone(trainer.rgb_teacher)
        with mock.patch.object(trainer, "save_checkpoint") as save:
            trainer.phase_evaluator.on_new_best_fn({"mAP@0.5": 0.6})
        save.assert_called_once_with("best_model.pth")


if __name__ == "__main__":
    unittest.main()

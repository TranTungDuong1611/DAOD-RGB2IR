import math
import unittest

import torch

from loss.d3t_criterion import D3TLossCriterion
from models.d3t_adapter import (
    DistillationPair,
    DistillationSettings,
    Predictions,
    SupervisedBatch,
)


def supervised_batch(class_logits, boxes, quality_logits, class_targets, box_targets,
                     quality_targets, foreground):
    prediction = Predictions(class_logits, boxes, quality_logits)
    return SupervisedBatch((prediction,), ({
        'class_targets': class_targets,
        'box_targets': box_targets,
        'quality_targets': quality_targets,
        'foreground': foreground,
    },))


class D3TCriterionTests(unittest.TestCase):
    def test_supervised_reduction_matches_literal_hm_focal_and_quality(self):
        criterion = D3TLossCriterion(alpha=0.75, gamma=2.0, weight_type='iou')
        batch = supervised_batch(
            class_logits=torch.tensor([[0.0], [0.0]], requires_grad=True),
            boxes=torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]],
                               requires_grad=True),
            quality_logits=torch.tensor([0.0, 0.0], requires_grad=True),
            class_targets=torch.tensor([[0.5], [0.0]]),
            box_targets=torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]]),
            quality_targets=torch.tensor([0.5, 0.0]),
            foreground=torch.tensor([True, False]),
        )
        losses = criterion.supervised(batch)
        bce = math.log(2.0)
        expected_cls = bce * 0.5 + bce * (0.75 * 0.5 ** 2)
        self.assertAlmostEqual(losses['loss_cls'].item(), expected_cls, places=6)
        self.assertAlmostEqual(losses['loss_box'].item(), 0.0, places=6)
        self.assertAlmostEqual(losses['loss_quality'].item(), bce, places=6)
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(batch.predictions[0].class_logits.grad).all())

    def test_supervised_box_iou_loss_for_shifted_box(self):
        criterion = D3TLossCriterion()
        batch = supervised_batch(
            class_logits=torch.zeros(1, 1, requires_grad=True),
            boxes=torch.tensor([[1.0, 0.0, 3.0, 2.0]], requires_grad=True),
            quality_logits=torch.zeros(1, requires_grad=True),
            class_targets=torch.ones(1, 1),
            box_targets=torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
            quality_targets=torch.ones(1),
            foreground=torch.ones(1, dtype=torch.bool),
        )
        losses = criterion.supervised(batch)
        self.assertAlmostEqual(losses['loss_box'].item(), 2.0 / 3.0, places=6)

    def test_no_foreground_returns_finite_graph_connected_zeros(self):
        criterion = D3TLossCriterion()
        logits = torch.zeros(2, 2, requires_grad=True)
        boxes = torch.zeros(2, 4, requires_grad=True)
        quality = torch.zeros(2, requires_grad=True)
        losses = criterion.supervised(supervised_batch(
            logits, boxes, quality,
            torch.zeros(2, 2), torch.zeros(2, 4), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        ))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        (losses['loss_box'] + losses['loss_quality']).backward()
        self.assertEqual(boxes.grad.abs().sum().item(), 0.0)
        self.assertEqual(quality.grad.abs().sum().item(), 0.0)

    def test_distillation_selects_top_ratio_then_threshold_and_detaches_teacher(self):
        criterion = D3TLossCriterion()
        student_logits = torch.zeros(3, 1, requires_grad=True)
        student_boxes = torch.tensor([
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
        ], requires_grad=True)
        student_quality = torch.zeros(3, requires_grad=True)
        teacher_logits = torch.tensor([
            [math.log(9.0)], [math.log(4.0)], [math.log(1.0 / 9.0)]
        ], requires_grad=True)
        teacher_boxes = student_boxes.detach().clone()
        teacher_quality = torch.tensor([
            [math.log(9.0)], [math.log(4.0)], [0.0]
        ], requires_grad=True).flatten()
        pair = DistillationPair(
            (Predictions(student_logits, student_boxes, student_quality),),
            (Predictions(teacher_logits, teacher_boxes, teacher_quality),),
        )
        result = criterion.distillation(pair, DistillationSettings(
            top_ratio=0.5, min_hm=0.5, hm_alpha=1.0, hm_beta=1.0,
            uncertainty_alpha=4.0,
        ))
        self.assertEqual(result.metrics['kd_selected_count'].item(), 2.0)
        self.assertAlmostEqual(result.metrics['kd_selected_ratio'].item(), 2.0 / 3.0)
        self.assertGreater(result.metrics['kd_hm_sum'].item(), 1.0)
        total = sum(result.losses.values())
        total.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())
        self.assertEqual(student_logits.grad[2].item(), 0.0)
        self.assertIsNone(teacher_logits.grad)

    def test_distillation_empty_selection_returns_zero_without_student_gradient(self):
        criterion = D3TLossCriterion()
        student = Predictions(
            torch.zeros(2, 1, requires_grad=True),
            torch.zeros(2, 4, requires_grad=True),
            torch.zeros(2, requires_grad=True),
        )
        teacher = Predictions(
            torch.full((2, 1), -20.0),
            torch.zeros(2, 4),
            torch.full((2,), -20.0),
        )
        result = criterion.distillation(
            DistillationPair((student,), (teacher,)),
            DistillationSettings(top_ratio=1.0, min_hm=0.9),
        )
        self.assertTrue(all(value.item() == 0.0 for value in result.losses.values()))
        sum(result.losses.values()).backward()
        self.assertEqual(student.class_logits.grad.abs().sum().item(), 0.0)

    def test_teacher_predictions_are_not_mutated_or_connected(self):
        criterion = D3TLossCriterion()
        teacher_logits = torch.zeros(1, 1, requires_grad=True)
        result = criterion.distillation(
            DistillationPair(
                (Predictions(torch.ones(1, 1, requires_grad=True),
                             torch.ones(1, 4, requires_grad=True),
                             torch.ones(1, requires_grad=True)),),
                (Predictions(teacher_logits, torch.ones(1, 4), torch.zeros(1)),),
            ),
            DistillationSettings(top_ratio=1.0, min_hm=0.0),
        )
        sum(result.losses.values()).backward()
        self.assertIsNone(teacher_logits.grad)


if __name__ == '__main__':
    unittest.main()

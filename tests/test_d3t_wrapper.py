import unittest
import torch
from torch import nn
from models.d3t_adapter import (
    AdapterOutput,
    CriterionResult,
    D3TCriterion,
    DetectorAdapter,
    DistillationPair,
    DistillationSettings,
    Predictions,
    SupervisedBatch,
)
from models.d3t_wrapper import D3TWrapper


class TinyAdapter(DetectorAdapter):
    """Small differentiable detector exercising the public adapter contract."""
    def __init__(self, count=2):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.count = count
        self.forward_calls = 0

    def forward(self, images, targets=None, sample_ids=None):
        self.forward_calls += 1
        predictions = tuple(Predictions(
            self.weight.expand(self.count, 3),
            self.weight.expand(self.count, 4),
            self.weight.expand(self.count),
        ) for _ in images)
        return AdapterOutput(predictions, context=tuple(image.shape[-2:] for image in images))

    def prepare_supervised(self, output, targets):
        prepared = tuple({
            'class_targets': torch.zeros_like(p.class_logits),
            'box_targets': torch.zeros_like(p.boxes),
            'quality_targets': torch.zeros_like(p.quality_logits),
            'foreground': torch.zeros_like(p.quality_logits, dtype=torch.bool),
        } for p in output.predictions)
        return SupervisedBatch(output.predictions, prepared, {'rpn': self.weight.square()})

    def prepare_distillation(self, student, teacher):
        # This adapter only supports identical, corresponding prediction layouts.
        return DistillationPair(student.predictions, teacher.predictions)

    def postprocess(self, output):
        return [{'boxes': p.boxes, 'scores': p.class_logits.sigmoid().max(-1).values,
                 'labels': p.class_logits.argmax(-1)} for p in output.predictions]


class TinyCriterion(D3TCriterion):
    def supervised(self, batch):
        return {'classification': sum(p.class_logits.square().mean() for p in batch.predictions)}

    def distillation(self, pair, settings=None):
        return CriterionResult(
            losses={'kd': sum((s.class_logits - t.class_logits).square().mean()
                              for s, t in zip(pair.student, pair.teacher))},
            metrics={},
        )


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.images = [torch.zeros(3, 8, 8)]
        self.targets = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]

    def test_supervised_keeps_native_loss_and_gradient(self):
        model = D3TWrapper(TinyAdapter(), TinyCriterion())
        losses = model(self.images, self.targets)
        self.assertEqual(set(losses), {'classification', 'rpn'})
        sum(losses.values()).backward()
        self.assertEqual(model.adapter.weight.grad.item(), 4.0)

    def test_eval_uses_adapter_postprocessing(self):
        model = D3TWrapper(TinyAdapter(5), TinyCriterion()).eval()
        self.assertEqual(model(self.images)[0]['boxes'].shape, (5, 4))

    def test_requires_targets_in_training(self):
        with self.assertRaisesRegex(ValueError, 'targets'):
            D3TWrapper(TinyAdapter(), TinyCriterion())(self.images)

    def test_distillation_only_updates_student(self):
        student = D3TWrapper(TinyAdapter(), TinyCriterion())
        teacher = D3TWrapper(TinyAdapter(), TinyCriterion()).eval()
        with torch.no_grad():
            teacher.adapter.weight.fill_(2)
        losses = student.distill(
            self.images, teacher, self.images,
            sample_ids=('frame-1',), teacher_sample_ids=('frame-1',),
        )
        sum(losses.values()).backward()
        self.assertEqual(student.adapter.weight.grad.item(), -2.0)
        self.assertIsNone(teacher.adapter.weight.grad)
        self.assertFalse(teacher.training)

    def test_teacher_must_be_in_eval_mode(self):
        student = D3TWrapper(TinyAdapter(), TinyCriterion())
        with self.assertRaisesRegex(ValueError, 'eval'):
            student.distill(self.images, D3TWrapper(TinyAdapter(), TinyCriterion()), self.images)

    def test_different_prediction_counts_need_explicit_matching(self):
        student = D3TWrapper(TinyAdapter(2), TinyCriterion())
        teacher = D3TWrapper(TinyAdapter(3), TinyCriterion()).eval()
        with self.assertRaisesRegex(ValueError, 'matched'):
            student.distill(
                self.images, teacher, self.images,
                sample_ids=('frame-1',), teacher_sample_ids=('frame-1',),
            )

    def test_registered_parameters_and_checkpoint(self):
        model = D3TWrapper(TinyAdapter(), TinyCriterion()).double()
        self.assertEqual(dict(model.named_parameters())['adapter.weight'].dtype, torch.float64)
        copy = D3TWrapper(TinyAdapter(), TinyCriterion()).double()
        copy.load_state_dict(model.state_dict())
        self.assertEqual(copy.adapter.weight.item(), 1.0)

    def test_prediction_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'boxes'):
            Predictions(torch.zeros(2, 3), torch.zeros(3, 4), torch.zeros(2))

    def test_native_loss_collision_is_not_silent(self):
        class ConflictingCriterion(TinyCriterion):
            def supervised(self, batch):
                return {'rpn': batch.predictions[0].class_logits.mean()}
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            D3TWrapper(TinyAdapter(), ConflictingCriterion())(self.images, self.targets)

    def test_loss_from_existing_output_does_not_forward_again(self):
        adapter = TinyAdapter()
        model = D3TWrapper(adapter, TinyCriterion())
        output = model.raw(self.images, self.targets, sample_ids=('frame-1',))
        losses = model.supervised_from_output(output, self.targets)
        self.assertEqual(adapter.forward_calls, 1)
        self.assertIn('classification', losses)

    def test_distillation_settings_validate_range(self):
        with self.assertRaises(ValueError):
            DistillationSettings(top_ratio=0.0, min_hm=0.5)
        with self.assertRaises(ValueError):
            DistillationSettings(top_ratio=0.1, min_hm=1.1)

class ContractTests(unittest.TestCase):
    def test_supervised_target_rows_must_match_predictions(self):
        p = Predictions(torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2))
        with self.assertRaisesRegex(ValueError, 'class_targets'):
            SupervisedBatch((p,), ({'class_targets': torch.zeros(3, 3),
                'box_targets': torch.zeros(2, 4), 'quality_targets': torch.zeros(2),
                'foreground': torch.zeros(2, dtype=torch.bool)},))

    def test_empty_predictions_can_produce_differentiable_zero(self):
        model = D3TWrapper(TinyAdapter(0), TinyCriterion())
        teacher = D3TWrapper(TinyAdapter(0), TinyCriterion()).eval()
        class SumCriterion(TinyCriterion):
            def distillation(self, pair, settings=None):
                return CriterionResult(
                    losses={'kd': sum((s.class_logits - t.class_logits).square().sum()
                                      for s, t in zip(pair.student, pair.teacher))},
                    metrics={},
                )
        model.criterion = SumCriterion()
        loss = model.distill(
            [torch.zeros(3, 8, 8)], teacher, [torch.zeros(3, 8, 8)],
            sample_ids=('frame-1',), teacher_sample_ids=('frame-1',),
        )['kd']
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(model.adapter.weight.grad.item(), 0.0)

    def test_raw_rejects_empty_or_mismatched_batches(self):
        model = D3TWrapper(TinyAdapter(), TinyCriterion())
        with self.assertRaisesRegex(ValueError, 'images'):
            model.raw([])
        with self.assertRaisesRegex(ValueError, 'targets'):
            model.raw([torch.zeros(3, 8, 8)], [])


if __name__ == '__main__':
    unittest.main()

from collections import OrderedDict
import unittest

import torch
from torch import nn

from models.torchvision_fcos_adapter import (
    ClassificationInitMode,
    FCOSIoUHead,
    TorchvisionFCOSAdapter,
)
from torchvision.models.detection import FCOS


class TinyBackbone(nn.Module):
    out_channels = 64

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, self.out_channels, kernel_size=1)

    def forward(self, images):
        features = []
        current = self.proj(images)
        for level in range(5):
            size = max(1, images.shape[-1] // (8 * (2 ** level)))
            height = max(1, images.shape[-2] // (8 * (2 ** level)))
            features.append(nn.functional.adaptive_avg_pool2d(current, (height, size)))
        return OrderedDict((str(index), feature) for index, feature in enumerate(features))


def build_adapter(min_size=64, max_size=128):
    detector = FCOS(
        TinyBackbone(),
        num_classes=91,
        min_size=min_size,
        max_size=max_size,
        score_thresh=0.0,
        nms_thresh=1.0,
        detections_per_img=20,
        topk_candidates=20,
    )
    detector.head = FCOSIoUHead(
        detector.head,
        num_classes=3,
        classification_init_mode=ClassificationInitMode.COCO_TOWER,
    )
    return TorchvisionFCOSAdapter(detector)


class TorchvisionFCOSAdapterTests(unittest.TestCase):
    def test_raw_forward_supports_variable_list_and_decodes_with_box_coder(self):
        adapter = build_adapter()
        adapter.train()
        images = [torch.rand(3, 96, 128), torch.rand(3, 80, 112)]
        output = adapter(images, sample_ids=('a', 'b'))
        self.assertEqual(len(output.predictions), 2)
        self.assertTrue(output.predictions[0].boxes.requires_grad)
        self.assertEqual(
            sum(output.context.num_anchors_per_level),
            output.predictions[0].boxes.shape[0],
        )
        expected = adapter.detector.box_coder.decode(
            output.context.raw_head_outputs['bbox_regression'][0],
            output.context.anchors_per_image[0],
        )
        self.assertTrue(torch.equal(output.predictions[0].boxes, expected))
        self.assertEqual(output.context.original_image_sizes, ((96, 128), (80, 112)))
        self.assertEqual(len(output.context.num_anchors_per_level), 5)

    def test_raw_forward_accepts_batched_tensor(self):
        adapter = build_adapter()
        output = adapter(torch.rand(2, 3, 64, 64), sample_ids=('a', 'b'))
        self.assertEqual(len(output.predictions), 2)

    def test_inference_uses_iou_quality_in_score_and_zero_based_labels(self):
        adapter = build_adapter()
        detector = adapter.detector
        with torch.no_grad():
            for module in detector.head.classification_head.conv:
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    module.bias.zero_()
            detector.head.classification_head.cls_logits.weight.zero_()
            detector.head.classification_head.cls_logits.bias.copy_(
                torch.tensor([0.0, -10.0, -10.0])
            )
            detector.head.quality_predictor.weight.zero_()
            detector.head.quality_predictor.bias.zero_()
            for module in detector.head.regression_head.conv:
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    module.bias.zero_()
            detector.head.regression_head.bbox_reg.weight.zero_()
            detector.head.regression_head.bbox_reg.bias.zero_()
        adapter.eval()
        predictions = adapter.postprocess(
            adapter([torch.rand(3, 64, 64)], sample_ids=('a',))
        )
        self.assertTrue(len(predictions[0]['scores']) > 0)
        self.assertTrue(torch.allclose(predictions[0]['scores'], torch.full_like(
            predictions[0]['scores'], 0.5), atol=1e-5
        ))
        self.assertTrue(torch.all(predictions[0]['labels'] == 0))

    def test_assignment_empty_targets_is_all_background(self):
        adapter = build_adapter()
        anchors = torch.tensor([
            [0.0, 0.0, 8.0, 8.0],
            [8.0, 0.0, 16.0, 8.0],
        ])
        matched = adapter._match_anchors_to_targets(
            anchors,
            {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)},
            (2,),
        )
        self.assertTrue(torch.equal(matched, torch.tensor([-1, -1])))

    def test_prepare_supervised_uses_transformed_targets_and_detaches_iou(self):
        adapter = build_adapter()
        adapter.eval()
        target = {
            'boxes': torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
            'labels': torch.tensor([2]),
        }
        output = adapter([torch.rand(3, 64, 64)], [target], sample_ids=('a',))
        batch = adapter.prepare_supervised(output, [target])
        prepared = batch.targets[0]
        self.assertFalse(prepared['class_targets'].requires_grad)
        self.assertFalse(prepared['quality_targets'].requires_grad)
        self.assertEqual(prepared['class_targets'].shape[1], 3)
        self.assertTrue(torch.all(prepared['quality_targets'] >= 0))
        self.assertIn('matched_idxs', prepared)

    def test_distillation_rejects_different_ids(self):
        adapter = build_adapter()
        adapter.eval()
        student = adapter([torch.rand(3, 64, 64)], sample_ids=('a',))
        teacher = adapter([torch.rand(3, 64, 64)], sample_ids=('b',))
        with self.assertRaisesRegex(ValueError, 'sample'):
            adapter.prepare_distillation(student, teacher)

    def test_distillation_accepts_photometric_difference_with_same_geometry(self):
        adapter = build_adapter()
        adapter.eval()
        student = adapter([torch.rand(3, 64, 64)], sample_ids=('a',))
        teacher = adapter([torch.rand(3, 64, 64)], sample_ids=('a',))
        pair = adapter.prepare_distillation(student, teacher)
        self.assertEqual(len(pair.student), 1)
        self.assertEqual(pair.student[0].boxes.shape, pair.teacher[0].boxes.shape)


if __name__ == '__main__':
    unittest.main()

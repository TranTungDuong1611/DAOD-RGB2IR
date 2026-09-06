import copy
import unittest

import torch

from models.torchvision_fcos_adapter import ClassificationInitMode, FCOSIoUHead
from torchvision.models.detection import fcos_resnet50_fpn


def build_test_head(mode):
    detector = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=91,
    )
    return FCOSIoUHead(detector.head, 3, mode), detector


class FCOSIoUHeadTests(unittest.TestCase):
    def test_iou_head_outputs_three_classes_and_one_quality_value(self):
        head, _ = build_test_head(ClassificationInitMode.COCO_TOWER)
        output = head([torch.randn(2, 256, 8, 8) for _ in range(5)])
        self.assertEqual(output['cls_logits'].shape, (2, 5 * 64, 3))
        self.assertEqual(output['bbox_regression'].shape, (2, 5 * 64, 4))
        self.assertEqual(output['quality_logits'].shape, (2, 5 * 64, 1))
        self.assertGreaterEqual(output['bbox_regression'].min().item(), 0.0)

    def test_coco_mode_copies_classification_tower(self):
        head, detector = build_test_head(ClassificationInitMode.COCO_TOWER)
        self.assertTrue(torch.equal(
            head.classification_head.conv[0].weight,
            detector.head.classification_head.conv[0].weight,
        ))

    def test_random_mode_replaces_only_classification_tower(self):
        head, detector = build_test_head(ClassificationInitMode.RANDOM_HEAD)
        self.assertFalse(torch.equal(
            head.classification_head.conv[0].weight,
            detector.head.classification_head.conv[0].weight,
        ))
        self.assertTrue(torch.equal(
            head.regression_head.conv[0].weight,
            detector.head.regression_head.conv[0].weight,
        ))
        self.assertTrue(torch.equal(
            head.regression_head.bbox_reg.weight,
            detector.head.regression_head.bbox_reg.weight,
        ))

    def test_predictors_are_new_and_centerness_is_not_registered(self):
        head, detector = build_test_head(ClassificationInitMode.COCO_TOWER)
        self.assertEqual(head.classification_head.cls_logits.out_channels, 3)
        self.assertEqual(head.quality_logits.out_channels, 1)
        self.assertFalse(any('bbox_ctrness' in name for name, _ in head.named_parameters()))
        self.assertFalse(torch.equal(
            head.classification_head.cls_logits.weight,
            detector.head.classification_head.cls_logits.weight[:3],
        ))

    def test_quality_branch_receives_gradient_from_classification_features(self):
        head, _ = build_test_head(ClassificationInitMode.COCO_TOWER)
        features = [torch.randn(2, 256, 4, 4, requires_grad=True) for _ in range(5)]
        output = head(features)
        output['quality_logits'].sum().backward()
        self.assertTrue(any(feature.grad is not None for feature in features))


if __name__ == '__main__':
    unittest.main()

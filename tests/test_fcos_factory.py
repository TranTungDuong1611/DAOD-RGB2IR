import copy
import unittest
from unittest import mock

import torch

from config import FCOSModelConfig, TrainingConfig
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_wrapper import D3TWrapper
from models.fcos_factory import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_metadata,
    build_fcos_d3t_model,
    build_fcos_d3t_trio,
    validate_checkpoint_metadata,
)
from models.torchvision_fcos_adapter import ClassificationInitMode
from torchvision.models.detection import fcos_resnet50_fpn


def build_config(
    mode=ClassificationInitMode.COCO_TOWER, teacher_mode="two_teacher"
):
    return TrainingConfig(model=FCOSModelConfig(
        weights=None,
        pretrained_backbone=False,
        classification_init_mode=mode,
        min_sizes=(64, 96),
        max_size=128,
    ), device='cpu', teacher_mode=teacher_mode)


class FactoryTests(unittest.TestCase):
    def test_factory_installs_adapter_and_criterion_without_downloads(self):
        config = build_config()
        sentinel = fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=91,
            min_size=64, max_size=128,
        )
        wrapper = build_fcos_d3t_model(config, base_detector=sentinel)
        self.assertIsInstance(wrapper, D3TWrapper)
        self.assertIs(wrapper.adapter.detector, sentinel)
        self.assertIsInstance(wrapper.criterion, D3TLossCriterion)
        self.assertEqual(wrapper.adapter.class_names, ('person', 'car', 'bicycle'))

    def test_factory_honors_both_classification_modes(self):
        coco = build_fcos_d3t_model(build_config(ClassificationInitMode.COCO_TOWER),
                                     base_detector=fcos_resnet50_fpn(
                                         weights=None, weights_backbone=None, num_classes=91,
                                     ))
        random_head = build_fcos_d3t_model(build_config(ClassificationInitMode.RANDOM_HEAD),
                                            base_detector=fcos_resnet50_fpn(
                                                weights=None, weights_backbone=None, num_classes=91,
                                            ))
        self.assertEqual(coco.adapter.detector.head.classification_init_mode,
                         ClassificationInitMode.COCO_TOWER)
        self.assertEqual(random_head.adapter.detector.head.classification_init_mode,
                         ClassificationInitMode.RANDOM_HEAD)
        self.assertEqual(coco.adapter.detector.head.num_classes, 3)
        self.assertEqual(random_head.adapter.detector.head.num_classes, 3)

    def test_factory_forwards_trainable_backbone_layers(self):
        calls = {}

        def builder(**kwargs):
            calls.update(kwargs)
            return fcos_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=91,
                min_size=64, max_size=128,
            )

        config = build_config()
        config.model.trainable_backbone_layers = 2
        build_fcos_d3t_model(config, detector_builder=builder)
        self.assertEqual(calls['trainable_backbone_layers'], 2)
        self.assertIsNone(calls['weights'])
        self.assertEqual(calls['min_size'], (64, 96))
        self.assertEqual(calls['center_sampling_radius'], 0.0)

    def test_factory_configures_train_choice_and_eval_last_resize(self):
        wrapper = build_fcos_d3t_model(
            build_config(),
            base_detector=fcos_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=91,
                min_size=64,
                max_size=128,
            ),
        )
        transform = wrapper.adapter.detector.transform
        image = torch.rand(3, 32, 32)

        transform.train()
        with mock.patch.object(transform, "torch_choice", return_value=64) as choice:
            train_image, _ = transform.resize(image)
        choice.assert_called_once_with((64, 96))
        self.assertEqual(train_image.shape[-2:], (64, 64))

        transform.eval()
        with mock.patch.object(
            transform,
            "torch_choice",
            side_effect=AssertionError("eval must use the final configured size"),
        ):
            eval_image, _ = transform.resize(image)
        self.assertEqual(eval_image.shape[-2:], (96, 96))

    def test_trio_deepcopies_complete_student_and_freezes_teachers(self):
        config = build_config()
        student, rgb_teacher, ir_teacher = build_fcos_d3t_trio(
            config,
            base_detector=fcos_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=91,
            ),
        )
        self.assertIsNot(student, rgb_teacher)
        self.assertIsNot(student, ir_teacher)
        self.assertEqual(student.state_dict().keys(), rgb_teacher.state_dict().keys())
        self.assertFalse(rgb_teacher.training)
        self.assertFalse(ir_teacher.training)
        self.assertTrue(all(not p.requires_grad for p in rgb_teacher.parameters()))
        self.assertTrue(all(not p.requires_grad for p in ir_teacher.parameters()))

    def test_factory_can_build_a_single_enabled_teacher(self):
        config = build_config(teacher_mode="rgb")
        student, rgb_teacher, ir_teacher = build_fcos_d3t_trio(
            config,
            base_detector=fcos_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=91,
            ),
        )
        self.assertIsNotNone(student)
        self.assertIsNotNone(rgb_teacher)
        self.assertIsNone(ir_teacher)
        self.assertFalse(rgb_teacher.training)

    def test_checkpoint_metadata_rejects_incompatible_classification_mode(self):
        config = build_config()
        metadata = build_checkpoint_metadata(config)
        self.assertEqual(metadata['schema_version'], CHECKPOINT_SCHEMA_VERSION)
        validate_checkpoint_metadata(metadata, config)
        bad = dict(metadata)
        bad['classification_init_mode'] = ClassificationInitMode.RANDOM_HEAD.value
        with self.assertRaisesRegex(ValueError, 'classification'):
            validate_checkpoint_metadata(bad, config)


if __name__ == '__main__':
    unittest.main()

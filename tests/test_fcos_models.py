import copy
import unittest
from unittest import mock

import torch

from config import FCOSModelConfig, TrainingConfig
from loss.detection_criterion import DetectionLossCriterion
from models.detector_wrapper import DetectorWrapper
from models import fcos_models
from models.fcos_models import (
    build_fcos_model,
    build_fcos_trio,
)
from models.fcos_adapter import ClassificationInitMode
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


class TeacherLifecycleTests(unittest.TestCase):
    def test_initialize_selected_teacher_copies_parameters_and_buffers(self):
        student = torch.nn.BatchNorm1d(2)
        rgb, ir = copy.deepcopy(student), copy.deepcopy(student)
        with torch.no_grad():
            student.weight.fill_(3)
            student.running_mean.fill_(4)
            student.num_batches_tracked.fill_(7)
        fcos_models.initialize_teachers(student, {"rgb": rgb, "ir": ir}, names=("ir",))
        for name, value in student.state_dict().items():
            self.assertTrue(torch.equal(ir.state_dict()[name], value))
        self.assertTrue(torch.equal(rgb.weight, torch.ones(2)))
        self.assertEqual(rgb.num_batches_tracked.item(), 0)

    def test_ema_preserves_warmup_and_integer_buffer_behavior(self):
        for step, expected in ((0, 9.), (10, 4.5), (100, 1.)):
            with self.subTest(step=step):
                student = torch.nn.BatchNorm1d(2)
                teacher = copy.deepcopy(student).requires_grad_(False)
                with torch.no_grad():
                    student.weight.fill_(10)
                    student.running_mean.fill_(10)
                    student.num_batches_tracked.fill_(7)
                    teacher.weight.zero_()
                fcos_models.ema_update(teacher, student, alpha=.9, global_step=step)
                torch.testing.assert_close(teacher.weight, torch.full((2,), expected))
                torch.testing.assert_close(teacher.running_mean, torch.full((2,), expected))
                self.assertEqual(teacher.num_batches_tracked.item(), 7)
                self.assertIsNone(student.weight.grad)

    def test_teacher_forward_uses_eval_without_autograd_and_reuses_geometry(self):
        class Teacher(torch.nn.Module):
            def raw(self, images, **kwargs):
                self.seen = (self.training, kwargs)
                return images.square()

        teacher = fcos_models.get_teacher({"ir": Teacher()}, "ir")
        output = fcos_models.forward_teacher(
            teacher, torch.ones(1, requires_grad=True), sample_ids=("ir-1",), resize_short_edge=96,
        )
        self.assertFalse(output.requires_grad)
        self.assertEqual(teacher.seen, (False, {"sample_ids": ("ir-1",), "resize_short_edge": 96}))


class FactoryTests(unittest.TestCase):
    def test_factory_installs_adapter_and_criterion_without_downloads(self):
        config = build_config()
        sentinel = fcos_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=91,
            min_size=64, max_size=128,
        )
        wrapper = build_fcos_model(config, base_detector=sentinel)
        self.assertIsInstance(wrapper, DetectorWrapper)
        self.assertIs(wrapper.adapter.detector, sentinel)
        self.assertIsInstance(wrapper.criterion, DetectionLossCriterion)
        self.assertEqual(wrapper.adapter.class_names, ('person', 'car', 'bicycle'))

    def test_factory_honors_both_classification_modes(self):
        coco = build_fcos_model(build_config(ClassificationInitMode.COCO_TOWER),
                                     base_detector=fcos_resnet50_fpn(
                                         weights=None, weights_backbone=None, num_classes=91,
                                     ))
        random_head = build_fcos_model(build_config(ClassificationInitMode.RANDOM_HEAD),
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
        build_fcos_model(config, detector_builder=builder)
        self.assertEqual(calls['trainable_backbone_layers'], 2)
        self.assertIsNone(calls['weights'])
        self.assertEqual(calls['min_size'], (64, 96))
        self.assertEqual(calls['center_sampling_radius'], 0.0)

    def test_factory_configures_train_choice_and_eval_last_resize(self):
        wrapper = build_fcos_model(
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
        student, rgb_teacher, ir_teacher = build_fcos_trio(
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
        student, rgb_teacher, ir_teacher = build_fcos_trio(
            config,
            base_detector=fcos_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=91,
            ),
        )
        self.assertIsNotNone(student)
        self.assertIsNotNone(rgb_teacher)
        self.assertIsNone(ir_teacher)
        self.assertFalse(rgb_teacher.training)

if __name__ == '__main__':
    unittest.main()

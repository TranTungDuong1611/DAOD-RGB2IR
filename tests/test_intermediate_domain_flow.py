from dataclasses import asdict
import sys
import unittest
from unittest import mock

import torch

from config import IntermediateDomainConfig
from randommono_btf import RandomMonoBTF
from saga import SoftSAGA
from training.data_flow import RouteDataFlow


class IdentityAugmentor:
    def __init__(self, *, mode="randommono_btf", randommono_btf=None, saga=None):
        self.intermediate_mode = mode
        self.saga = saga or SoftSAGA()
        self.randommono_btf = randommono_btf or RandomMonoBTF(IntermediateDomainConfig(texture_strength=0))

    def apply_weak_aug(self, images, targets=None):
        return images, targets, False

    def apply_photometric_aug(self, images):
        return images


class BridgeFlowTests(unittest.TestCase):
    def test_same_batch_endpoint_for_teacher_student_without_box_mutation(self):
        images = torch.rand(2, 3, 10, 12)
        boxes = torch.tensor([[1., 2., 7., 8.]])
        targets = [dict(boxes=boxes.clone(), labels=torch.tensor([0]), stem=f"s{i}") for i in range(2)]
        augmentor = IdentityAugmentor(
            mode="randommono_btf",
            randommono_btf=RandomMonoBTF(IntermediateDomainConfig(texture_strength=0)),
        )
        flow = RouteDataFlow([(images, targets, ("s0", "s1"))], None, device="cpu", augmentor=augmentor)
        result = flow.prepare(source_domain="rgb", use_gt=True, student_alpha=.25, teacher_alpha=.5)
        endpoint_s = (result["student_images"] - .25 * images) / .75
        endpoint_t = (result["teacher_images"] - .5 * images) / .5
        torch.testing.assert_close(endpoint_s, endpoint_t)
        for target in result["targets"]:
            torch.testing.assert_close(target["boxes"], boxes)
        self.assertEqual(result["sample_ids"], ("s0", "s1"))

    def test_saga_mode_dispatches_to_saga_not_randommono_btf(self):
        images = torch.rand(1, 3, 10, 12)
        boxes = torch.tensor([[1., 2., 7., 8.]])
        targets = [dict(boxes=boxes.clone(), labels=torch.tensor([0]), stem="s0")]
        randommono = mock.Mock()
        augmentor = IdentityAugmentor(mode="saga", randommono_btf=randommono)
        flow = RouteDataFlow([(images, targets, ("s0",))], None, device="cpu", augmentor=augmentor)
        result = flow.prepare(source_domain="rgb", use_gt=True, student_alpha=.25, teacher_alpha=.5)
        randommono.make_views.assert_not_called()
        torch.testing.assert_close(result["student_images"], SoftSAGA().apply_to_batch(images, [boxes], .25))
        torch.testing.assert_close(result["teacher_images"], SoftSAGA().apply_to_batch(images, [boxes], .5))

    def test_ir_route_never_invokes_bridge(self):
        images = torch.rand(1, 3, 10, 12)
        randommono = mock.Mock()
        saga = mock.Mock()
        augmentor = IdentityAugmentor(mode="randommono_btf", randommono_btf=randommono, saga=saga)
        flow = RouteDataFlow([], [(images, ("ir0",))], device="cpu", augmentor=augmentor)
        result = flow.prepare(source_domain="ir", use_gt=False, student_alpha=None, teacher_alpha=None)
        randommono.make_views.assert_not_called()
        saga.apply_to_batch.assert_not_called()
        self.assertIsNone(result["targets"])
        torch.testing.assert_close(result["student_images"], images)

    def test_cli_defaults_and_ablation_overrides(self):
        from example_flir import parse_args, build_training_config
        with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "unused"]):
            cfg = build_training_config(parse_args())
        self.assertEqual(cfg.intermediate_domain.mode, "randommono_btf")
        self.assertEqual(cfg.intermediate_domain.texture_strength, .25)
        with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "unused", "--intermediate-domain", "saga", "--texture-strength", "0", "--btf-radius", "3", "--btf-iterations", "2"]):
            cfg = build_training_config(parse_args())
        self.assertEqual(cfg.intermediate_domain.mode, "saga")
        self.assertEqual(cfg.intermediate_domain.btf_radius, 3)
        self.assertEqual(cfg.intermediate_domain.btf_iterations, 2)
        self.assertEqual(asdict(cfg)["intermediate_domain"]["mode"], "saga")

    def test_gan_bridge_saga_domain_uses_randommono_bridge(self):
        images = torch.rand(2, 3, 10, 12)
        boxes = torch.tensor([[1., 2., 7., 8.]])
        targets = [dict(boxes=boxes.clone(), labels=torch.tensor([0]), stem=f"s{i}") for i in range(2)]
        augmentor = IdentityAugmentor(
            mode="randommono_btf",
            randommono_btf=RandomMonoBTF(IntermediateDomainConfig(mode="randommono_btf", texture_strength=0)),
        )
        flow = RouteDataFlow([(images, targets, ("s0", "s1"))], None, device="cpu", augmentor=augmentor)
        result = flow.prepare(
            source_domain="rgb",
            appearance_domain="saga",
            use_gt=True,
            student_alpha=0.25,
            teacher_alpha=0.5,
        )
        endpoint_s = (result["student_images"] - 0.25 * images) / 0.75
        endpoint_t = (result["teacher_images"] - 0.5 * images) / 0.5
        torch.testing.assert_close(endpoint_s, endpoint_t)

import sys
import unittest
from unittest import mock

import torch

from config import IntermediateDomainConfig
from intermediate_domain import IntermediateDomainAugmentation
from training.data_flow import RouteDataFlow


class IdentityAugmentor:
    def apply_weak_aug(self, images, targets=None):
        return images, targets, False
    def apply_photometric_aug(self, images):
        return images


class BridgeFlowTests(unittest.TestCase):
    def test_same_batch_endpoint_for_teacher_student_without_box_mutation(self):
        images = torch.rand(2, 3, 10, 12)
        boxes = torch.tensor([[1., 2., 7., 8.]])
        targets = [dict(boxes=boxes.clone(), labels=torch.tensor([0]), stem=f"s{i}") for i in range(2)]
        bridge = IntermediateDomainAugmentation(IntermediateDomainConfig(texture_strength=0))
        flow = RouteDataFlow([(images, targets, ("s0", "s1"))], None, device="cpu", augmentor=IdentityAugmentor(), bridge=bridge)
        result = flow.prepare(source_domain="rgb", use_gt=True, student_alpha=.25, teacher_alpha=.5)
        endpoint_s = (result["student_images"] - .25 * images) / .75
        endpoint_t = (result["teacher_images"] - .5 * images) / .5
        torch.testing.assert_close(endpoint_s, endpoint_t)
        for target in result["targets"]:
            torch.testing.assert_close(target["boxes"], boxes)
        self.assertEqual(result["sample_ids"], ("s0", "s1"))

    def test_ir_route_never_invokes_bridge(self):
        images = torch.rand(1, 3, 10, 12)
        bridge = mock.Mock()
        flow = RouteDataFlow([], [(images, ("ir0",))], device="cpu", augmentor=IdentityAugmentor(), bridge=bridge)
        result = flow.prepare(source_domain="ir", use_gt=False, student_alpha=None, teacher_alpha=None)
        bridge.make_views.assert_not_called()
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
        from models.fcos_factory import build_checkpoint_metadata
        self.assertEqual(build_checkpoint_metadata(cfg)["config"]["intermediate_domain"]["mode"], "saga")

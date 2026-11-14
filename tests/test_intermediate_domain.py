import sys
import unittest
from unittest import mock

import numpy as np
import torch

from config import IntermediateDomainConfig
from intermediate_domain import IntermediateDomainAugmentation


class IntermediateDomainTests(unittest.TestCase):
    def test_identity_bypasses_rng_and_optional_backend(self):
        images = torch.rand(2, 3, 9, 11)
        rng = torch.random.get_rng_state().clone()
        with mock.patch.dict(sys.modules, {"cv2": None}):
            views = IntermediateDomainAugmentation().make_views(images, (1.0, 1.0))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(all(torch.equal(v, images) for v in views))

    def test_random_projection_is_shared_across_levels_and_whole_image(self):
        images = torch.rand(2, 3, 9, 11)
        original = images.clone()
        torch.manual_seed(19)
        bridge = IntermediateDomainAugmentation(IntermediateDomainConfig(texture_strength=0))
        zero, quarter, half = bridge.make_views(images, (0.0, 0.25, 0.5), boxes=[torch.empty(0, 4)] * 2)
        torch.manual_seed(19)
        weights = torch.rand(2, 3, 1, 1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        expected = (images * weights).sum(dim=1, keepdim=True).expand_as(images)
        torch.testing.assert_close(zero, expected)
        torch.testing.assert_close(quarter, .25 * images + .75 * zero)
        torch.testing.assert_close(half, .5 * images + .5 * zero)
        self.assertTrue(torch.equal(images, original))

    def test_filter_runs_once_per_image_and_restores_dtype(self):
        calls = []
        def filter_image(image):
            calls.append(image.copy())
            self.assertEqual(image.dtype, np.float32)
            self.assertEqual(image.ndim, 2)
            return image * .5
        images = torch.rand(2, 3, 8, 10).half()
        cfg = IntermediateDomainConfig(texture_strength=.25)
        endpoint, middle, rgb = IntermediateDomainAugmentation(cfg, texture_filter=filter_image).make_views(images, (0, .5, 1))
        self.assertEqual(len(calls), 2)
        expected = np.stack(calls) * .875
        torch.testing.assert_close(endpoint[:, 0].float(), torch.from_numpy(expected), atol=.001, rtol=.001)
        torch.testing.assert_close(middle, .5 * images + .5 * endpoint, atol=.001, rtol=.001)
        self.assertEqual(endpoint.dtype, images.dtype)
        self.assertEqual(endpoint.device, images.device)
        self.assertTrue(torch.equal(rgb, images))

    def test_ragged_image_batches_keep_shapes_and_share_each_endpoint(self):
        images = (torch.rand(3, 5, 7), torch.rand(3, 8, 6))
        bridge = IntermediateDomainAugmentation(IntermediateDomainConfig(texture_strength=0))
        endpoints, halves = bridge.make_views(images, (0, .5))
        for image, endpoint, half in zip(images, endpoints, halves):
            self.assertEqual(endpoint.shape, image.shape)
            torch.testing.assert_close(half, .5 * image + .5 * endpoint)

    def test_legacy_mode_matches_saga(self):
        from saga import SoftSAGA
        image = torch.rand(1, 3, 8, 10)
        boxes = [torch.tensor([[1., 2., 5., 6.]])]
        bridge = IntermediateDomainAugmentation(IntermediateDomainConfig(mode="saga"))
        (view,) = bridge.make_views(image, (.3,), boxes=boxes)
        torch.testing.assert_close(view, SoftSAGA().apply_to_batch(image, boxes, .3))

    def test_missing_btf_is_an_explicit_error(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "opencv-contrib"):
                IntermediateDomainAugmentation().make_views(torch.rand(1, 3, 8, 8), (0.,))

    def test_invalid_config_and_inputs_fail(self):
        for kwargs in ({"texture_strength": float("nan")}, {"texture_strength": 1.1}, {"btf_radius": 0}, {"btf_iterations": 0}, {"mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                IntermediateDomainConfig(**kwargs)
        bridge = IntermediateDomainAugmentation(IntermediateDomainConfig(texture_strength=0))
        for images, alpha in ((torch.ones(1, 3, 2, 2), -1), (torch.ones(1, 3, 2, 2), float("nan")), (torch.ones(1, 2, 2, 2), .5), (torch.full((1, 3, 2, 2), 2.), .5)):
            with self.assertRaises(ValueError):
                bridge.make_views(images, (alpha,))

    def test_real_btf_reduces_synthetic_texture(self):
        import cv2
        if not hasattr(getattr(cv2, "ximgproc", None), "bilateralTextureFilter"):
            self.skipTest("requires actual OpenCV contrib BTF")
        yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
        mono = .3 + .4 * ((xx + yy) % 2).float()
        images = mono[None, None].repeat(1, 3, 1, 1)
        (view,) = IntermediateDomainAugmentation(IntermediateDomainConfig(texture_strength=1)).make_views(images, (0.,))
        self.assertLess(view[0, 0, 10:-10, 10:-10].std().item(), mono[10:-10, 10:-10].std().item())
        self.assertTrue(torch.isfinite(view).all())
        self.assertGreaterEqual(view.min().item(), 0.)
        self.assertLessEqual(view.max().item(), 1.)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp16_roundtrip_with_real_btf(self):
        images = torch.rand(1, 3, 12, 14, device="cuda", dtype=torch.float16)
        (view,) = IntermediateDomainAugmentation().make_views(images, (.5,))
        self.assertEqual(view.device, images.device)
        self.assertEqual(view.dtype, images.dtype)

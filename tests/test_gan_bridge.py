"""Image contracts for the frozen RGB-to-IR bridge."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from config import AugConfig, TrainingConfig
from data.augmentations import StudentAugmentor
from training.data_flow import RouteDataFlow


class FirstChannelGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.detach().clone())
        return images[:, :1] * self.scale


class GANTranslationTests(unittest.TestCase):
    def test_translation_normalizes_pads_crops_and_freezes_without_input_gradient(self):
        from training.gan_bridge import FrozenGANTranslator

        generator = FirstChannelGenerator()
        translator = FrozenGANTranslator(generator)
        translator.train()
        images = torch.linspace(0, 1, 2 * 3 * 9 * 11).reshape(2, 3, 9, 11).requires_grad_()
        output = translator(images)
        self.assertEqual(output.shape, images.shape)
        self.assertTrue(torch.allclose(output[:, 0], images[:, 0]))
        self.assertTrue(torch.equal(output[:, 0], output[:, 2]))
        self.assertEqual(generator.inputs[0].shape, (1, 3, 12, 12))
        self.assertAlmostEqual(generator.inputs[0][0, 0, 0, 0].item(), -1)
        self.assertFalse(output.requires_grad)
        self.assertFalse(generator.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in generator.parameters()))
        self.assertEqual(output.device, images.device)

    def test_bad_input_and_output_contracts_are_rejected(self):
        from training.gan_bridge import FrozenGANTranslator

        translator = FrozenGANTranslator(FirstChannelGenerator())
        for images in (torch.zeros(3, 8, 8), torch.zeros(1, 1, 8, 8),
                       torch.zeros(0, 3, 8, 8), torch.zeros(1, 3, 8, 8, dtype=torch.uint8),
                       torch.full((1, 3, 8, 8), float("nan")), torch.full((1, 3, 8, 8), 2.0)):
            with self.subTest(shape=images.shape), self.assertRaises(ValueError):
                translator(images)
        with self.assertRaisesRegex(ValueError, "output"):
            FrozenGANTranslator(nn.Identity())(torch.zeros(1, 3, 8, 8))

    def test_generator_has_reference_state_keys_and_odd_shape_translation(self):
        from models.cyclegan_generator import CycleGANGenerator
        from training.gan_bridge import FrozenGANTranslator

        generator = CycleGANGenerator(ngf=4)
        state = generator.state_dict()
        self.assertEqual(state["model.1.weight"].shape, (4, 3, 7, 7))
        self.assertEqual(state["model.10.conv_block.1.weight"].shape, (16, 16, 3, 3))
        self.assertEqual(state["model.18.conv_block.5.weight"].shape, (16, 16, 3, 3))
        self.assertEqual(state["model.26.weight"].shape, (1, 4, 7, 7))
        self.assertNotIn("model.2.running_mean", state)
        output = FrozenGANTranslator(generator)(torch.rand(1, 3, 17, 19))
        self.assertEqual(output.shape, (1, 3, 17, 19))
        self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))

    def test_checkpoint_load_preserves_weights_rng_and_translation_at_any_path(self):
        from models.cyclegan_generator import CycleGANGenerator
        from training.gan_bridge import load_gan_translator

        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "G_A.pth", Path(directory) / "copy.pth"
            source = CycleGANGenerator(ngf=4)
            torch.save(source.state_dict(), first)
            second.write_bytes(first.read_bytes())
            rng = torch.random.get_rng_state().clone()
            def factory():
                return CycleGANGenerator(ngf=4)

            loaded = load_gan_translator(str(first), "cpu", generator_factory=factory)
            moved = load_gan_translator(str(second), "cpu", generator_factory=factory)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.generator.state_dict()[name]))
            images = torch.rand(1, 3, 17, 19)
            self.assertTrue(torch.equal(loaded(images), moved(images)))
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            broken = copy.deepcopy(source.state_dict())
            broken.pop("model.1.weight")
            torch.save(broken, first)
            with self.assertRaisesRegex(ValueError, "checkpoint|architecture"):
                load_gan_translator(str(first), "cpu", generator_factory=factory)
            with self.assertRaises(FileNotFoundError):
                load_gan_translator(str(Path(directory) / "missing.pth"), "cpu")


class BridgeDataFlowTests(unittest.TestCase):
    def build_flow(self, flip=0.0):
        self.images = torch.zeros(1, 3, 8, 8)
        self.images[:, 0] = 1.0
        self.target = {"boxes": torch.tensor([[1., 2., 4., 6.]]), "labels": torch.tensor([0]), "stem": "source"}
        config = TrainingConfig(aug=AugConfig(hflip_prob=flip, blur_prob=0, brightness_prob=0, contrast_prob=0))
        self.generator = FirstChannelGenerator()
        from training.gan_bridge import FrozenGANTranslator
        return RouteDataFlow(
            [(self.images, [self.target], ("source",))],
            [(torch.full_like(self.images, 0.2), ("target",))],
            device="cpu", augmentor=StudentAugmentor(config),
            gan_translator=FrozenGANTranslator(self.generator),
        )

    def prepare(self, flow, domain, alpha=0.5):
        return flow.prepare(
            source_domain="ir" if domain == "ir_real" else "rgb", use_gt=domain != "ir_real",
            student_alpha=alpha, teacher_alpha=alpha, appearance_domain=domain,
        )

    def test_rgb_batch_requires_explicit_sample_ids(self):
        prepared = self.build_flow()
        flow = RouteDataFlow(
            [(self.images, [self.target])], None,
            device="cpu", augmentor=prepared.augmentor,
        )
        with self.assertRaises(ValueError):
            self.prepare(flow, "rgb")

    def test_ir_batch_rejects_extra_fields(self):
        prepared = self.build_flow()
        flow = RouteDataFlow(
            [], [(self.images, ("target",), [self.target])],
            device="cpu", augmentor=prepared.augmentor,
        )
        with self.assertRaises(ValueError):
            self.prepare(flow, "ir_real")

    def test_each_appearance_selects_the_correct_source_and_gt(self):
        flow = self.build_flow()
        rgb = self.prepare(flow, "rgb")
        saga = self.prepare(flow, "saga")
        gan = self.prepare(flow, "ir_gan")
        ir = self.prepare(flow, "ir_real")
        self.assertTrue(torch.equal(rgb["teacher_images"], self.images))
        self.assertTrue(torch.equal(saga["teacher_images"][0, :, 0, 0], self.images[0, :, 0, 0]))
        self.assertTrue(torch.allclose(saga["teacher_images"][0, :, 3, 2], torch.tensor([0.6495, 0.1495, 0.1495])))
        self.assertTrue(torch.equal(gan["teacher_images"], torch.ones_like(self.images)))
        self.assertEqual(len(self.generator.inputs), 1)
        self.assertTrue(torch.equal(self.generator.inputs[0], self.images * 2 - 1))
        for batch in (rgb, saga, gan):
            self.assertEqual(batch["sample_ids"], ("source",))
            self.assertTrue(torch.equal(batch["targets"][0]["boxes"], self.target["boxes"]))
        self.assertIsNone(ir["targets"])
        self.assertEqual(ir["sample_ids"], ("target",))

    def test_shared_flip_keeps_gan_views_boxes_and_identity_aligned(self):
        flow = self.build_flow(flip=1.0)
        batch = self.prepare(flow, "ir_gan")
        self.assertTrue(torch.equal(batch["targets"][0]["boxes"], torch.tensor([[4., 2., 7., 6.]])))
        self.assertTrue(torch.equal(batch["student_images"], batch["teacher_images"]))
        self.assertEqual(batch["sample_ids"], ("source",))

    def test_empty_gt_keeps_saga_images_unchanged(self):
        flow = self.build_flow()
        self.target["boxes"] = torch.empty(0, 4)
        self.target["labels"] = torch.empty(0, dtype=torch.long)
        self.assertTrue(torch.equal(self.prepare(flow, "saga")["teacher_images"], self.images))


if __name__ == "__main__":
    unittest.main()

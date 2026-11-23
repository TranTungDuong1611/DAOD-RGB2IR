"""Load and execute frozen RGB-to-IR GAN translation."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.cyclegan_generator import CycleGANGenerator


class FrozenGANTranslator(nn.Module):
    """Translate [B,3,H,W] RGB in [0,1] without changing detector coordinates."""

    def __init__(self, generator: nn.Module, *, identity=None):
        super().__init__()
        self.generator = generator
        self.identity = dict(identity or {})
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # The translator remains inference-only even if an owning module trains.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[1] != 3 or any(size == 0 for size in images.shape)
                or not images.is_floating_point() or not bool(torch.isfinite(images).all())
                or bool((images < 0).any()) or bool((images > 1).any())):
            raise ValueError("GAN input must be a finite nonempty float [B,3,H,W] tensor in [0,1]")
        height, width = images.shape[-2:]
        pad_h = max(8, (height + 3) // 4 * 4) - height
        pad_w = max(8, (width + 3) // 4 * 4) - width
        outputs = []
        # One image at a time bounds generator activations independently of batch size.
        for image in images.split(1):
            gan_input = F.pad(image * 2 - 1, (0, pad_w, 0, pad_h), mode="replicate")
            output = self.generator(gan_input)
            if (not isinstance(output, torch.Tensor)
                    or output.shape != (1, 1, height + pad_h, width + pad_w)
                    or output.device != images.device or not bool(torch.isfinite(output).all())
                    or bool((output.abs() > 1.00001).any())):
                raise ValueError("GAN output must be finite [1,1,H,W] in [-1,1] at the padded input size")
            thermal = ((output[:, :, :height, :width] + 1) * 0.5).clamp(0, 1)
            outputs.append(thermal.repeat(1, 3, 1, 1))
        return torch.cat(outputs, dim=0)


def load_gan_translator(checkpoint: str, device, *, generator_factory=CycleGANGenerator):
    """Load a strict G_A state dict; identify its bytes independently of its path."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"GAN checkpoint not found: {path}")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
        raise ValueError("GAN checkpoint must contain a generator state dict")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    # Discard initialization RNG consumption: loading a frozen auxiliary model
    # must not change detector initialization or augmentation randomness.
    with torch.random.fork_rng(devices=[]):
        generator = generator_factory()
    try:
        generator.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("GAN checkpoint is incompatible with the RGB(3)->IR(1) ResNet-9/instance-norm architecture") from error
    identity = {"sha256": digest.hexdigest(), "architecture": "cyclegan_resnet9_rgb3_ir1_instance_ngf64"}
    return FrozenGANTranslator(generator, identity=identity).to(device)

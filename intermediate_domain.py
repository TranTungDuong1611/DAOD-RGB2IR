"""Source-only RandomMono/BTF views, shared across teacher/student levels."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

from config import IntermediateDomainConfig


def require_btf():
    """Resolve the optional real algorithm lazily; never substitute another blur."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("BTF requires opencv-contrib-python-headless; install it in the training environment") from error
    function = getattr(getattr(cv2, "ximgproc", None), "bilateralTextureFilter", None)
    if function is None:
        raise RuntimeError("BTF requires opencv-contrib-python-headless with ximgproc.bilateralTextureFilter; use one OpenCV distribution in the environment")
    return function


class IntermediateDomainAugmentation:
    """Generate one endpoint per image, then render multiple alpha views.

    Alpha=1 is RGB and consumes no bridge random numbers/backend work.
    Inputs may be a BCHW tensor or a sequence of CHW tensors. Outputs preserve
    that representation, geometry, device and dtype. BTF is CPU preprocessing,
    not a trainable generator; its result is detached from input autograd.
    """

    def __init__(self, config=None, *, texture_filter: Callable | None = None):
        self.config = config or IntermediateDomainConfig()
        self._texture_filter = texture_filter

    def _filter(self, mono: np.ndarray) -> np.ndarray:
        if self._texture_filter is not None:
            return self._texture_filter(mono)
        function = require_btf()
        return function(mono, fr=self.config.btf_radius,
                        numIter=self.config.btf_iterations,
                        sigmaAlpha=-1.0, sigmaAvg=-1.0)

    def make_views(self, images, alphas: Sequence[float], *, boxes=None):
        alphas = tuple(float(alpha) for alpha in alphas)
        if any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas):
            raise ValueError("bridge alphas must be finite values in [0, 1]")
        stacked = isinstance(images, torch.Tensor)
        if stacked and images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W]")
        items = tuple(images)
        if not items or any(image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point() for image in items):
            raise ValueError("images must contain non-empty floating-point RGB tensors")
        if any(image.shape[1] == 0 or image.shape[2] == 0 for image in items):
            raise ValueError("image dimensions must be positive")
        if not alphas:
            return ()
        if all(alpha == 1 for alpha in alphas):
            return tuple(images.clone() if stacked else tuple(image.clone() for image in items) for _ in alphas)
        if self.config.mode == "saga":
            from saga import SoftSAGA
            if boxes is None or len(boxes) != len(items):
                raise ValueError("legacy SAGA requires boxes for each image")
            views = [tuple(SoftSAGA().apply(image, box, alpha) for image, box in zip(items, boxes)) for alpha in alphas]
        else:
            if any(not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()) for image in items):
                raise ValueError("bridge inputs must be finite values in [0, 1] before detector normalization")
            weights = torch.rand(len(items), 3, 1, 1)
            sums = weights.sum(dim=1, keepdim=True)
            weights = torch.where(sums > 0, weights / sums.clamp_min(1e-12), torch.full_like(weights, 1 / 3))
            endpoints = []
            for image, weight in zip(items, weights):
                mono = (image.float() * weight.to(image.device)).sum(dim=0)
                if self.config.texture_strength:
                    source = np.ascontiguousarray(mono.detach().cpu().numpy(), dtype=np.float32)
                    filtered = np.asarray(self._filter(source), dtype=np.float32)
                    if filtered.shape != source.shape or not np.isfinite(filtered).all():
                        raise ValueError("BTF must return a finite intensity image of the original size")
                    endpoint = torch.from_numpy(source + self.config.texture_strength * (filtered - source))
                    endpoint = endpoint.to(device=image.device, dtype=image.dtype).clamp(0, 1)
                else:
                    endpoint = mono.to(image.dtype)
                endpoints.append(endpoint.unsqueeze(0).expand_as(image))
            views = [tuple(image.clone() if alpha == 1 else alpha * image + (1 - alpha) * endpoint
                           for image, endpoint in zip(items, endpoints)) for alpha in alphas]
        return tuple(torch.stack(view) if stacked else view for view in views)

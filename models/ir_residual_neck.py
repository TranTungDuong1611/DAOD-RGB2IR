"""Residual feature adaptation applied only to infrared FPN features."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class ResidualAdapterBlock(nn.Module):
    """A bottleneck residual adapter that starts as an exact identity."""

    def __init__(
        self,
        channels: int = 256,
        bottleneck_channels: int = 64,
        norm_groups: int = 16,
    ) -> None:
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channels and bottleneck_channels must be positive")
        if norm_groups <= 0 or bottleneck_channels % norm_groups != 0:
            raise ValueError("norm_groups must divide bottleneck_channels")

        self.adapter = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.GroupNorm(norm_groups, bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )
        # A zero gate preserves the pretrained FPN exactly at initialization.
        # The adapter convolutions keep their normal initialization so the gate
        # receives a useful gradient on the first optimization step.
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, feature: Tensor) -> Tensor:
        return feature + self.scale * self.adapter(feature)


class IRResidualNeck(nn.Module):
    """Apply independent residual adapters to an ordered FPN pyramid for IR only.

    Domain routing stays explicit through ``is_ir``. A non-IR call returns the
    original mapping object, avoiding both computation and accidental changes to
    RGB features. The IR path preserves every key and tensor shape.
    """

    def __init__(
        self,
        channels: int = 256,
        bottleneck_channels: int = 64,
        num_levels: int = 5,
        norm_groups: int = 16,
    ) -> None:
        super().__init__()
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")

        self.num_levels = num_levels
        self.adapters = nn.ModuleList(
            ResidualAdapterBlock(
                channels=channels,
                bottleneck_channels=bottleneck_channels,
                norm_groups=norm_groups,
            )
            for _ in range(num_levels)
        )

    def forward(
        self,
        features: Mapping[str, Tensor],
        *,
        is_ir: bool,
    ) -> Mapping[str, Tensor]:
        if not is_ir:
            return features
        if len(features) != self.num_levels:
            raise ValueError(
                f"IRResidualNeck expected {self.num_levels} FPN levels, "
                f"but received {len(features)}"
            )

        return OrderedDict(
            (name, adapter(feature))
            for (name, feature), adapter in zip(features.items(), self.adapters)
        )


__all__ = ["IRResidualNeck", "ResidualAdapterBlock"]

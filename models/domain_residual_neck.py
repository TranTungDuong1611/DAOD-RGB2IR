"""Post-FPN residual bottlenecks with separate RGB/IR adapters and channel SE."""

from collections import OrderedDict
from collections.abc import Mapping

from torch import Tensor, nn

from .outward_reference import OutwardReference


def normalize_domain(domain: str) -> str:
    """Validate the selected RGB/IR branch; thermal aliases IR."""
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    normalized = domain.strip().lower()
    if normalized == "thermal":
        normalized = "ir"
    if normalized not in {"rgb", "ir"}:
        raise ValueError("domain must be 'rgb', 'ir', or 'thermal'")
    return normalized


class SEBlock(nn.Module):
    """Generate one sigmoid weight per channel and broadcast over H/W."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class DomainBottleneck(nn.Module):
    """Shared convolutions plus domain adapters, domain SE and projected skip.

    Matches the approved block: adapters consume conv1's activated output in
    parallel with conv2. The learned skip and final ReLU are intentional; this
    module does not start as an identity and has no scalar residual gate.
    """

    def __init__(
        self, in_channels: int = 256, mid_channels: int = 64,
        out_channels: int = 256, reduction: int = 16,
        *, out_ref_enabled: bool = False, out_ref_channels: int = 16,
        out_ref_batch_norm: bool = True,
    ):
        super().__init__()
        if min(in_channels, mid_channels, out_channels, reduction) <= 0:
            raise ValueError("channel dimensions and reduction must be positive")
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.rgb_adapter = nn.Conv2d(mid_channels, mid_channels, 1)
        self.ir_adapter = nn.Conv2d(mid_channels, mid_channels, 1)
        self.rgb_se = SEBlock(out_channels, reduction)
        self.ir_se = SEBlock(out_channels, reduction)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.out_ref = (
            OutwardReference(out_channels, out_ref_channels, out_ref_batch_norm)
            if out_ref_enabled else None
        )

    def forward(self, x: Tensor, domain: str) -> Tensor:
        return self.forward_with_aux(x, domain)[0]

    def forward_with_aux(self, x: Tensor, domain: str):
        domain = normalize_domain(domain)
        identity = self.skip(x)
        h = self.relu(self.conv1(x))
        adapter = self.rgb_adapter if domain == "rgb" else self.ir_adapter
        h = self.relu(self.conv2(h) + adapter(h))
        h = self.conv3(h)
        attention = self.rgb_se if domain == "rgb" else self.ir_se
        channel_feature = attention(h)
        edge_logits = None
        if self.out_ref is not None:
            spatial_attention, edge_logits = self.out_ref(h)
            channel_feature = channel_feature * spatial_attention
        return self.relu(channel_feature + identity), edge_logits


class DomainResidualNeck(nn.Module):
    """Apply an independent bottleneck to each ordered feature pyramid level."""

    def __init__(
        self, channels: int = 256, bottleneck_channels: int = 64,
        num_levels: int = 5, se_reduction: int = 16,
        *, out_ref_enabled: bool = False, out_ref_channels: int = 16,
        out_ref_batch_norm: bool = True,
    ):
        super().__init__()
        if min(channels, bottleneck_channels, num_levels, se_reduction) <= 0:
            raise ValueError("dimensions, num_levels and se_reduction must be positive")
        self.num_levels = num_levels
        self.blocks = nn.ModuleList(
            DomainBottleneck(
                channels, bottleneck_channels, channels, se_reduction,
                out_ref_enabled=out_ref_enabled and level < 2,
                out_ref_channels=out_ref_channels, out_ref_batch_norm=out_ref_batch_norm,
            )
            for level in range(num_levels)
        )

    def initialize_ir_from_rgb(self, source=None) -> None:
        """Copy learned RGB branches, preserving IR Parameters and shared weights.

        A teacher can use the student's RGB branches as its source without
        resetting its own RGB branches or EMA-trained shared convolutions.
        """
        source = self if source is None else source
        for target_block, source_block in zip(self.blocks, source.blocks, strict=True):
            target_block.ir_adapter.load_state_dict(source_block.rgb_adapter.state_dict())
            target_block.ir_se.load_state_dict(source_block.rgb_se.state_dict())

    def forward(
        self, features: Mapping[str, Tensor], *, domain: str,
    ) -> Mapping[str, Tensor]:
        return self.forward_with_aux(features, domain=domain)[0]

    def forward_with_aux(self, features: Mapping[str, Tensor], *, domain: str):
        domain = normalize_domain(domain)
        if len(features) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} FPN levels, got {len(features)}")
        output, logits = OrderedDict(), []
        for (name, feature), block in zip(features.items(), self.blocks):
            output[name], edge = block.forward_with_aux(feature, domain)
            if edge is not None:
                logits.append(edge)
        return output, tuple(logits)

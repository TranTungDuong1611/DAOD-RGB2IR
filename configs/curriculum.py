"""Routing and iteration policies for the original curriculum workflow."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class StepRouting:
    """
    Defines the behavior for a specific training step.
    """
    ema_target: str = "none"      # Which teacher to update: "rgb" | "ir" | "none"
    use_gt: bool = True           # Whether to include Supervised Ground Truth loss
    student_saga_level: str = "rgb"  # "rgb" | "weak" | "mid" | "high" | "ir"
    teacher_saga_level: str = "rgb"  # "rgb" | "weak" | "mid" | "high" | "ir"
    teacher_names: Tuple[str, ...] = ()


@dataclass
class MidRoutingConfig:
    """
    Routes the data flow and EMA updates based on the DomainStep
    returned by the CurriculumScheduler.
    """
    # PHASE 1: Supervised Warmup
    p1_rgb_supervised: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="none", use_gt=True, student_saga_level="rgb", teacher_saga_level="rgb",
        teacher_names=(),
    ))

    # PHASE 2: Transition (RGB Dominant)
    p2_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="weak", teacher_saga_level="rgb",
        teacher_names=("rgb",),
    ))
    p2_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=True, student_saga_level="high", teacher_saga_level="mid",
        teacher_names=("ir",),
    ))

    # PHASE 3: Adaptation (IR Dominant)
    p3_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="mid", teacher_saga_level="weak",
        teacher_names=("rgb",),
    ))
    p3_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="high",
        teacher_names=("ir",),
    ))

    # PHASE 4: supervised RGB steps retain source-domain supervision, while
    # the existing IR route continues distillation and IR-teacher EMA updates.
    p4_rgb_supervised: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="none", use_gt=True,
        student_saga_level="rgb", teacher_saga_level="rgb", teacher_names=(),
    ))

    p4_ir_focus: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="ir",
        teacher_names=("ir",),
    ))

    def get_routing(self, step_name: str) -> StepRouting:
        """Helper to fetch routing params using the step name string."""
        return getattr(self, step_name, self.p4_ir_focus)


@dataclass
class CurriculumConfig:
    """Phase boundaries and step ratios for alternating domain training."""
    phase1_end: int = 2000
    phase2_end: int = 5000
    phase3_end: int = 8000

    phase2_rgb_sampling_ratio: float = 0.7
    phase3_rgb_sampling_ratio: float = 0.3
    sampling_mode: str = "grouped"
    phase4_rgb_sampling_ratio: float = 0.2

    def __post_init__(self):
        if self.sampling_mode not in {"grouped", "interleaved"}:
            raise ValueError("sampling_mode must be 'grouped' or 'interleaved'")
        boundaries = (self.phase1_end, self.phase2_end, self.phase3_end)
        if any(not isinstance(value, int) or value < 0 for value in boundaries):
            raise ValueError("phase boundaries must be non-negative integers")
        if tuple(boundaries) != tuple(sorted(boundaries)):
            raise ValueError("phase boundaries must be increasing")
        for name in (
            "phase2_rgb_sampling_ratio", "phase3_rgb_sampling_ratio",
            "phase4_rgb_sampling_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


__all__ = ['StepRouting', 'MidRoutingConfig', 'CurriculumConfig']

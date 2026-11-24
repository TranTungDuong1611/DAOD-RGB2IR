"""Teacher selection, EMA and supervised/distillation loss settings."""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from .types import Phase


@dataclass
class TeacherSchedule:
    """Holds (ratio, min_hm) pairs for each curriculum phase."""
    # Format: (ratio, min_hm)
    phase1: Tuple[float, float]
    phase2: Tuple[float, float]
    phase3: Tuple[float, float]
    phase4: Tuple[float, float]

    def __post_init__(self):
        for phase_name in ("phase1", "phase2", "phase3", "phase4"):
            ratio, min_hm = getattr(self, phase_name)
            if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
                raise ValueError(f"{phase_name} top ratio must be in (0, 1]")
            if not math.isfinite(min_hm) or not 0.0 <= min_hm <= 1.0:
                raise ValueError(f"{phase_name} min_hm must be in [0, 1]")

    def get_params(self, phase: Phase) -> Tuple[float, float]:
        mapping = {
            Phase.PHASE1_RGB_WARMUP: self.phase1,
            Phase.PHASE2_TRANSITION:    self.phase2,
            Phase.PHASE3_ADAPTATION:     self.phase3,
            Phase.PHASE4_IR_FOCUS:   self.phase4,
        }
        return mapping[phase]


@dataclass
class DistillConfig:
    # Adaptive Threshold Schedules for RGB and IR Teachers
    # RGB Teacher: Learns faster, more reliable early on
    rgb_teacher: TeacherSchedule = field(default_factory=lambda: TeacherSchedule(
        phase1=(0.010, 0.45), phase2=(0.020, 0.35),
        phase3=(0.040, 0.25), phase4=(0.060, 0.20)
    ))

    # IR Teacher: Noisier, requires stricter filtering in early phases
    ir_teacher: TeacherSchedule = field(default_factory=lambda: TeacherSchedule(
        phase1=(0.002, 0.60), phase2=(0.005, 0.50),
        phase3=(0.015, 0.35), phase4=(0.030, 0.25)
    ))

    # Harmony Measure (HM) parameters: (prob^alpha) * (iou^beta)
    hm_alpha: float = 0.5
    hm_beta: float = 0.5

    # Uncertainty Weighting: weight = exp(-(1-HM) / un_regular_alpha)
    un_regular_alpha: float = 1.0
    kd_gamma: float = 2.0

    def __post_init__(self):
        if self.hm_alpha < 0 or self.hm_beta < 0 or self.un_regular_alpha <= 0:
            raise ValueError("invalid HM/uncertainty settings")


@dataclass
class EMAConfig:
    """Exponential Moving Average settings for teacher update."""
    alpha: float = 0.9996          # EMA decay factor (higher = slower teacher update)
    start_steps: Optional[int] = None

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("EMA alpha must be in [0, 1]")
        if self.start_steps is not None and (
            not isinstance(self.start_steps, int) or self.start_steps < 0
        ):
            raise ValueError("EMA start_steps must be None or a non-negative integer")


@dataclass
class LossConfig:
    """
    Phase-based loss weights to control the balance between
    Supervised (Ground Truth) and Unsupervised (Knowledge Distillation) learning.
    """
    weight_logits: float = 4.0
    weight_deltas: float = 1.0
    weight_quality: float = 1.0
    weight_uhl: float = 1.0
    color_consistency_weight: float = 0.1
    color_consistency_levels: Tuple[str, ...] = ("p3", "p4", "p5")
    color_consistency_level_weights: Tuple[float, ...] = (1.0, 1.0, 1.0)

    # Phase 1: Pure Supervised Warmup (RGB + GT)
    p1_sup_weight: float = 1.0
    p1_distill_weight: float = 0.0

    # Phase 2: Transition (RGB/IR flows with GT + Distill)
    p2_sup_weight: float = 1.0
    p2_distill_weight: float = 0.7

    # Phase 3: Adaptation (Shift weight towards Distill)
    p3_sup_weight: float = 0.5
    p3_distill_weight: float = 1.0

    # Phase 4: GT applies only to RGB steps; unlabeled IR still uses KD only.
    p4_sup_weight: float = 1.0
    p4_distill_weight: float = 1.0

    def __post_init__(self):
        self.color_consistency_levels = tuple(self.color_consistency_levels)
        self.color_consistency_level_weights = tuple(self.color_consistency_level_weights)
        if not math.isfinite(self.color_consistency_weight) or self.color_consistency_weight < 0:
            raise ValueError("color_consistency_weight must be finite and non-negative")
        levels = self.color_consistency_levels
        weights = self.color_consistency_level_weights
        if not levels or len(set(levels)) != len(levels) or any(level not in {"p3", "p4", "p5"} for level in levels):
            raise ValueError("color_consistency_levels must be unique P3-P5 names")
        if len(weights) != len(levels) or any(not math.isfinite(w) or w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise ValueError("color consistency level weights must match levels and have a finite positive sum")
        if not math.isfinite(sum(weights)):
            raise ValueError("color consistency level weight sum must be finite")

    def get_phase_weights(self, phase: Phase) -> Tuple[float, float]:
        """Helper to retrieve (supervised_weight, distillation_weight) for a given phase."""
        mapping = {
            Phase.PHASE1_RGB_WARMUP: (self.p1_sup_weight, self.p1_distill_weight),
            Phase.PHASE2_TRANSITION: (self.p2_sup_weight, self.p2_distill_weight),
            Phase.PHASE3_ADAPTATION: (self.p3_sup_weight, self.p3_distill_weight),
            Phase.PHASE4_IR_FOCUS:   (self.p4_sup_weight, self.p4_distill_weight),
        }
        return mapping[phase]


__all__ = ['TeacherSchedule', 'DistillConfig', 'EMAConfig', 'LossConfig']

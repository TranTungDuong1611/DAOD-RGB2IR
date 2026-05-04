"""
AdaptiveThresholdScheduler — curriculum-based pseudo-label confidence threshold.

Problem with fixed threshold:
  - Early training: teacher is weak → fixed 0.7 may still pass noisy boxes
  - Late training:  teacher is strong → fixed 0.7 too strict → drops valid boxes

Strategy:
  Phase 1 (RGB warmup)  → high threshold (ir_teacher not trained yet)
  Phase 2 (RGB + MID)   → medium-high (ir_teacher begins learning)
  Phase 3 (MID + IR)    → medium (both teachers improving)
  Phase 4 (IR focus)    → lower threshold (teachers are good)

Each teacher can have an independent threshold since:
  rgb_teacher adapts faster (trained since phase 1)
  ir_teacher adapts slower (meaningful update only from phase 2/3)
"""

from dataclasses import dataclass, field
from typing import Optional

from scheduler import Phase


# ---------------------------------------------------------------------------
# Per-teacher threshold config
# ---------------------------------------------------------------------------

@dataclass
class TeacherThresholds:
    """Confidence thresholds per phase for a single teacher model."""
    phase1: float = 0.90   # RGB warmup
    phase2: float = 0.75   # RGB + mid_near_rgb
    phase3: float = 0.60   # full intermediate
    phase4: float = 0.50   # mid_near_ir + IR
    phase5: float = 0.45   # full IR — teacher most reliable

    def get(self, phase: Phase) -> float:
        mapping = {
            Phase.PHASE1_RGB_WARMUP:   self.phase1,
            Phase.PHASE2_RGB_NEAR_RGB: self.phase2,
            Phase.PHASE3_INTERMEDIATE: self.phase3,
            Phase.PHASE4_NEAR_IR_MIX:  self.phase4,
            Phase.PHASE5_IR_FOCUS:     self.phase5,
        }
        return mapping.get(phase, 0.7)


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive confidence thresholds.

    rgb_teacher trains earlier → becomes reliable earlier.
    ir_teacher is frozen until Phase 4 → needs higher threshold initially.
    """
    rgb_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.85, phase2=0.70, phase3=0.55, phase4=0.45, phase5=0.40,
    ))
    ir_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.95, phase2=0.80, phase3=0.65, phase4=0.55, phase5=0.45,
    ))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AdaptiveThresholdScheduler:
    """
    Returns per-teacher confidence thresholds based on the current curriculum phase.

    Usage:
        thresh_sched = AdaptiveThresholdScheduler(config)

        # In each training step:
        phase = scheduler.get_phase(global_step)
        rgb_thresh = thresh_sched.rgb_teacher(phase)
        ir_thresh  = thresh_sched.ir_teacher(phase)
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.config = config or AdaptiveThresholdConfig()

    def rgb_teacher(self, phase: Phase) -> float:
        """Threshold for filtering rgb_teacher pseudo-labels."""
        return self.config.rgb_teacher.get(phase)

    def ir_teacher(self, phase: Phase) -> float:
        """Threshold for filtering ir_teacher pseudo-labels."""
        return self.config.ir_teacher.get(phase)

    def both(self, phase: Phase) -> float:
        """
        Single shared threshold (average of both teachers).
        Use when you don't distinguish between teacher sources.
        """
        return (self.rgb_teacher(phase) + self.ir_teacher(phase)) / 2.0

    def summary(self) -> str:
        lines = ["AdaptiveThresholdScheduler:"]
        for p in Phase:
            rt = self.rgb_teacher(p)
            it = self.ir_teacher(p)
            lines.append(f"  {p.name:<25s}  rgb_teacher={rt:.2f}  ir_teacher={it:.2f}")
        return "\n".join(lines)

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
from typing import Dict, Optional

from scheduler import Phase


# ---------------------------------------------------------------------------
# Per-teacher threshold config
# ---------------------------------------------------------------------------

@dataclass
class TeacherThresholds:
    """Confidence thresholds per phase for a single teacher model."""
    phase1: float = 0.90   # RGB warmup — teacher predictions least reliable
    phase2: float = 0.75   # RGB+MID transition
    phase3: float = 0.60   # MID+IR — teacher significantly improved
    phase4: float = 0.50   # IR focus — teacher reliable, mine more boxes

    def get(self, phase: Phase) -> float:
        mapping = {
            Phase.PHASE1_RGB_WARMUP: self.phase1,
            Phase.PHASE2_RGB_MID:    self.phase2,
            Phase.PHASE3_MID_IR:     self.phase3,
            Phase.PHASE4_IR_FOCUS:   self.phase4,
        }
        return mapping.get(phase, 0.7)


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive confidence thresholds.

    rgb_teacher and ir_teacher can have different schedules because
    rgb_teacher trains earlier and thus becomes reliable earlier.
    """
    rgb_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.85, phase2=0.70, phase3=0.55, phase4=0.45,
    ))
    ir_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.95, phase2=0.80, phase3=0.65, phase4=0.50,
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

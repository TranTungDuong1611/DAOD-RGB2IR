"""
AdaptiveThresholdScheduler — curriculum-based pseudo-label confidence threshold.

Strategy:
  Phase 1 (RGB warmup)  → high threshold (teachers not trained yet)
  Phase 2 (MID / SAGA)  → medium (both teachers learning on SAGA domain)
  Phase 3 (IR focus)    → lower threshold (ir_teacher is mature)

Each teacher can have an independent threshold since:
  rgb_teacher adapts earlier (trained from phase 1)
  ir_teacher gets meaningful updates only from phase 2
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from scheduler import Phase

# Union type for global or class-wise threshold
ThreshType = Union[float, Dict[int, float]]


# ---------------------------------------------------------------------------
# Per-teacher threshold config
# ---------------------------------------------------------------------------

@dataclass
class TeacherThresholds:
    """
    Confidence thresholds per phase for a single teacher model.

    Each phase value can be:
      - float              : global threshold applied to all classes
      - Dict[int, float]   : per-class threshold, e.g. {0: 0.6, 1: 0.7, 2: 0.8}
                             Classes absent from the dict fall back to 0.7.
    """
    phase1: ThreshType = 0.90   # RGB warmup
    phase2: ThreshType = 0.70   # MID (SAGA)
    phase3: ThreshType = 0.50   # IR focus

    def get(self, phase: Phase) -> ThreshType:
        mapping = {
            Phase.PHASE1_RGB_WARMUP: self.phase1,
            Phase.PHASE2_RGB_MID:    self.phase2,
            Phase.PHASE3_IR_FOCUS:   self.phase3,
        }
        return mapping.get(phase, 0.7)


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive confidence thresholds.

    rgb_teacher trains earlier → becomes reliable earlier.
    ir_teacher is frozen until Phase 2 → needs higher threshold initially.
    """
    rgb_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.85, phase2=0.65, phase3=0.50,
    ))
    ir_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.95, phase2=0.75, phase3=0.55,
    ))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AdaptiveThresholdScheduler:
    """
    Returns per-teacher confidence thresholds based on the current curriculum phase.

    Usage:
        thresh_sched = AdaptiveThresholdScheduler(config)

        phase = scheduler.get_phase(global_step)
        rgb_thresh = thresh_sched.rgb_teacher(phase)
        ir_thresh  = thresh_sched.ir_teacher(phase)
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.config = config or AdaptiveThresholdConfig()

    def rgb_teacher(self, phase: Phase) -> ThreshType:
        """Threshold for filtering rgb_teacher pseudo-labels."""
        return self.config.rgb_teacher.get(phase)

    def ir_teacher(self, phase: Phase) -> ThreshType:
        """Threshold for filtering ir_teacher pseudo-labels."""
        return self.config.ir_teacher.get(phase)

    def both(self, phase: Phase) -> ThreshType:
        """
        Shared threshold when both teachers are used.
        Returns the rgb_teacher threshold (lower = more permissive).
        """
        return self.config.rgb_teacher.get(phase)

    def summary(self) -> str:
        def _fmt(v: ThreshType) -> str:
            if isinstance(v, dict):
                return "{" + ", ".join(f"{k}:{t:.2f}" for k, t in sorted(v.items())) + "}"
            return f"{v:.2f}"

        lines = ["AdaptiveThresholdScheduler:"]
        for p in Phase:
            rt = self.rgb_teacher(p)
            it = self.ir_teacher(p)
            lines.append(f"  {p.name:<25s}  rgb_teacher={_fmt(rt)}  ir_teacher={_fmt(it)}")
        return "\n".join(lines)

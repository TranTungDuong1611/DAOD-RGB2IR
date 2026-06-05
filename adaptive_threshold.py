"""
AdaptiveThresholdScheduler — curriculum-based pseudo-label confidence threshold.

4-phase strategy:
  Phase 1 (RGB warmup)        → high threshold (teachers not trained yet)
  Phase 2 (mixed RGB + MID)   → medium (rgb_teacher specializes)
  Phase 3 (mixed MID + IR)    → medium (ir_teacher specializes)
  Phase 4 (IR focus)          → starts low, ramps UP linearly to suppress FPs

Each teacher can have an independent threshold per phase. The Phase 4 linear
ramp applies to ir_teacher only (rgb_teacher is no longer updated past Phase 2).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from scheduler import Phase

ThreshType = Union[float, Dict[int, float]]


# ---------------------------------------------------------------------------
# Ramp config
# ---------------------------------------------------------------------------

@dataclass
class ThreshRampConfig:
    """
    Linear threshold ramp-up within Phase 4 (IR focus).

    enabled     : set True to activate
    start       : threshold at the first step of Phase 4
                  float → global,  Dict[int, float] → per-class
    end         : threshold after ramp_steps steps (stays here afterwards)
                  same type as start
    ramp_steps  : number of Phase 4 steps to ramp over
    """
    enabled:    bool       = False
    start:      ThreshType = 0.55
    end:        ThreshType = 0.80
    ramp_steps: int        = 5_000

    def get(self, steps_into_phase: int) -> ThreshType:
        if not self.enabled or self.ramp_steps <= 0:
            return self.start
        t = min(1.0, steps_into_phase / self.ramp_steps)
        return self._interp(self.start, self.end, t)

    @staticmethod
    def _interp(start: ThreshType, end: ThreshType, t: float) -> ThreshType:
        """Linearly interpolate between start and end (float or per-class dict)."""
        if isinstance(start, dict) and isinstance(end, dict):
            return {
                cls: s + t * (end.get(cls, s) - s)
                for cls, s in start.items()
            }
        s = start if isinstance(start, (int, float)) else 0.7
        e = end   if isinstance(end,   (int, float)) else 0.7
        return s + t * (e - s)


# ---------------------------------------------------------------------------
# Per-teacher threshold config
# ---------------------------------------------------------------------------

@dataclass
class TeacherThresholds:
    """
    Confidence thresholds per phase.

    Each value can be:
      float            — global threshold for all classes
      Dict[int, float] — per-class threshold; missing classes fall back to 0.7
    """
    phase1: ThreshType = 0.90
    phase2: ThreshType = 0.70
    phase3: ThreshType = 0.60
    phase4: ThreshType = 0.55   # base for Phase 4 (overridden by ramp if enabled)

    def get(self, phase: Phase) -> ThreshType:
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

    phase4_ir_ramp applies only to ir_teacher in Phase 4.
    rgb_teacher does not ramp (it is not updated past Phase 2).
    """
    rgb_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.85, phase2=0.65, phase3=0.55, phase4=0.55,
    ))
    ir_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.95, phase2=0.75, phase3=0.65, phase4=0.55,
    ))
    phase4_ir_ramp: ThreshRampConfig = field(default_factory=ThreshRampConfig)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AdaptiveThresholdScheduler:
    """
    Returns per-teacher confidence thresholds based on phase and step.

    Usage:
        phase = curriculum_scheduler.get_phase(global_step)
        steps_into_phase = max(0, global_step - curriculum.phase3_end)  # Phase 4 ramp

        rgb_thresh = thresh_sched.rgb_teacher(phase)
        ir_thresh  = thresh_sched.ir_teacher(phase, steps_into_phase)
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.config = config or AdaptiveThresholdConfig()

    def rgb_teacher(self, phase: Phase, steps_into_phase: int = 0) -> ThreshType:
        """Threshold for rgb_teacher — no ramp (rgb_teacher frozen after Phase 2)."""
        return self.config.rgb_teacher.get(phase)

    def ir_teacher(self, phase: Phase, steps_into_phase: int = 0) -> ThreshType:
        """
        Threshold for ir_teacher.
        In Phase 4, applies linear ramp-up if phase4_ir_ramp.enabled.
        """
        if phase == Phase.PHASE4_IR_FOCUS and self.config.phase4_ir_ramp.enabled:
            return self.config.phase4_ir_ramp.get(steps_into_phase)
        return self.config.ir_teacher.get(phase)

    def both(self, phase: Phase, steps_into_phase: int = 0) -> ThreshType:
        """Shared threshold when both teachers are used (mixed-batch steps)."""
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
            lines.append(f"  {p.name:<25s}  rgb={_fmt(rt)}  ir={_fmt(it)}")

        ramp = self.config.phase4_ir_ramp
        if ramp.enabled:
            lines.append(
                f"  Phase4 ir ramp: {_fmt(ramp.start)} → {_fmt(ramp.end)} "
                f"over {ramp.ramp_steps} steps"
            )
        return "\n".join(lines)

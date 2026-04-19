"""
CurriculumScheduler — decides which domain step to execute at each iteration.

Curriculum (NOT zigzag):
  Phase 1  [0,          phase1_end)  → RGB only        (supervised warmup)
  Phase 2  [phase1_end, phase2_end)  → RGB + MID       (bridge begins)
  Phase 3  [phase2_end, phase3_end)  → MID + IR        (bridge ends)
  Phase 4  [phase3_end, ∞)           → IR focus         (final adaptation)

Within-phase alternation is ratio-based:
  e.g. phase2_rgb_ratio=0.5 → RGBMIDRGBMID...
       phase2_rgb_ratio=0.67 → RGBRGBMIDRGBRGBMID...
"""

from enum import Enum, auto
from typing import Literal

from .config import CurriculumConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(Enum):
    PHASE1_RGB_WARMUP = 1
    PHASE2_RGB_MID    = 2
    PHASE3_MID_IR     = 3
    PHASE4_IR_FOCUS   = 4


DomainStep = Literal["rgb", "mid", "ir"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Stateful curriculum scheduler.

    Call `get_next_step(global_step)` at every iteration.
    The scheduler tracks an internal counter per phase to implement
    the within-phase ratio — this counter is independent of global_step
    so phase transitions reset the alternation pattern cleanly.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        # Internal alternation counters — one per phase that needs them
        self._counters = {
            Phase.PHASE2_RGB_MID: 0,
            Phase.PHASE3_MID_IR:  0,
            Phase.PHASE4_IR_FOCUS: 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_phase(self, global_step: int) -> Phase:
        """Return the curriculum phase for a given global_step."""
        cfg = self.config
        if global_step < cfg.phase1_end:
            return Phase.PHASE1_RGB_WARMUP
        elif global_step < cfg.phase2_end:
            return Phase.PHASE2_RGB_MID
        elif global_step < cfg.phase3_end:
            return Phase.PHASE3_MID_IR
        else:
            return Phase.PHASE4_IR_FOCUS

    def get_next_step(self, global_step: int) -> DomainStep:
        """
        Determine (and advance) the next domain step to execute.

        Must be called exactly once per iteration in order.
        """
        phase = self.get_phase(global_step)

        if phase == Phase.PHASE1_RGB_WARMUP:
            return "rgb"

        elif phase == Phase.PHASE2_RGB_MID:
            return self._alternate(
                phase=Phase.PHASE2_RGB_MID,
                primary="rgb",
                secondary="mid",
                primary_ratio=self.config.phase2_rgb_ratio,
            )

        elif phase == Phase.PHASE3_MID_IR:
            return self._alternate(
                phase=Phase.PHASE3_MID_IR,
                primary="mid",
                secondary="ir",
                primary_ratio=self.config.phase3_mid_ratio,
            )

        else:  # PHASE4_IR_FOCUS
            count = self._counters[Phase.PHASE4_IR_FOCUS]
            self._counters[Phase.PHASE4_IR_FOCUS] += 1
            # Inject occasional MID for training stability
            if count % self.config.phase4_mid_every_n == 0:
                return "mid"
            return "ir"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alternate(
        self,
        phase: Phase,
        primary: DomainStep,
        secondary: DomainStep,
        primary_ratio: float,
    ) -> DomainStep:
        """
        Alternate between two domain steps according to primary_ratio.

        primary_ratio=0.5  → primary, secondary, primary, secondary, ...
        primary_ratio=0.67 → primary, primary, secondary, primary, primary, ...
        """
        period = self._ratio_to_period(primary_ratio)
        n_primary = max(1, round(period * primary_ratio))

        count = self._counters[phase]
        self._counters[phase] += 1

        position = count % period
        return primary if position < n_primary else secondary

    @staticmethod
    def _ratio_to_period(ratio: float) -> int:
        """
        Convert a ratio in (0, 1) to the smallest integer cycle period.

        Examples:
          0.5  → 2  (1 primary + 1 secondary per cycle)
          0.33 → 3  (1 primary + 2 secondary per cycle)
          0.67 → 3  (2 primary + 1 secondary per cycle)
          0.25 → 4  (1 primary + 3 secondary per cycle)
        """
        ratio = max(1e-6, min(1.0 - 1e-6, ratio))
        # period = round of 1 / min(ratio, 1-ratio)
        smaller = min(ratio, 1.0 - ratio)
        return max(2, round(1.0 / smaller))

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CurriculumScheduler("
            f"phase1_end={cfg.phase1_end}, "
            f"phase2_end={cfg.phase2_end}, "
            f"phase3_end={cfg.phase3_end})"
        )

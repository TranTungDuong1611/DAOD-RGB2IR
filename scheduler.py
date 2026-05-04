"""
CurriculumScheduler — decides which domain step to execute at each iteration.

Curriculum:
  Phase 1  [0,           phase1_end)  → RGB only             (supervised warmup)
  Phase 2  [phase1_end,  phase2_end)  → RGB + mid_near_rgb   (bridge starts)
  Phase 3  [phase2_end,  phase3_end)  → mid_intermediate      (full bridge)
  Phase 4  [phase3_end,  phase4_end)  → mid_near_ir + IR     (IR adaptation)
  Phase 5  [phase4_end,  ∞)           → IR only              (IR focus)

Within-phase alternation is ratio-based:
  phase2_rgb_ratio=0.67  → RGB RGB mid_near_rgb  RGB RGB mid_near_rgb  ...
  phase4_mid_ratio=0.67  → mid_near_ir mid_near_ir ir  mid_near_ir ...
"""

from enum import Enum
from typing import Literal

from config import CurriculumConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(Enum):
    PHASE1_RGB_WARMUP   = 1
    PHASE2_RGB_NEAR_RGB = 2
    PHASE3_INTERMEDIATE = 3
    PHASE4_NEAR_IR_MIX  = 4
    PHASE5_IR_FOCUS     = 5


DomainStep = Literal["rgb", "mid_near_rgb", "mid_intermediate", "mid_near_ir", "ir"]


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
        self._counters = {
            Phase.PHASE2_RGB_NEAR_RGB: 0,
            Phase.PHASE4_NEAR_IR_MIX:  0,
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
            return Phase.PHASE2_RGB_NEAR_RGB
        elif global_step < cfg.phase3_end:
            return Phase.PHASE3_INTERMEDIATE
        elif global_step < cfg.phase4_end:
            return Phase.PHASE4_NEAR_IR_MIX
        else:
            return Phase.PHASE5_IR_FOCUS

    def get_next_step(self, global_step: int) -> DomainStep:
        """
        Determine (and advance) the next domain step to execute.
        Must be called exactly once per iteration in order.
        """
        phase = self.get_phase(global_step)

        if phase == Phase.PHASE1_RGB_WARMUP:
            return "rgb"

        elif phase == Phase.PHASE2_RGB_NEAR_RGB:
            return self._alternate(
                phase=Phase.PHASE2_RGB_NEAR_RGB,
                primary="rgb",
                secondary="mid_near_rgb",
                primary_ratio=self.config.phase2_rgb_ratio,
            )

        elif phase == Phase.PHASE3_INTERMEDIATE:
            return "mid_intermediate"

        elif phase == Phase.PHASE4_NEAR_IR_MIX:
            return self._alternate(
                phase=Phase.PHASE4_NEAR_IR_MIX,
                primary="mid_near_ir",
                secondary="ir",
                primary_ratio=self.config.phase4_mid_ratio,
            )

        else:  # PHASE5_IR_FOCUS
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

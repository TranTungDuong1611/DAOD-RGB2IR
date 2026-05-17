"""
CurriculumScheduler — decides which domain step to execute at each iteration.

Curriculum:
  Phase 1  [0,           phase1_end)  → RGB only        (supervised warmup)
  Phase 2  [phase1_end,  phase2_end)  → RGB + MID       (alternating, ratio-based)
  Phase 3  [phase2_end,  ∞)           → IR only          (IR focus)

Phase 2 alternation (phase2_rgb_ratio=0.5):
  rgb, mid, rgb, mid, ...
Phase 2 alternation (phase2_rgb_ratio=0.67):
  rgb, rgb, mid, rgb, rgb, mid, ...
"""

from enum import Enum
from typing import Literal

from config import CurriculumConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(Enum):
    PHASE1_RGB_WARMUP = 1
    PHASE2_RGB_MID    = 2
    PHASE3_IR_FOCUS   = 3


DomainStep = Literal["rgb", "mid", "ir"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Curriculum scheduler with within-phase alternation for Phase 2.

    Call `get_next_step(global_step)` exactly once per iteration in order.
    The internal counter resets on phase transition so the alternation pattern
    always starts cleanly.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self._phase2_counter: int = 0

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
        else:
            return Phase.PHASE3_IR_FOCUS

    def get_next_step(self, global_step: int) -> DomainStep:
        """Determine (and advance) the next domain step to execute."""
        phase = self.get_phase(global_step)

        if phase == Phase.PHASE1_RGB_WARMUP:
            return "rgb"

        elif phase == Phase.PHASE2_RGB_MID:
            return self._alternate(phase2_rgb_ratio=self.config.phase2_rgb_ratio)

        else:  # PHASE3_IR_FOCUS
            return "ir"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alternate(self, phase2_rgb_ratio: float) -> DomainStep:
        """
        Alternate between rgb and mid according to phase2_rgb_ratio.

        ratio=0.5  → rgb, mid, rgb, mid, ...
        ratio=0.67 → rgb, rgb, mid, rgb, rgb, mid, ...
        """
        period = self._ratio_to_period(phase2_rgb_ratio)
        n_rgb  = max(1, round(period * phase2_rgb_ratio))

        position = self._phase2_counter % period
        self._phase2_counter += 1

        return "rgb" if position < n_rgb else "mid"

    @staticmethod
    def _ratio_to_period(ratio: float) -> int:
        """
        Convert a ratio in (0, 1) to the smallest integer cycle period.

        Examples:
          0.5  → 2  (1 rgb + 1 mid per cycle)
          0.67 → 3  (2 rgb + 1 mid per cycle)
          0.33 → 3  (1 rgb + 2 mid per cycle)
        """
        ratio = max(1e-6, min(1.0 - 1e-6, ratio))
        smaller = min(ratio, 1.0 - ratio)
        return max(2, round(1.0 / smaller))

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CurriculumScheduler("
            f"phase1_end={cfg.phase1_end}, "
            f"phase2_end={cfg.phase2_end}, "
            f"phase2_rgb_ratio={cfg.phase2_rgb_ratio})"
        )

"""
CurriculumScheduler — decides which domain step to execute at each iteration.

Curriculum:
  Phase 1  [0,           phase1_end)  → RGB only   (supervised warmup)
  Phase 2  [phase1_end,  phase2_end)  → MID (SAGA 100%, both teachers)
  Phase 3  [phase2_end,  ∞)           → IR only    (IR focus)
"""

from enum import Enum
from typing import Literal

from config import CurriculumConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(Enum):
    PHASE1_RGB_WARMUP = 1
    PHASE2_MID        = 2
    PHASE3_IR_FOCUS   = 3


DomainStep = Literal["rgb", "mid", "ir"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Stateless curriculum scheduler — no alternation counters needed.

    Call `get_next_step(global_step)` at every iteration.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_phase(self, global_step: int) -> Phase:
        """Return the curriculum phase for a given global_step."""
        cfg = self.config
        if global_step < cfg.phase1_end:
            return Phase.PHASE1_RGB_WARMUP
        elif global_step < cfg.phase2_end:
            return Phase.PHASE2_MID
        else:
            return Phase.PHASE3_IR_FOCUS

    def get_next_step(self, global_step: int) -> DomainStep:
        """Determine the next domain step to execute."""
        phase = self.get_phase(global_step)
        if phase == Phase.PHASE1_RGB_WARMUP:
            return "rgb"
        elif phase == Phase.PHASE2_MID:
            return "mid"
        else:
            return "ir"

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"CurriculumScheduler("
            f"phase1_end={cfg.phase1_end}, "
            f"phase2_end={cfg.phase2_end})"
        )

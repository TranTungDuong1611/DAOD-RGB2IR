from typing import Literal
from .config import Phase 

# Định nghĩa các bước để Trainer biết loại ảnh và nhãn cần dùng
DomainStep = Literal[
    "p1_rgb_supervised",    # Student(RGB+GT)
    "p2_rgb_flow",          # Student(SAGA-Weak) + Teacher(RGB) + GT
    "p2_ir_flow",           # Student(SAGA-High) + Teacher(SAGA-Mid) + GT
    "p3_rgb_flow",          # Student(SAGA-Mid) + Teacher(SAGA-Weak) + GT
    "p3_ir_flow",           # Student(IR) + Teacher(SAGA-High) [NO GT]
    "p4_ir_focus"           # Student(IR) + Teacher(IR) [NO GT]
]

class CurriculumScheduler:
    """
    Handles within-phase alternation (RGB vs IR flows) 
    using ratios defined in TrainingConfig.
    """
    def __init__(self, config) -> None:
        self.config = config
        # Counters are internal to maintain strict alternating patterns
        
        self.Phase = Phase
        self._counters = {
            Phase.PHASE2_RGB_MID: 0,
            Phase.PHASE3_MID_IR:  0,
        }

    def get_next_step(self, global_step: int) -> DomainStep:
        """
        Main logic to decide what the Student and Teachers see in this iteration.
        """
        # Call the phase logic from the Master Config
        phase = self.config.get_phase(global_step)

        if phase == self.Phase.PHASE1_RGB_WARMUP:
            return "p1_rgb_supervised"

        elif phase == self.Phase.PHASE2_RGB_MID:
            # Phase 2: RGB dominant (e.g., 80% RGB flow, 20% IR flow)
            return self._alternate(
                phase=self.Phase.PHASE2_RGB_MID,
                primary="p2_rgb_flow",
                secondary="p2_ir_flow",
                primary_ratio=self.config.curriculum.phase2_rgb_ratio
            )

        elif phase == self.Phase.PHASE3_MID_IR:
            # Phase 3: IR dominant (e.g., 80% IR flow, 20% RGB flow)
            return self._alternate(
                phase=self.Phase.PHASE3_MID_IR,
                primary="p3_ir_flow",
                secondary="p3_rgb_flow",
                primary_ratio=self.config.curriculum.phase3_mid_ratio # This is IR ratio now
            )

        else: # PHASE4_IR_FOCUS
            return "p4_ir_focus"

    def _alternate(self, phase, primary: DomainStep, secondary: DomainStep, ratio: float) -> DomainStep:
        """Helper to alternate between two steps based on a ratio."""
        period = self._ratio_to_period(ratio)
        n_primary = max(1, round(period * ratio))

        count = self._counters[phase]
        self._counters[phase] += 1

        position = count % period
        return primary if position < n_primary else secondary

    @staticmethod
    def _ratio_to_period(ratio: float) -> int:
        """Converts a float ratio to a repeating cycle period (e.g., 0.8 -> 5)."""
        ratio = max(0.01, min(0.99, ratio))
        for d in range(2, 11): # Check for cycles up to 10
            if abs(ratio * d - round(ratio * d)) < 0.02:
                return d
        return 10
from fractions import Fraction
from typing import Literal
from config import Phase 

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
            Phase.PHASE2_TRANSITION: 0,
            Phase.PHASE3_ADAPTATION:  0,
        }

    def get_next_step(self, global_step: int) -> DomainStep:
        """
        Main logic to decide what the Student and Teachers see in this iteration.
        """
        # Call the phase logic from the Master Config
        phase = self.config.get_phase(global_step)

        if phase == self.Phase.PHASE1_RGB_WARMUP:
            return "p1_rgb_supervised"

        elif phase == self.Phase.PHASE2_TRANSITION:
            # Phase 2: RGB dominant (e.g., 80% RGB flow, 20% IR flow)
            return self._alternate(
                phase=self.Phase.PHASE2_TRANSITION,
                primary="p2_rgb_flow",
                secondary="p2_ir_flow",
                ratio=self.config.curriculum.phase2_rgb_sampling_ratio
            )

        elif phase == self.Phase.PHASE3_ADAPTATION:
            # The configured value is always the RGB route ratio.
            return self._alternate(
                phase=self.Phase.PHASE3_ADAPTATION,
                primary="p3_rgb_flow",
                secondary="p3_ir_flow",
                ratio=self.config.curriculum.phase3_rgb_sampling_ratio
            )

        else: # PHASE4_IR_FOCUS
            return "p4_ir_focus"

    def _alternate(self, phase, primary: DomainStep, secondary: DomainStep, ratio: float) -> DomainStep:
        """Helper to alternate between two steps based on a ratio."""
        period = self._ratio_to_period(ratio)
        n_primary = round(period * ratio)

        count = self._counters[phase]
        self._counters[phase] += 1

        position = count % period
        return primary if position < n_primary else secondary

    @staticmethod
    def _ratio_to_period(ratio: float) -> int:
        """Return a short exact period for the requested RGB ratio."""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("sampling ratio must be in [0, 1]")
        if ratio in (0.0, 1.0):
            return 1
        return Fraction(float(ratio)).limit_denominator(100).denominator

    def state_dict(self):
        return {phase.name: count for phase, count in self._counters.items()}

    def load_state_dict(self, state):
        for phase in self._counters:
            if phase.name in state:
                self._counters[phase] = int(state[phase.name])

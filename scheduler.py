from fractions import Fraction
from typing import Literal
from config import Phase 

# Route identifiers describe the curriculum action; physical source selection
# is resolved separately so an IR-named route can still use RGB/SAGA inputs.
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
        # Resolve the phase from configured iteration boundaries before choosing
        # a route. Phase transitions do not consume either data loader here.
        phase = self.config.get_phase(global_step)

        if phase == self.Phase.PHASE1_RGB_WARMUP:
            # Warmup stays on the labeled RGB path without teacher-driven routing.
            return "p1_rgb_supervised"

        elif phase == self.Phase.PHASE2_TRANSITION:
            # Transition alternates two RGB-backed view policies using the
            # configured RGB-route proportion, while retaining GT on both.
            return self._alternate(
                phase=self.Phase.PHASE2_TRANSITION,
                primary="p2_rgb_flow",
                secondary="p2_ir_flow",
                ratio=self.config.curriculum.phase2_rgb_sampling_ratio
            )

        elif phase == self.Phase.PHASE3_ADAPTATION:
            # Adaptation alternates the RGB/SAGA route and real unlabeled IR.
            # The configured proportion continues to mean the RGB-route share.
            return self._alternate(
                phase=self.Phase.PHASE3_ADAPTATION,
                primary="p3_rgb_flow",
                secondary="p3_ir_flow",
                ratio=self.config.curriculum.phase3_rgb_sampling_ratio
            )

        else: # PHASE4_IR_FOCUS
            # Final focus always selects the real-IR distillation route.
            return "p4_ir_focus"

    @staticmethod
    def get_source_domain(step_name: str) -> str:
        """Resolve the existing physical source policy, including RGB-backed P2 IR."""
        # Supply data flow with the actual loader domain, independently of the
        # teacher name or SAGA appearance level carried by the route settings.
        if step_name in {
            "p1_rgb_supervised", "p2_rgb_flow", "p2_ir_flow", "p3_rgb_flow",
        }:
            return "rgb"
        if step_name in {"p3_ir_flow", "p4_ir_focus"}:
            return "ir"
        raise ValueError(f"Unsupported curriculum step: {step_name}")

    def should_evaluate_pseudo_labels(self, global_step, step_name, *, last_eval_step):
        """Check the completed iteration's window without consuming a route.

        An RGB batch leaves evaluation pending for the next real IR KD pass.
        Compare global windows so a deferred audit does not shift future ticks.
        """
        cfg = self.config.pseudo_label_eval
        completed_step = global_step + 1
        return (
            cfg.enabled
            and self.get_source_domain(step_name) == "ir"
            and completed_step // cfg.interval > last_eval_step // cfg.interval
        )

    def _alternate(self, phase, primary: DomainStep, secondary: DomainStep, ratio: float) -> DomainStep:
        """Helper to alternate between two steps based on a ratio."""
        # Keep a deterministic per-phase route sequence. Persisted counters
        # allow resume to continue its position instead of restarting the cycle.
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
        # Persist route-selection progress independently from model/LR state.
        return {phase.name: count for phase, count in self._counters.items()}

    def load_state_dict(self, state):
        for phase in self._counters:
            if phase.name in state:
                self._counters[phase] = int(state[phase.name])

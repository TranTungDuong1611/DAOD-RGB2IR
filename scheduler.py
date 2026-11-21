from bisect import bisect_right
from fractions import Fraction
import math
from numbers import Real
from typing import Literal
from config import BridgeScheduleConfig, Phase


def should_evaluate_pseudo_labels(config, global_step, *, last_eval_step):
    """Keep fixed IR audits on global windows independently of domain routing."""
    completed_step = global_step + 1
    return (
        config.enabled
        and completed_step // config.interval > last_eval_step // config.interval
    )


class DomainRatioSampler:
    """Spread weighted domains deterministically, with insertion order breaking ties."""

    def __init__(self, ratios: dict[str, Real]):
        if not ratios or any(
            not isinstance(name, str) or not name.strip() for name in ratios
        ):
            raise ValueError("domain ratios require nonempty domain names")
        if any(
            not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 for value in ratios.values()
        ) or not any(value > 0 for value in ratios.values()):
            raise ValueError("domain ratios must be finite, nonnegative and have a positive sum")
        self._domains = tuple(ratios)
        # Rational arithmetic avoids float roundoff changing ties or scaled ratios.
        self._ratios = tuple(Fraction(str(value)) for value in ratios.values())
        total = sum(self._ratios)
        self._weights = tuple(value / total for value in self._ratios)
        self._active = tuple(i for i, weight in enumerate(self._weights) if weight > 0)
        self._count = 0
        self._allocations = [0] * len(self._domains)

    @property
    def count(self) -> int:
        return self._count

    def next_domain(self) -> str:
        self._count += 1
        selected = max(
            self._active,
            key=lambda i: self._weights[i] * self._count - self._allocations[i],
        )
        self._allocations[selected] += 1
        return self._domains[selected]

    def state_dict(self):
        return {
            "domains": list(self._domains),
            "ratios": [str(value) for value in self._ratios],
            "count": self._count,
            "allocations": list(self._allocations),
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        if state.get("domains") != expected["domains"] or state.get("ratios") != expected["ratios"]:
            raise ValueError("saved domain ratio schedule does not match")
        count, allocations = state.get("count"), state.get("allocations")
        if (type(count) is not int or count < 0
                or not isinstance(allocations, list) or len(allocations) != len(self._domains)
                or any(type(value) is not int or value < 0 for value in allocations)
                or sum(allocations) != count
                or any(allocations[i] for i, weight in enumerate(self._weights) if weight == 0)):
            raise ValueError("invalid saved domain allocation counts")
        self._count = count
        self._allocations = list(allocations)


class BridgeDomainScheduler:
    """Five-stage, four-domain schedule; iteration indices are zero-based and sequential."""

    DOMAINS = ("rgb", "saga", "ir_gan", "ir_real")

    def __init__(self, config: BridgeScheduleConfig, total_iters: int):
        if type(total_iters) is not int or total_iters <= 0:
            raise ValueError("total_iters must be a positive integer")
        # Snapshot validated configuration so later config mutation cannot change a run.
        config = BridgeScheduleConfig(config.phase_ratios, config.phase_fractions)
        self._ratios = config.phase_ratios
        self._fractions = config.phase_fractions
        self._ends = config.phase_ends(total_iters)
        self._samplers = self._new_samplers()
        self._next_step = 0

    def _new_samplers(self):
        return [DomainRatioSampler(dict(zip(self.DOMAINS, row))) for row in self._ratios]

    def phase_for_iter(self, global_step: int) -> int:
        if type(global_step) is not int or not 0 <= global_step < self._ends[-1]:
            raise ValueError("iteration must be an integer in the bridge schedule range")
        return bisect_right(self._ends, global_step)

    def get_next_domain(self, global_step: int) -> str:
        phase = self.phase_for_iter(global_step)
        if global_step != self._next_step:
            raise ValueError("bridge iterations must be sequential; restore state before resuming")
        domain = self._samplers[phase].next_domain()
        self._next_step += 1
        return domain

    def state_dict(self):
        return {
            "phase_ends": list(self._ends),
            "phase_fractions": list(self._fractions),
            "next_step": self._next_step,
            "samplers": [sampler.state_dict() for sampler in self._samplers],
        }

    def load_state_dict(self, state):
        if (state.get("phase_ends") != list(self._ends)
                or state.get("phase_fractions") != list(self._fractions)):
            raise ValueError("saved bridge schedule does not match")
        next_step, saved = state.get("next_step"), state.get("samplers")
        if (type(next_step) is not int or not 0 <= next_step <= self._ends[-1]
                or not isinstance(saved, list) or len(saved) != len(self._samplers)):
            raise ValueError("invalid saved bridge schedule progress")
        samplers = self._new_samplers()
        start = 0
        for sampler, sampler_state, end in zip(samplers, saved, self._ends):
            sampler.load_state_dict(sampler_state)
            if sampler.count != max(0, min(next_step, end) - start):
                raise ValueError("saved allocation counts do not match bridge iteration")
            start = end
        self._samplers, self._next_step = samplers, next_step

# Route identifiers describe the curriculum action; physical source selection
# is resolved separately so an IR-named route can still use RGB/SAGA inputs.
DomainStep = Literal[
    "p1_rgb_supervised",    # Student(RGB+GT)
    "p2_rgb_flow",          # Student(SAGA-Weak) + Teacher(RGB) + GT
    "p2_ir_flow",           # Student(SAGA-High) + Teacher(SAGA-Mid) + GT
    "p3_rgb_flow",          # Student(SAGA-Mid) + Teacher(SAGA-Weak) + GT
    "p3_ir_flow",           # Student(IR) + Teacher(SAGA-High) [NO GT]
    "p4_rgb_supervised",   # Student(RGB+GT), no KD or teacher update
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
            Phase.PHASE4_IR_FOCUS:    0,
        }
        self._samplers = self._new_interleaved_samplers()

    def _new_interleaved_samplers(self):
        return {
            phase: DomainRatioSampler({"primary": Fraction(str(ratio)), "secondary": 1 - Fraction(str(ratio))})
            for phase, ratio in (
                (Phase.PHASE2_TRANSITION, self.config.curriculum.phase2_rgb_sampling_ratio),
                (Phase.PHASE3_ADAPTATION, self.config.curriculum.phase3_rgb_sampling_ratio),
                (Phase.PHASE4_IR_FOCUS, self.config.curriculum.phase4_rgb_sampling_ratio),
            )
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
            # Retain source supervision within the fixed phase-4 budget.
            # The existing alternation gives one RGB and four IR steps at 0.2.
            return self._alternate(
                phase=self.Phase.PHASE4_IR_FOCUS,
                primary="p4_rgb_supervised",
                secondary="p4_ir_focus",
                ratio=self.config.curriculum.phase4_rgb_sampling_ratio,
            )

    @staticmethod
    def get_source_domain(step_name: str) -> str:
        """Resolve the existing physical source policy, including RGB-backed P2 IR."""
        # Supply data flow with the actual loader domain, independently of the
        # teacher name or SAGA appearance level carried by the route settings.
        if step_name in {
            "p1_rgb_supervised", "p2_rgb_flow", "p2_ir_flow", "p3_rgb_flow",
            "p4_rgb_supervised",
        }:
            return "rgb"
        if step_name in {"p3_ir_flow", "p4_ir_focus"}:
            return "ir"
        raise ValueError(f"Unsupported curriculum step: {step_name}")

    def should_evaluate_pseudo_labels(self, global_step, *, last_eval_step):
        """Check global windows; fixed IR evaluation is independent of train routes.

        The trainer separately requires initialized teachers. A delayed first
        pass does not shift later iteration boundaries or consume loader routes.
        """
        return should_evaluate_pseudo_labels(
            self.config.pseudo_label_eval, global_step, last_eval_step=last_eval_step,
        )

    def _alternate(self, phase, primary: DomainStep, secondary: DomainStep, ratio: float) -> DomainStep:
        """Helper to alternate between two steps based on a ratio."""
        if self.config.curriculum.sampling_mode == "interleaved":
            selected = self._samplers[phase].next_domain()
            return primary if selected == "primary" else secondary
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
        if self.config.curriculum.sampling_mode == "interleaved":
            return {
                "mode": "interleaved",
                "samplers": {phase.name: sampler.state_dict() for phase, sampler in self._samplers.items()},
            }
        return {phase.name: count for phase, count in self._counters.items()}

    def load_state_dict(self, state):
        if self.config.curriculum.sampling_mode == "interleaved":
            if state.get("mode") != "interleaved":
                raise ValueError("cannot restore grouped scheduler state into interleaved mode")
            saved = state.get("samplers")
            expected = {phase.name for phase in self._samplers}
            legacy = expected - {Phase.PHASE4_IR_FOCUS.name}
            if not isinstance(saved, dict) or set(saved) not in (expected, legacy):
                raise ValueError("invalid saved interleaved scheduler state")
            samplers = self._new_interleaved_samplers()
            for phase, sampler in samplers.items():
                # Pre-merge checkpoints tracked only phases 2/3; initialize
                # their new phase-4 allocation just like legacy grouped state.
                if phase.name in saved:
                    sampler.load_state_dict(saved[phase.name])
            self._samplers = samplers
            return
        if state.get("mode") == "interleaved":
            raise ValueError("cannot restore interleaved scheduler state into grouped mode")
        for phase in self._counters:
            if phase.name in state:
                self._counters[phase] = int(state[phase.name])

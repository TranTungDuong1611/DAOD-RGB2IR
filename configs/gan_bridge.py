"""Five-stage domain allocation and frozen GAN bridge settings."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
from typing import Optional, Tuple


@dataclass
class BridgeScheduleConfig:
    """Five stages mixing RGB, SAGA, GAN-IR and real IR, in that order."""

    phase_ratios: Tuple[Tuple[float, float, float, float], ...] = (
        (10, 0, 0, 0),
        (5, 3, 2, 0),
        (3, 3, 3, 1),
        (2, 2, 4, 2),
        (1, 1, 2, 6),
    )
    phase_fractions: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)

    def __post_init__(self):
        self.phase_ratios = tuple(tuple(row) for row in self.phase_ratios)
        self.phase_fractions = tuple(self.phase_fractions)
        if len(self.phase_ratios) != 5 or len(self.phase_fractions) != 5:
            raise ValueError("bridge schedule requires five ratios and five end fractions")
        for row in self.phase_ratios:
            if len(row) != 4 or any(
                not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 for value in row
            ) or not any(value > 0 for value in row):
                raise ValueError("each bridge ratio requires four finite nonnegative weights with a positive sum")
        if self.phase_ratios[0][0] <= 0 or any(self.phase_ratios[0][1:]):
            raise ValueError("the first bridge stage must be RGB-only warmup")
        previous = 0.0
        for fraction in self.phase_fractions:
            if (not isinstance(fraction, Real) or isinstance(fraction, bool)
                    or not math.isfinite(fraction) or not previous < fraction <= 1.0):
                raise ValueError("bridge end fractions must increase strictly in (0, 1]")
            previous = fraction
        if self.phase_fractions[-1] != 1.0:
            raise ValueError("the final bridge end fraction must be 1.0")

    def phase_ends(self, total_iters: int) -> Tuple[int, ...]:
        if type(total_iters) is not int or total_iters <= 0:
            raise ValueError("total_iters must be a positive integer")
        return tuple(math.ceil(Fraction(str(f)) * total_iters) for f in self.phase_fractions)

    def stage_for_step(self, global_step: int, total_iters: int) -> int:
        """Include the final step for end-of-run evaluation labels."""
        if type(global_step) is not int or not 0 <= global_step <= total_iters:
            raise ValueError("bridge global_step is outside the run")
        return min(4, bisect_right(self.phase_ends(total_iters), global_step))


@dataclass
class GANBridgeConfig:
    checkpoint: Optional[str] = None
    source_sup_weight: float = 1.0
    saga_student_alphas: Tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.25)
    saga_teacher_alphas: Tuple[float, ...] = (1.0, 0.75, 0.50, 0.50, 0.25)
    saga_teachers: Tuple[str, ...] = ("none", "rgb", "ir", "ir", "ir")

    def __post_init__(self):
        if (not isinstance(self.source_sup_weight, Real)
                or not math.isfinite(self.source_sup_weight) or self.source_sup_weight <= 0):
            raise ValueError("bridge source_sup_weight must be finite and positive")
        for name in ("saga_student_alphas", "saga_teacher_alphas"):
            values = tuple(getattr(self, name))
            if len(values) != 5 or any(
                not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v <= 1
                for v in values
            ):
                raise ValueError(f"{name} requires five finite values in [0, 1]")
            setattr(self, name, values)
        self.saga_teachers = tuple(self.saga_teachers)
        if (len(self.saga_teachers) != 5 or self.saga_teachers[0] != "none"
                or any(name not in {"rgb", "ir"} for name in self.saga_teachers[1:])):
            raise ValueError("saga_teachers requires 'none' for warmup and four RGB/IR teacher names")


__all__ = ['BridgeScheduleConfig', 'GANBridgeConfig']

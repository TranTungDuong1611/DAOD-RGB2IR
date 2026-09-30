"""
BlockDomainScheduler — decides which domain step to execute at each iteration.

Domains: rgb / mid / ir  (mid = GAN-translated bridge domain, ir = real thermal).
The mixing ratio (rgb : mid : ir) changes across a sequence of PHASES. Within
each phase, iterations are grouped into short CYCLES of length `cycle_iters`.
Each cycle is split into three CONTIGUOUS blocks trained in order
rgb -> mid -> ir, sized according to the phase's ratio (largest-remainder
allocation), then the cycle repeats. This keeps the unlabeled `ir` block from
ever running longer than one cycle, so training regularly "anchors" back on
supervised rgb (mirrors D3T's block-scheduling behavior).

Stateless: get_domain(it) is a pure function of `it` (no internal state).
"""

from typing import List, Literal, Tuple

DomainStep = Literal["rgb", "mid", "ir"]

DEFAULT_RATIO_SCHEDULE = "7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8"


def parse_ratio_schedule(s: str) -> List[Tuple[int, int, int]]:
    """Parse "a:b:c,a:b:c,..." into a list of (rgb, mid, ir) ratio tuples."""
    phases = []
    for chunk in s.split(","):
        a, b, c = (int(x) for x in chunk.strip().split(":"))
        phases.append((a, b, c))
    return phases


class BlockDomainScheduler:
    """
    Domain scheduler with short-cycle block scheduling per phase.

    Divides `total_iters` into `len(phases)` equal-length phases. Within each
    phase, iterations are grouped into cycles of length `cycle_iters` (the
    final cycle of a phase may be shorter). Each cycle is split into three
    contiguous blocks (rgb, mid, ir) per the phase's ratio, using
    largest-remainder allocation so ratio=0 domains always get exactly 0 iters.

    Call `get_domain(it)` (0-indexed global iteration) once per iteration.
    """

    DOMAINS: Tuple[DomainStep, ...] = ("rgb", "mid", "ir")

    def __init__(
        self,
        phases: List[Tuple[int, int, int]],
        total_iters: int,
        cycle_iters: int = 200,
    ) -> None:
        self.phases = phases
        self.total_iters = total_iters
        self.n_phases = len(phases)
        self.cycle_iters = max(1, cycle_iters)

        # Phase boundaries (equal split): phase p covers [bounds[p], bounds[p+1])
        self.phase_bounds = [
            p * total_iters // self.n_phases for p in range(self.n_phases + 1)
        ]

        # Block sizes for one FULL cycle of each phase (for logging/inspection).
        self.cycle_blocks = []
        for p in range(self.n_phases):
            length = self.phase_bounds[p + 1] - self.phase_bounds[p]
            full = min(self.cycle_iters, length) if length > 0 else 0
            self.cycle_blocks.append(self._blocks_for_phase(full, phases[p]))

    @staticmethod
    def _blocks_for_phase(length: int, ratio: Tuple[int, int, int]) -> List[int]:
        """Split `length` iterations into 3 contiguous blocks per `ratio`.

        Largest-remainder method: floor-divide, then distribute the remainder
        to the domains with ratio > 0 in order of largest fractional part.
        A domain with ratio == 0 always gets exactly 0 iterations.
        """
        total = sum(ratio)
        if length <= 0 or total == 0:
            return [max(0, length), 0, 0]
        raw = [length * r / total for r in ratio]
        counts = [int(x) for x in raw]
        rem = length - sum(counts)
        order = sorted(
            (i for i in range(3) if ratio[i] > 0),
            key=lambda i: raw[i] - counts[i],
            reverse=True,
        )
        for j in range(rem):
            counts[order[j % len(order)]] += 1
        return counts

    def phase_for_iter(self, it: int) -> int:
        """Return the phase INDEX (0-based) containing iteration `it`."""
        for p in range(self.n_phases):
            if it < self.phase_bounds[p + 1]:
                return p
        return self.n_phases - 1

    def ratio_for_iter(self, it: int) -> Tuple[int, int, int]:
        return self.phases[self.phase_for_iter(it)]

    def _cycle_len_at(self, p: int, offset: int) -> Tuple[int, int]:
        """Return (offset_within_cycle, cycle_length) for `offset` within phase p."""
        length = self.phase_bounds[p + 1] - self.phase_bounds[p]
        cycle_start = (offset // self.cycle_iters) * self.cycle_iters
        cycle_len = min(self.cycle_iters, length - cycle_start)   # last cycle may be short
        return offset - cycle_start, cycle_len

    def blocks_for_iter(self, it: int) -> List[int]:
        p = self.phase_for_iter(it)
        offset = it - self.phase_bounds[p]
        _, cycle_len = self._cycle_len_at(p, offset)
        return self._blocks_for_phase(cycle_len, self.phases[p])

    def get_domain(self, it: int) -> DomainStep:
        """Return the domain ("rgb" | "mid" | "ir") to train at iteration `it`."""
        p = self.phase_for_iter(it)
        offset = it - self.phase_bounds[p]
        off_in_cycle, cycle_len = self._cycle_len_at(p, offset)
        n_rgb, n_mid, _ = self._blocks_for_phase(cycle_len, self.phases[p])
        if off_in_cycle < n_rgb:
            return "rgb"
        if off_in_cycle < n_rgb + n_mid:
            return "mid"
        return "ir"

    # Alias kept for interface parity with the previous CurriculumScheduler.
    def get_next_step(self, it: int) -> DomainStep:
        return self.get_domain(it)

    def __repr__(self) -> str:
        return (
            f"BlockDomainScheduler(phases={self.phases}, "
            f"total_iters={self.total_iters}, cycle_iters={self.cycle_iters})"
        )

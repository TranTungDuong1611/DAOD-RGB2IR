"""Execute loss computation with weights and KD settings resolved by the caller."""

import torch

from models.d3t_adapter import CriterionResult


def component_weight(name: str, *, uhl: float, quality: float, deltas: float, logits: float):
    """Apply the original name matching order to explicitly supplied weights."""
    if "uhl" in name:
        return uhl
    if "quality" in name:
        return quality
    if "box" in name or "delta" in name:
        return deltas
    if "cls" in name or "logit" in name:
        return logits
    return 1.0


class LossFlow:
    """Accumulate one iteration's losses without selecting phases or teachers."""

    def __init__(self, *, uhl: float, quality: float, deltas: float, logits: float):
        # Each instance belongs to one iteration. Resolve configuration upstream
        # and keep its differentiable total separate from detached log values.
        self._weights = dict(uhl=uhl, quality=quality, deltas=deltas, logits=logits)
        self.total_loss = None
        self.logs = {}

    def _add(self, name, value, phase_weight, log_name):
        # Keep multiplication and accumulation order for numerical parity.
        weighted = value * phase_weight * component_weight(name, **self._weights)
        self.total_loss = (
            weighted if self.total_loss is None else self.total_loss + weighted
        )
        self.logs[log_name] = float(weighted.detach().item())

    def supervised(self, student, student_output, targets, *, phase_weight: float):
        # Reuse the existing student forward to compute GT losses; apply the
        # supplied phase/component weights through the shared accumulator.
        for name, value in student.supervised_from_output(student_output, targets).items():
            self._add(name, value, phase_weight, f"sup_{name}")

    def distill(
        self, student, student_output, teacher_output, settings,
        *, teacher_name: str, phase_weight: float,
    ):
        # The caller has already selected and forwarded this teacher. Delegate
        # prediction matching and loss formulas to the existing wrapper/criterion.
        result = student.distill_from_outputs(student_output, teacher_output, settings)
        # Normalize the supported result contracts before recording losses and
        # diagnostics, preserving compatibility with wrappers that return a dict.
        if isinstance(result, dict):
            result = CriterionResult(result, {})
        if not isinstance(result, CriterionResult):
            raise TypeError("wrapper distillation must return CriterionResult")
        # Teacher-specific names keep contributions distinct when aggregating
        # multiple distillation results into the same iteration total.
        for name, value in result.losses.items():
            self._add(name, value, phase_weight, f"{name}_{teacher_name}")
        # Diagnostics are observations only; they never contribute to backward.
        for name, value in result.metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.logs[f"{name}_{teacher_name}"] = float(value.detach().item())
        # The observer may inspect selection metadata from this same KD call.
        # Only the loss/metric fields above enter the existing training path.
        return result

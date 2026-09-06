"""Detector-independent contracts used by the D3T training path.

Detector adapters own transforms, target assignment, decoding, postprocessing,
and the rules used to establish student/teacher correspondence.  The criterion
only receives the normalized tensors declared in this module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

Targets = Sequence[Mapping[str, Any]]
Losses = Dict[str, Tensor]
SampleIds = Tuple[str, ...]


@dataclass(frozen=True)
class DistillationSettings:
    """Immutable settings for one teacher's position-wise KD pass."""

    top_ratio: float
    min_hm: float
    hm_alpha: float = 1.0
    hm_beta: float = 1.0
    uncertainty_alpha: float = 4.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.top_ratio) or not 0.0 < self.top_ratio <= 1.0:
            raise ValueError("top_ratio must be in (0, 1]")
        if not math.isfinite(self.min_hm) or not 0.0 <= self.min_hm <= 1.0:
            raise ValueError("min_hm must be in [0, 1]")
        if (
            not math.isfinite(self.hm_alpha)
            or not math.isfinite(self.hm_beta)
            or not math.isfinite(self.uncertainty_alpha)
            or self.hm_alpha < 0
            or self.hm_beta < 0
            or self.uncertainty_alpha <= 0
        ):
            raise ValueError(
                "HM exponents must be non-negative and uncertainty_alpha positive"
            )


@dataclass(frozen=True)
class CriterionResult:
    """Losses plus detector-neutral metrics from a criterion call."""

    losses: Losses
    metrics: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class Predictions:
    """Predictions for one image in decoded XYXY pixel coordinates.

    ``class_logits`` contains foreground-only sigmoid logits.  ``boxes`` are
    differentiable decoded boxes in transformed-image coordinates and
    ``quality_logits`` predicts IoU, never centerness.
    """

    class_logits: Tensor
    boxes: Tensor
    quality_logits: Tensor

    def __post_init__(self) -> None:
        if self.class_logits.ndim != 2 or self.class_logits.shape[1] == 0:
            raise ValueError("class_logits must have shape [N, C] with C > 0")
        n = self.class_logits.shape[0]
        if self.boxes.shape != (n, 4):
            raise ValueError("boxes must have shape [N, 4] matching class_logits")
        if self.quality_logits.shape != (n,):
            raise ValueError("quality_logits must have shape [N]")
        tensors = (self.class_logits, self.boxes, self.quality_logits)
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise ValueError("Predictions must be floating point tensors")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("Predictions must share a device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("Predictions must share a dtype")

    def detached(self) -> "Predictions":
        return Predictions(
            self.class_logits.detach(),
            self.boxes.detach(),
            self.quality_logits.detach(),
        )


@dataclass(frozen=True)
class AdapterOutput:
    predictions: Tuple[Predictions, ...]
    # Private adapter data: transformed targets, features, anchors/proposals,
    # original/transformed image sizes, and stable sample IDs.
    context: Any = None
    sample_ids: Optional[SampleIds] = None


@dataclass(frozen=True)
class SupervisedBatch:
    """Loss-ready predictions and per-image aligned targets."""

    predictions: Tuple[Predictions, ...]
    targets: Targets
    native_losses: Losses = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.targets):
            raise ValueError("Supervised targets must match the prediction batch")
        for prediction, target in zip(self.predictions, self.targets):
            expected_shapes = {
                "class_targets": prediction.class_logits.shape,
                "box_targets": prediction.boxes.shape,
                "quality_targets": prediction.quality_logits.shape,
                "foreground": prediction.quality_logits.shape,
            }
            for key, shape in expected_shapes.items():
                if key not in target or target[key].shape != shape:
                    raise ValueError(f"{key} must have shape {tuple(shape)}")
                if target[key].device != prediction.class_logits.device:
                    raise ValueError(f"{key} must share the prediction device")
                if target[key].requires_grad:
                    raise ValueError(f"{key} must be detached")
            if target["foreground"].dtype != torch.bool:
                raise ValueError("foreground must be a boolean mask")


@dataclass(frozen=True)
class DistillationPair:
    """Explicitly matched student and teacher predictions."""

    student: Tuple[Predictions, ...]
    teacher: Tuple[Predictions, ...]

    def __post_init__(self) -> None:
        if len(self.student) != len(self.teacher):
            raise ValueError("Distillation requires matched image batches")
        for student, teacher in zip(self.student, self.teacher):
            if student.class_logits.shape != teacher.class_logits.shape:
                raise ValueError("Distillation requires matched rows and classes")
            if student.boxes.shape != teacher.boxes.shape:
                raise ValueError("Distillation requires matched decoded boxes")
            if student.quality_logits.shape != teacher.quality_logits.shape:
                raise ValueError("Distillation requires matched quality rows")
            if student.class_logits.device != teacher.class_logits.device:
                raise ValueError("Distillation pairs must share a device")

    def detached_teacher(self) -> "DistillationPair":
        return DistillationPair(
            self.student,
            tuple(prediction.detached() for prediction in self.teacher),
        )


class DetectorAdapter(nn.Module, ABC):
    """Adapter seam shared by detector families."""

    @staticmethod
    def validate_sample_ids(
        sample_ids: Optional[Sequence[str]], batch_size: int
    ) -> Optional[SampleIds]:
        """Validate stable IDs when a caller supplies them.

        The optional form keeps the original small adapter contract usable for
        inference and legacy unit fixtures.  The FCOS adapter records IDs in
        its context and rejects an ambiguous pair before position-wise KD.
        """

        if sample_ids is None:
            return None
        ids = tuple(sample_ids)
        if len(ids) != batch_size:
            raise ValueError("sample_ids must contain one ID per image")
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("sample_ids must contain non-empty stable strings")
        return ids

    @abstractmethod
    def forward(
        self,
        images: Sequence[Tensor],
        targets: Optional[Targets] = None,
        sample_ids: Optional[SampleIds] = None,
    ) -> AdapterOutput:
        raise NotImplementedError

    @abstractmethod
    def prepare_supervised(
        self, output: AdapterOutput, targets: Targets
    ) -> SupervisedBatch:
        raise NotImplementedError

    @abstractmethod
    def prepare_distillation(
        self, student: AdapterOutput, teacher: AdapterOutput
    ) -> DistillationPair:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self, output: AdapterOutput
    ) -> Sequence[Mapping[str, Tensor]]:
        raise NotImplementedError


class D3TCriterion(nn.Module, ABC):
    """Shared loss formulas receive prepared data, never detector internals."""

    @abstractmethod
    def supervised(self, batch: SupervisedBatch) -> Losses:
        raise NotImplementedError

    @abstractmethod
    def distillation(
        self, pair: DistillationPair, settings: DistillationSettings
    ) -> CriterionResult:
        raise NotImplementedError

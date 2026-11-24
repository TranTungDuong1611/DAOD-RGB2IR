"""Detector-independent detection and distillation orchestration."""

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .detector_adapter import (
    AdapterOutput,
    CriterionResult,
    DetectorCriterion,
    DetectorAdapter,
    DistillationSettings,
    Losses,
    Targets,
)


_DEFAULT_DISTILLATION_SETTINGS = DistillationSettings(
    top_ratio=1.0,
    min_hm=0.0,
)


class DetectorWrapper(nn.Module):
    """Combine one detector adapter with one detector-neutral criterion."""

    def __init__(self, adapter: DetectorAdapter, criterion: DetectorCriterion):
        super().__init__()
        self.adapter = adapter
        self.criterion = criterion

    def select_training_resize_short_edge(self) -> Optional[int]:
        """Delegate detector-specific geometry selection to the adapter."""

        return self.adapter.select_training_resize_short_edge()

    @staticmethod
    def _normalize_images(images: Sequence[Tensor] | Tensor) -> Tuple[Tensor, ...]:
        if isinstance(images, Tensor):
            if images.ndim != 4:
                raise ValueError("batched images must have shape [B, C, H, W]")
            image_list = tuple(images.unbind(dim=0))
        else:
            image_list = tuple(images)
        if not image_list:
            raise ValueError("images must contain at least one image")
        for image in image_list:
            if not isinstance(image, Tensor) or image.ndim != 3:
                raise ValueError("each image must have shape [C, H, W]")
        return image_list

    @staticmethod
    def _normalize_sample_ids(sample_ids, batch_size):
        if sample_ids is None:
            return None
        ids = tuple(sample_ids)
        if len(ids) != batch_size or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("sample_ids must contain one non-empty string per image")
        return ids

    @staticmethod
    def _criterion_result(result) -> CriterionResult:
        if isinstance(result, CriterionResult):
            return result
        if isinstance(result, dict):
            return CriterionResult(losses=result, metrics={})
        raise TypeError(
            "criterion must return CriterionResult (or a loss dict for compatibility)"
        )

    @staticmethod
    def _merge_losses(*loss_maps: Losses) -> Losses:
        merged: Losses = {}
        for losses in loss_maps:
            duplicates = merged.keys() & losses.keys()
            if duplicates:
                raise ValueError(
                    f"Duplicate native/criterion loss names: {sorted(duplicates)}"
                )
            merged.update(losses)
        return merged

    def raw(
        self,
        images: Sequence[Tensor] | Tensor,
        targets: Optional[Targets] = None,
        sample_ids: Optional[Sequence[str]] = None,
        resize_short_edge: Optional[int] = None,
    ) -> AdapterOutput:
        image_list = self._normalize_images(images)
        if targets is not None and len(targets) != len(image_list):
            raise ValueError("targets must match the image batch length")

        resolved_ids = self._normalize_sample_ids(sample_ids, len(image_list))

        if resize_short_edge is not None and resize_short_edge <= 0:
            raise ValueError("resize_short_edge must be positive")
        with self.adapter.resize_short_edge(resize_short_edge):
            output = self.adapter(
                image_list,
                targets,
                sample_ids=resolved_ids,
            )
        if output.sample_ids is None and resolved_ids is not None:
            output = replace(output, sample_ids=resolved_ids)
        return output

    def supervised_from_output(
        self, output: AdapterOutput, targets: Targets
    ) -> Losses:
        if len(output.predictions) != len(targets):
            raise ValueError("targets must match the output batch length")
        batch = self.adapter.prepare_supervised(output, targets)
        result = self._criterion_result(self.criterion.supervised(batch))
        return self._merge_losses(result.losses, batch.native_losses)

    def distill_from_outputs(
        self,
        student_output: AdapterOutput,
        teacher_output: AdapterOutput,
        settings: Optional[DistillationSettings] = None,
    ) -> CriterionResult:
        pair = self.adapter.prepare_distillation(student_output, teacher_output)
        result = self._criterion_result(
            self.criterion.distillation(
                pair.detached_teacher(),
                settings or _DEFAULT_DISTILLATION_SETTINGS,
            )
        )
        return result

    def forward(
        self,
        images: Sequence[Tensor] | Tensor,
        targets: Optional[Targets] = None,
        sample_ids: Optional[Sequence[str]] = None,
    ):
        if self.training and targets is None:
            raise ValueError("Training requires targets; use raw() for unlabeled images")
        output = self.raw(images, targets, sample_ids)
        if not self.training:
            return self.adapter.postprocess(output)
        return self.supervised_from_output(output, targets)

    def distill(
        self,
        student_images: Sequence[Tensor] | Tensor,
        teacher: "DetectorWrapper",
        teacher_images: Sequence[Tensor] | Tensor,
        settings: Optional[DistillationSettings] = None,
        sample_ids: Optional[Sequence[str]] = None,
        teacher_sample_ids: Optional[Sequence[str]] = None,
    ) -> Losses:
        """Convenience one-teacher call; the trainer owns routing and EMA."""

        if teacher.training or any(
            module.training for module in teacher.adapter.modules()
        ):
            raise ValueError("Teacher and its adapter must be in eval mode")
        resize_short_edge = self.select_training_resize_short_edge()
        student_output = self.raw(
            student_images,
            sample_ids=sample_ids,
            resize_short_edge=resize_short_edge,
        )
        with torch.no_grad():
            teacher_output = teacher.raw(
                teacher_images,
                sample_ids=teacher_sample_ids,
                resize_short_edge=resize_short_edge,
            )
        return self.distill_from_outputs(
            student_output,
            teacher_output,
            settings or _DEFAULT_DISTILLATION_SETTINGS,
        ).losses

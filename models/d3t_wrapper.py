"""Detector-independent D3T orchestration."""

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .d3t_adapter import (
    AdapterOutput,
    CriterionResult,
    D3TCriterion,
    DetectorAdapter,
    DistillationSettings,
    Losses,
    Targets,
)


_DEFAULT_DISTILLATION_SETTINGS = DistillationSettings(
    top_ratio=1.0,
    min_hm=0.0,
)


class D3TWrapper(nn.Module):
    """Combine one detector adapter with one detector-neutral criterion."""

    def __init__(self, adapter: DetectorAdapter, criterion: D3TCriterion):
        super().__init__()
        self.adapter = adapter
        self.criterion = criterion

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
    def _ids_from_targets(
        targets: Optional[Targets], batch_size: int
    ) -> Optional[Tuple[str, ...]]:
        if targets is None or not all("stem" in target for target in targets):
            return None
        stems = tuple(target["stem"] for target in targets)
        if all(isinstance(stem, str) and stem.strip() for stem in stems):
            return stems
        return None

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
                    f"Duplicate native/D3T loss names: {sorted(duplicates)}"
                )
            merged.update(losses)
        return merged

    def raw(
        self,
        images: Sequence[Tensor] | Tensor,
        targets: Optional[Targets] = None,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> AdapterOutput:
        image_list = self._normalize_images(images)
        if targets is not None and len(targets) != len(image_list):
            raise ValueError("targets must match the image batch length")

        resolved_ids = sample_ids
        if resolved_ids is None:
            resolved_ids = self._ids_from_targets(targets, len(image_list))
        resolved_ids = self.adapter.validate_sample_ids(resolved_ids, len(image_list))

        output = self.adapter(
            image_list,
            targets,
            sample_ids=resolved_ids,
        )
        if len(output.predictions) != len(image_list):
            raise ValueError("Adapter output must match the image batch length")
        if output.sample_ids is None and resolved_ids is not None:
            output = replace(output, sample_ids=resolved_ids)
        elif output.sample_ids is not None:
            self.adapter.validate_sample_ids(output.sample_ids, len(image_list))
            if resolved_ids is not None and tuple(output.sample_ids) != tuple(resolved_ids):
                raise ValueError("adapter output sample IDs differ from the input batch")
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
        teacher: "D3TWrapper",
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
        student_list = self._normalize_images(student_images)
        teacher_list = self._normalize_images(teacher_images)
        if len(student_list) != len(teacher_list):
            raise ValueError("Distillation requires matching image batch lengths")

        student_output = self.raw(student_list, sample_ids=sample_ids)
        with torch.no_grad():
            teacher_output = teacher.raw(teacher_list, sample_ids=teacher_sample_ids)
        return self.distill_from_outputs(
            student_output,
            teacher_output,
            settings or _DEFAULT_DISTILLATION_SETTINGS,
        ).losses

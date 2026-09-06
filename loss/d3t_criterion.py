"""Detector-neutral supervised and distillation losses for D3T."""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn

from models.d3t_adapter import (
    CriterionResult,
    D3TCriterion,
    DistillationPair,
    DistillationSettings,
    Losses,
    SupervisedBatch,
)


def _hm_focal_elementwise(
    logits: Tensor,
    targets: Tensor,
    alpha: float,
    gamma: float,
    weight_type: str = "iou",
) -> Tensor:
    """Return unreduced HM/varifocal-style BCE terms."""

    if logits.shape != targets.shape:
        raise ValueError("HM-Focal logits and targets must have the same shape")
    probabilities = logits.sigmoid()
    targets = targets.to(dtype=logits.dtype)
    positive = targets > 0
    negative_weight = alpha * (probabilities - targets).abs().pow(gamma)
    if weight_type == "iou":
        positive_weight = targets
    elif weight_type == "hm":
        positive_weight = torch.sqrt(
            probabilities.detach().clamp_min(0) * targets.clamp_min(0)
        )
    elif weight_type == "hm_rev":
        positive_weight = 1.0 - torch.sqrt(
            probabilities.detach().clamp_min(0) * targets.clamp_min(0)
        )
    elif weight_type in {"binary", "ones"}:
        positive_weight = torch.ones_like(targets)
    else:
        raise ValueError(f"Unsupported HM-Focal weight_type: {weight_type}")
    focal_weight = torch.where(positive, positive_weight, negative_weight)
    return F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ) * focal_weight


def _iou_loss_xyxy(
    pred_boxes: Tensor, target_boxes: Tensor, eps: float = 1e-7
) -> Tensor:
    """Return unreduced aligned ``1 - IoU`` for decoded XYXY boxes."""

    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes and target_boxes must both have shape [N, 4]")
    top_left = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    bottom_right = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp_min(0)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp_min(0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = pred_area + target_area - intersection
    return 1.0 - intersection / union.clamp_min(eps)


def _giou_loss_xyxy(
    pred_boxes: Tensor, target_boxes: Tensor, eps: float = 1e-7
) -> Tensor:
    """Return unreduced aligned GIoU loss for decoded XYXY boxes."""

    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes and target_boxes must both have shape [N, 4]")
    top_left = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    bottom_right = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp_min(0)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp_min(0)
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    target_area = target_wh[..., 0] * target_wh[..., 1]
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp_min(eps)
    enclosing_top_left = torch.minimum(pred_boxes[..., :2], target_boxes[..., :2])
    enclosing_bottom_right = torch.maximum(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosing_wh = (enclosing_bottom_right - enclosing_top_left).clamp_min(0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(eps)
    return 1.0 - giou


def _zero_with_grad(*tensors: Tensor) -> Tensor:
    """Create a scalar zero connected to every supplied computation graph."""

    connected = None
    for tensor in tensors:
        term = tensor.sum() * 0.0
        connected = term if connected is None else connected + term
    if connected is None:
        return torch.tensor(0.0)
    return connected


def _cat_or_none(tensors: Sequence[Tensor]) -> Optional[Tensor]:
    if not tensors:
        return None
    return torch.cat(tuple(tensors), dim=0)


class D3TLossCriterion(D3TCriterion):
    """HM-Focal supervised loss and position-wise D3T distillation."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        weight_type: str = "iou",
        kd_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or gamma < 0 or kd_gamma < 0:
            raise ValueError("loss exponents must be non-negative")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.weight_type = weight_type
        self.kd_gamma = float(kd_gamma)

    @staticmethod
    def _supervised_tensors(batch: SupervisedBatch):
        predictions = batch.predictions
        targets = batch.targets
        if len(predictions) != len(targets):
            raise ValueError("supervised predictions and targets must be aligned")
        class_logits = _cat_or_none([p.class_logits for p in predictions])
        boxes = _cat_or_none([p.boxes for p in predictions])
        quality_logits = _cat_or_none([p.quality_logits for p in predictions])
        class_targets = _cat_or_none([t["class_targets"] for t in targets])
        box_targets = _cat_or_none([t["box_targets"] for t in targets])
        quality_targets = _cat_or_none([t["quality_targets"] for t in targets])
        foreground = _cat_or_none([t["foreground"] for t in targets])
        return (
            class_logits,
            boxes,
            quality_logits,
            class_targets,
            box_targets,
            quality_targets,
            foreground,
        )

    def supervised(self, batch: SupervisedBatch) -> Losses:
        (
            class_logits,
            boxes,
            quality_logits,
            class_targets,
            box_targets,
            quality_targets,
            foreground,
        ) = self._supervised_tensors(batch)
        if class_logits is None:
            zero = _zero_with_grad()
            return {"loss_cls": zero, "loss_box": zero, "loss_quality": zero}

        num_foreground = int(foreground.sum().item())
        denominator = max(1, num_foreground)
        cls_elementwise = _hm_focal_elementwise(
            class_logits,
            class_targets,
            alpha=self.alpha,
            gamma=self.gamma,
            weight_type=self.weight_type,
        )
        loss_cls = cls_elementwise.sum() / denominator
        if num_foreground:
            loss_box = _iou_loss_xyxy(
                boxes[foreground], box_targets[foreground]
            ).sum() / denominator
            loss_quality = F.binary_cross_entropy_with_logits(
                quality_logits[foreground],
                quality_targets[foreground].to(dtype=quality_logits.dtype),
                reduction="sum",
            ) / denominator
        else:
            loss_box = _zero_with_grad(boxes)
            loss_quality = _zero_with_grad(quality_logits)
        return {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_quality": loss_quality,
        }

    @staticmethod
    def _pair_tensors(pair: DistillationPair):
        student = pair.student
        teacher = pair.teacher
        if len(student) != len(teacher):
            raise ValueError("distillation pair image counts must match")
        s_cls = _cat_or_none([p.class_logits for p in student])
        t_cls = _cat_or_none([p.class_logits for p in teacher])
        s_boxes = _cat_or_none([p.boxes for p in student])
        t_boxes = _cat_or_none([p.boxes for p in teacher])
        s_quality = _cat_or_none([p.quality_logits for p in student])
        t_quality = _cat_or_none([p.quality_logits for p in teacher])
        return s_cls, t_cls, s_boxes, t_boxes, s_quality, t_quality

    @staticmethod
    def _metrics(
        device: torch.device,
        count: int,
        total: int,
        hm_sum: float,
    ):
        ratio = count / total if total else 0.0
        empty = float(count == 0)
        values = {
            "kd_selected_count": torch.tensor(float(count), device=device),
            "kd_selected_ratio": torch.tensor(ratio, device=device),
            "kd_hm_sum": torch.tensor(hm_sum, device=device),
            "kd_empty_mask": torch.tensor(empty, device=device),
        }
        # Short aliases make the metric contract convenient for standalone
        # callers while the prefixed names remain stable for trainer logging.
        values.update({
            "selected_count": values["kd_selected_count"],
            "selected_ratio": values["kd_selected_ratio"],
            "hm_sum": values["kd_hm_sum"],
            "empty_mask": values["kd_empty_mask"],
        })
        return values

    def distillation(
        self,
        pair: DistillationPair,
        settings: DistillationSettings,
    ):
        (
            student_cls,
            teacher_cls,
            student_boxes,
            teacher_boxes,
            student_quality,
            teacher_quality,
        ) = self._pair_tensors(pair)
        if student_cls is None:
            zero = _zero_with_grad()
            metrics = self._metrics(torch.device("cpu"), 0, 0, 0.0)
            return CriterionResult(
                losses={
                    "loss_kd_cls": zero,
                    "loss_kd_box": zero,
                    "loss_kd_quality": zero,
                },
                metrics=metrics,
            )

        total_rows = student_cls.shape[0]
        if total_rows != teacher_cls.shape[0]:
            raise ValueError("distillation rows must match")
        device = student_cls.device
        with torch.no_grad():
            teacher_prob = teacher_cls.detach().sigmoid()
            teacher_quality_prob = teacher_quality.detach().sigmoid()
            hm = teacher_prob.max(dim=1).values.pow(settings.hm_alpha)
            hm = hm * teacher_quality_prob.pow(settings.hm_beta)
            count = min(
                total_rows,
                max(1, int(math.ceil(total_rows * settings.top_ratio))),
            )
            top_indices = hm.topk(count, dim=0).indices
            top_mask = torch.zeros(total_rows, dtype=torch.bool, device=device)
            top_mask[top_indices] = True
            selected = top_mask & (hm >= settings.min_hm)
            selected_hm = hm[selected]
            hm_sum = float(selected_hm.sum().item())
            denominator = selected_hm.sum().clamp_min(1.0)
            reliability = torch.exp(
                -(1.0 - hm) / settings.uncertainty_alpha
            ).detach()

        if selected.any():
            student_prob = student_cls.sigmoid()
            cls_elementwise = F.binary_cross_entropy(
                student_prob[selected],
                teacher_prob[selected],
                reduction="none",
            ) * (
                student_prob[selected] - teacher_prob[selected]
            ).abs().pow(self.kd_gamma)
            cls_per_row = cls_elementwise.sum(dim=1)
            loss_kd_cls = (
                cls_per_row * reliability[selected]
            ).sum() / denominator

            box_per_row = _giou_loss_xyxy(
                student_boxes[selected], teacher_boxes[selected].detach()
            )
            loss_kd_box = (
                box_per_row * reliability[selected]
            ).sum() / denominator
            quality_per_row = F.binary_cross_entropy_with_logits(
                student_quality[selected],
                teacher_quality_prob[selected],
                reduction="none",
            )
            loss_kd_quality = (
                quality_per_row * reliability[selected]
            ).sum() / denominator
        else:
            loss_kd_cls = _zero_with_grad(student_cls)
            loss_kd_box = _zero_with_grad(student_boxes)
            loss_kd_quality = _zero_with_grad(student_quality)

        return CriterionResult(
            losses={
                "loss_kd_cls": loss_kd_cls,
                "loss_kd_box": loss_kd_box,
                "loss_kd_quality": loss_kd_quality,
            },
            metrics=self._metrics(device, int(selected.sum().item()), total_rows, hm_sum),
        )


__all__ = [
    "D3TLossCriterion",
    "_hm_focal_elementwise",
    "_iou_loss_xyxy",
    "_zero_with_grad",
]

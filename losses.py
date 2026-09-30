"""
Dense distillation losses (D3T-style) for domain-adaptive FCOS training.

Replaces the earlier hard pseudo-label pipeline (filter_pseudo_labels +
compute_rgb_mid_loss / compute_mid_ir_loss) with DENSE distillation: instead
of NMS-filtering teacher boxes into hard GT-like targets, we tap the FCOS
head's dense per-location outputs (cls_logits, bbox_regression, bbox_ctrness)
for both student (strong aug) and teacher (weak aug, no_grad) and compute a
soft multi-term loss directly on the dense grid:

  1. QFLv2   on cls_logits          (quality focal loss v2)
  2. GIoU    on decoded boxes       (bbox_regression deltas)
  3. BCE     on bbox_ctrness        (quality/centerness branch)
  4. VFL/UHL on top-ratio region    (uncertainty-harmonized loss)

Region selection uses the "harmony measure" hm = cls_prob^alpha * iou^beta
(iou = sigmoid(centerness), since torchvision FCOS has no separate IoU head)
and per-location uncertainty weighting exp(-(1-hm)/un_alpha).

FCOS ONLY: dense_forward requires access to the torchvision FCOS internals
(transform / backbone / head / anchor_generator / box_coder), which
FasterRCNNDetector does not expose in a compatible shape (RPN + RoI heads,
no dense per-location classification/centerness grid). Faster-RCNN training
must keep using standard supervised losses; do not call get_distill_loss /
dense_forward with a FasterRCNNDetector.

Expected detector API (FCOS, batched-tensor wrapper — see fcos_wrapper.py):
  Training:  model(images, targets) → Dict[str, Tensor]
  Inference: model(images)          → List[Dict]
  Dense:     model.dense_forward(images) → (cls_logits, bbox_reg, bbox_ctrness,
                                             boxes_xyxy, anchors)

All teacher forward passes must run under torch.no_grad() — callers must not
wrap this module in no_grad() since the student path needs gradients.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss

from config import DistillConfig

DenseOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def dense_forward(model, images) -> DenseOutput:
    """
    Dense (no-NMS) FCOS forward pass.

    Args:
        model  : an FCOSDetector (see fcos_wrapper.py) — must expose
                 `.dense_forward(images)` which taps the wrapped torchvision
                 FCOS's transform/backbone/head/anchor_generator/box_coder.
        images : Tensor[B, C, H, W] or List[Tensor[C, H, W]] (whatever the
                 wrapper's dense_forward accepts).

    Returns:
        (cls_logits, bbox_regression, bbox_ctrness, boxes_xyxy, anchors)
        each shaped (N, HWA, ...) — dense per-location, no NMS.
    """
    return model.dense_forward(images)


def QFLv2(
    pred_sigmoid: torch.Tensor,
    teacher_sigmoid: torch.Tensor,
    weight: torch.Tensor = None,
    beta: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """Quality Focal Loss v2. `weight` is used only as a pos/valid mask (>0 / >=0)."""
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(pred_sigmoid, zerolabel, reduction="none") * pt.pow(beta)
    pos = weight > 0
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction="none"
    ) * pt.pow(beta)
    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def vfl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Varifocal Loss (weight_type='iou'), reduction=mean."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    focal_weight = target * (target > 0.0).float() + \
        alpha * (pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    return loss.mean()


def get_distill_loss(
    student_out: DenseOutput,
    teacher_out: DenseOutput,
    num_classes: int,
    ratio: float,
    hm_alpha: float,
    hm_beta: float,
    un_alpha: float,
    w_logits: float,
    w_deltas: float,
    w_quality: float,
    w_uhl: float,
) -> torch.Tensor:
    """Sum of the 4 weighted D3T distillation loss terms. Returns a scalar."""
    s_logits, s_deltas, s_ctr, s_boxes, _ = student_out
    t_logits, t_deltas, t_ctr, t_boxes, _ = teacher_out
    N = s_logits.shape[0]

    s_logits_f = s_logits.reshape(-1, num_classes)
    t_logits_f = t_logits.reshape(-1, num_classes)
    s_ctr_f = s_ctr.reshape(-1, 1)
    t_ctr_f = t_ctr.reshape(-1, 1)
    s_boxes_f = s_boxes.reshape(-1, 4)
    t_boxes_f = t_boxes.reshape(-1, 4)

    # ── Region selection by harmony measure (teacher only, no grad) ──────────
    with torch.no_grad():
        count_num = max(1, int(t_logits_f.size(0) * ratio))
        t_probs = t_logits_f.sigmoid()
        cls_prob = t_probs.max(1)[0]
        iou = t_ctr_f.sigmoid().squeeze(1)
        hm = cls_prob.pow(hm_alpha) * iou.pow(hm_beta)
        sorted_vals, sorted_inds = torch.topk(hm, hm.numel())
        mask = torch.zeros_like(hm)
        mask[sorted_inds[:count_num]] = 1.0
        fg_num = sorted_vals[:count_num].sum().clamp(min=1.0)
        b_mask = mask > 0.0

    # ── Uncertainty weighting (detached) ──────────────────────────────────────
    cls_prob = t_probs.max(1)[0]
    iou = t_ctr_f.sigmoid().squeeze(1)
    hm = cls_prob.pow(hm_alpha) * iou.pow(hm_beta)
    loss_weight = torch.exp(-((1.0 - hm) / un_alpha)).detach()

    # ── 1) logits: QFLv2 (dense, normalized by fg_num) ────────────────────────
    loss_logits = QFLv2(
        s_logits_f.sigmoid(), t_probs, weight=loss_weight, reduction="sum"
    ) / fg_num

    # ── 2) deltas: GIoU between student/teacher decoded boxes (dense, weighted) ─
    giou = generalized_box_iou_loss(s_boxes_f, t_boxes_f, reduction="none", eps=1e-7)
    loss_deltas = (giou * loss_weight).mean()

    # ── 3) quality: BCE centerness student vs teacher (dense, weighted) ──────
    loss_quality = F.binary_cross_entropy(
        s_ctr_f.sigmoid(), t_ctr_f.sigmoid(),
        weight=loss_weight.unsqueeze(1), reduction="mean",
    )

    # ── 4) UHL: VFL with target = IoU of neighboring boxes, on top-ratio region ─
    n_b_mask = b_mask.view(N, -1)
    n_s_logits = s_logits
    t_probs_n = t_probs.view(N, -1, num_classes)
    logits_list, targets_list = [], []
    for idx in range(N):
        m = n_b_mask[idx]
        if m.sum() <= 1:
            continue
        pos_boxes = s_boxes[idx][m].detach()
        ious = box_iou(pos_boxes, pos_boxes)
        ious.fill_diagonal_(0)
        ious_max, _ = ious.max(dim=1)
        loss_mask = ious_max > 0
        cls_iou_targets = torch.zeros_like(n_s_logits[idx][m])
        pseudo_labels = t_probs_n[idx].max(1)[1][m]
        cls_iou_targets[loss_mask, pseudo_labels[loss_mask]] = ious_max[loss_mask]
        logits_list.append(n_s_logits[idx][m][loss_mask])
        targets_list.append(cls_iou_targets[loss_mask])

    if logits_list and sum(t.shape[0] for t in logits_list) > 0:
        loss_uhl = vfl_loss(torch.cat(logits_list), torch.cat(targets_list))
    else:
        loss_uhl = s_logits.sum() * 0.0

    return (
        w_logits * loss_logits
        + w_deltas * loss_deltas
        + w_quality * loss_quality
        + w_uhl * loss_uhl
    )


def distill(student_out: DenseOutput, teacher_out: DenseOutput,
            num_classes: int, cfg: DistillConfig) -> torch.Tensor:
    """Convenience wrapper: calls get_distill_loss with a DistillConfig."""
    return get_distill_loss(
        student_out, teacher_out, num_classes=num_classes,
        ratio=cfg.ratio, hm_alpha=cfg.hm_alpha, hm_beta=cfg.hm_beta,
        un_alpha=cfg.un_alpha, w_logits=cfg.w_logits, w_deltas=cfg.w_deltas,
        w_quality=cfg.w_quality, w_uhl=cfg.w_uhl,
    )


def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum all scalar tensors in a detector loss dict (used for the supervised GT loss)."""
    return sum(v for v in loss_dict.values())

"""
Loss functions for each domain step.

Expected detector API (Faster-RCNN / FCOS / DINO style):
  Training:  model(images, targets) → Dict[str, Tensor]  (named loss components)
  Inference: model(images)          → List[Dict]          (boxes, labels, scores)

All teacher forward passes run under torch.no_grad() — callers must not wrap
this module in no_grad() since the student path needs gradients.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import LossConfig


# ---------------------------------------------------------------------------
# Pseudo-label filtering
# ---------------------------------------------------------------------------

def filter_pseudo_labels(
    predictions: List[Dict[str, torch.Tensor]],
    conf_thresh: float = 0.7,
) -> List[Dict[str, torch.Tensor]]:
    """
    Filter teacher predictions by confidence score.

    Args:
        predictions : output of model(images) in inference mode
                      each dict has "boxes" [N,4], "labels" [N], "scores" [N]
        conf_thresh : minimum score to keep a box

    Returns:
        filtered list of dicts (same length as predictions, empty dicts possible)
    """
    pseudo = []
    for pred in predictions:
        scores = pred.get("scores", torch.zeros(0))
        if scores.numel() == 0:
            pseudo.append({
                "boxes":  torch.zeros(0, 4, device=scores.device),
                "labels": torch.zeros(0, dtype=torch.long, device=scores.device),
                "scores": scores,
            })
            continue

        keep = scores >= conf_thresh
        pseudo.append({
            "boxes":  pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
        })
    return pseudo


def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum all scalar tensors in a detector loss dict."""
    return sum(v for v in loss_dict.values())


# ---------------------------------------------------------------------------
# RGB step loss
# ---------------------------------------------------------------------------

def compute_rgb_loss(
    student: nn.Module,
    images: torch.Tensor,
    gt_targets: List[Dict[str, torch.Tensor]],
    rgb_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: float = 0.7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    RGB supervised step:
      - mandatory:  supervised GT loss
      - optional:   pseudo-label loss from rgb_teacher (rgb_pseudo_weight > 0)

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    # --- Supervised GT loss (always active) ---
    gt_loss_dict = student(images, gt_targets)
    gt_loss = _sum_loss_dict(gt_loss_dict) * config.rgb_gt_weight
    components.append(gt_loss)
    log["rgb_gt_loss"] = gt_loss.item()

    # --- Optional pseudo-label loss from rgb_teacher ---
    if rgb_teacher is not None and config.rgb_pseudo_weight > 0.0:
        with torch.no_grad():
            pseudo_preds = rgb_teacher(images)
        pseudo_targets = filter_pseudo_labels(pseudo_preds, conf_thresh)

        pseudo_loss_dict = student(images, pseudo_targets)
        pseudo_loss = _sum_loss_dict(pseudo_loss_dict) * config.rgb_pseudo_weight
        components.append(pseudo_loss)
        log["rgb_pseudo_loss"] = pseudo_loss.item()

    total_loss = sum(components)
    log["rgb_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# MID step loss
# ---------------------------------------------------------------------------

def compute_mid_loss(
    student: nn.Module,
    mid_images: torch.Tensor,                              # student sees this (strong aug)
    rgb_teacher: nn.Module,
    ir_teacher: nn.Module,
    gt_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: float = 0.7,
    teacher_source: str = "both",                          # "rgb" | "ir" | "both"
    rgb_weight_override: Optional[float] = None,
    ir_weight_override: Optional[float] = None,
    teacher_images: Optional[torch.Tensor] = None,         # teacher sees this (weak aug)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MID (intermediate domain) step:
      - student receives SAGA-transformed images
      - learns from rgb_teacher pseudo-labels (mid_rgb_weight)
      - learns from ir_teacher  pseudo-labels (mid_ir_weight)
      - optional: GT loss if gt_targets provided  (mid_gt_weight)

    Both teacher forward passes run under no_grad.
    Student forward passes require grad.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    rgb_w = rgb_weight_override if rgb_weight_override is not None else config.mid_rgb_weight
    ir_w  = ir_weight_override  if ir_weight_override  is not None else config.mid_ir_weight

    # Teacher infers on weak images; student trains on strong images
    t_images = teacher_images if teacher_images is not None else mid_images

    # --- rgb_teacher pseudo-labels (weak) → student loss (strong) ---
    if teacher_source in ("rgb", "both") and rgb_w > 0.0:
        with torch.no_grad():
            rgb_preds = rgb_teacher(t_images)
        rgb_pseudo = filter_pseudo_labels(rgb_preds, conf_thresh)

        loss_dict = student(mid_images, rgb_pseudo)
        loss = _sum_loss_dict(loss_dict) * rgb_w
        components.append(loss)
        log["mid_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo-labels (weak) → student loss (strong) ---
    if teacher_source in ("ir", "both") and ir_w > 0.0:
        with torch.no_grad():
            ir_preds = ir_teacher(t_images)
        ir_pseudo = filter_pseudo_labels(ir_preds, conf_thresh)

        loss_dict = student(mid_images, ir_pseudo)
        loss = _sum_loss_dict(loss_dict) * ir_w
        components.append(loss)
        log["mid_ir_teacher_loss"] = loss.item()

    # --- Optional GT loss (uses original RGB GT mapped to MID space) ---
    if gt_targets is not None and config.mid_gt_weight > 0.0:
        loss_dict = student(mid_images, gt_targets)
        loss = _sum_loss_dict(loss_dict) * config.mid_gt_weight
        components.append(loss)
        log["mid_gt_loss"] = loss.item()

    if not components:
        # Safety: all weights are zero — return a zero-grad loss
        total_loss = mid_images.sum() * 0.0
    else:
        total_loss = sum(components)

    log["mid_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# IR step loss
# ---------------------------------------------------------------------------

def compute_ir_loss(
    student: nn.Module,
    ir_images: torch.Tensor,
    rgb_teacher: nn.Module,
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: float = 0.7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    IR (target domain) unsupervised step:
      - student learns from rgb_teacher pseudo-labels on IR images
      - student learns from ir_teacher  pseudo-labels on IR images

    Both teacher forward passes run under no_grad.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    # --- rgb_teacher pseudo-labels on IR ---
    if config.ir_rgb_teacher_weight > 0.0:
        with torch.no_grad():
            rgb_preds = rgb_teacher(ir_images)
        rgb_pseudo = filter_pseudo_labels(rgb_preds, conf_thresh)

        loss_dict = student(ir_images, rgb_pseudo)
        loss = _sum_loss_dict(loss_dict) * config.ir_rgb_teacher_weight
        components.append(loss)
        log["ir_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo-labels on IR ---
    if config.ir_ir_teacher_weight > 0.0:
        with torch.no_grad():
            ir_preds = ir_teacher(ir_images)
        ir_pseudo = filter_pseudo_labels(ir_preds, conf_thresh)

        loss_dict = student(ir_images, ir_pseudo)
        loss = _sum_loss_dict(loss_dict) * config.ir_ir_teacher_weight
        components.append(loss)
        log["ir_ir_teacher_loss"] = loss.item()

    if not components:
        total_loss = ir_images.sum() * 0.0
    else:
        total_loss = sum(components)

    log["ir_total_loss"] = total_loss.item()
    return total_loss, log

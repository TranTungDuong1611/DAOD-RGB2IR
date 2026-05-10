"""
Loss functions for each domain step.

Expected detector API (Faster-RCNN / FCOS / DINO style):
  Training:  model(images, targets) → Dict[str, Tensor]  (named loss components)
  Inference: model(images)          → List[Dict]          (boxes, labels, scores)

All teacher forward passes run under torch.no_grad() — callers must not wrap
this module in no_grad() since the student path needs gradients.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import LossConfig

# Global or per-class confidence threshold
ThreshType = Union[float, Dict[int, float]]

_DEFAULT_THRESH = 0.7   # fallback for classes absent from per-class dict


# ---------------------------------------------------------------------------
# Pseudo-label filtering
# ---------------------------------------------------------------------------

def filter_pseudo_labels(
    predictions: List[Dict[str, torch.Tensor]],
    conf_thresh: ThreshType = 0.7,
) -> List[Dict[str, torch.Tensor]]:
    """
    Filter teacher predictions by confidence score.

    Args:
        predictions : output of model(images) in inference mode
                      each dict has "boxes" [N,4], "labels" [N], "scores" [N]
        conf_thresh : float  — global threshold applied to all classes
                      Dict[int, float] — per-class threshold; classes absent
                      from the dict fall back to _DEFAULT_THRESH (0.7)

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

        if isinstance(conf_thresh, dict):
            labels = pred["labels"]
            keep = torch.tensor(
                [scores[i].item() >= conf_thresh.get(int(labels[i].item()), _DEFAULT_THRESH)
                 for i in range(len(scores))],
                dtype=torch.bool,
                device=scores.device,
            )
        else:
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
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    RGB supervised step:
      - mandatory:  supervised GT loss (student sees strong aug)
      - optional:   pseudo-label loss from rgb_teacher (rgb_pseudo_weight > 0)

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on images (strong aug).
                    If None, both use images.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else images

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
            pseudo_preds = rgb_teacher(t_images)
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
    conf_thresh: ThreshType = 0.7,
    teacher_source: str = "both",                          # "rgb" | "ir" | "both"
    rgb_weight_override: Optional[float] = None,
    ir_weight_override: Optional[float] = None,
    teacher_images: Optional[torch.Tensor] = None,         # teacher sees this (weak aug)
    use_kl: bool = False,
    kl_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MID (intermediate domain) step.

    When use_kl=False (default):
      Hard pseudo-label mode — teacher predictions are filtered by confidence
      threshold and used as hard targets for the student detector loss.

    When use_kl=True:
      KL distillation mode — teacher cls head logits are captured via a forward
      hook and used as soft targets for the student cls head. No box filtering.
      GT loss (mid_gt_weight) is unchanged in both modes.

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

    t_images = teacher_images if teacher_images is not None else mid_images

    if use_kl:
        # ----------------------------------------------------------------
        # KL distillation path
        # ----------------------------------------------------------------
        from distillation import get_cls_hook, cls_kl_loss

        # --- Student cls logits ---
        # If GT targets present: capture logits during the train-mode GT forward
        # (saves one extra student forward pass).
        # If no GT: eval-mode forward with enable_grad to get cls logits.
        s_hook = get_cls_hook(student)

        if gt_targets is not None and config.mid_gt_weight > 0.0:
            gt_loss_dict = student(mid_images, gt_targets)   # train mode, hook fires
            gt_loss = _sum_loss_dict(gt_loss_dict) * config.mid_gt_weight
            components.append(gt_loss)
            log["mid_gt_loss"] = gt_loss.item()
        else:
            # No GT — separate eval-mode forward to get student cls logits
            student.eval()
            with torch.enable_grad():
                student(mid_images)
            student.train()

        s_logits = s_hook.output if s_hook is not None else None
        if s_hook is not None:
            s_hook.remove()

        # --- KL terms for each teacher source ---
        if s_logits is not None:
            if teacher_source in ("rgb", "both") and rgb_w > 0.0:
                t_hook = get_cls_hook(rgb_teacher)
                with torch.no_grad():
                    rgb_teacher(t_images)
                if t_hook is not None and t_hook.output is not None:
                    kl = cls_kl_loss(s_logits, t_hook.output, kl_temperature) * rgb_w
                    components.append(kl)
                    log["mid_rgb_kl_loss"] = kl.item()
                if t_hook is not None:
                    t_hook.remove()

            if teacher_source in ("ir", "both") and ir_w > 0.0:
                t_hook = get_cls_hook(ir_teacher)
                with torch.no_grad():
                    ir_teacher(t_images)
                if t_hook is not None and t_hook.output is not None:
                    kl = cls_kl_loss(s_logits, t_hook.output, kl_temperature) * ir_w
                    components.append(kl)
                    log["mid_ir_kl_loss"] = kl.item()
                if t_hook is not None:
                    t_hook.remove()

    else:
        # ----------------------------------------------------------------
        # Hard pseudo-label path (original behaviour)
        # ----------------------------------------------------------------

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

        # --- Optional GT loss ---
        if gt_targets is not None and config.mid_gt_weight > 0.0:
            loss_dict = student(mid_images, gt_targets)
            loss = _sum_loss_dict(loss_dict) * config.mid_gt_weight
            components.append(loss)
            log["mid_gt_loss"] = loss.item()

    if not components:
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
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
    use_kl: bool = False,
    kl_temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    IR (target domain) unsupervised step — ir_teacher supervision only.

    When use_kl=False (default):
      Hard pseudo-label mode.

    When use_kl=True:
      KL distillation mode — no box generation or confidence filtering.
      Student runs in eval mode (with enable_grad) to capture cls logits.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else ir_images
    log: Dict[str, float] = {}

    if use_kl:
        from distillation import get_cls_hook, cls_kl_loss

        # Teacher cls logits (no grad)
        t_hook = get_cls_hook(ir_teacher)
        with torch.no_grad():
            ir_teacher(t_images)
        t_logits = t_hook.output if t_hook is not None else None
        if t_hook is not None:
            t_hook.remove()

        # Student cls logits — eval-mode forward WITH gradient
        # (IR step has no GT targets, so train-mode requires targets → use eval)
        s_hook = get_cls_hook(student)
        if s_hook is not None:
            student.eval()
            with torch.enable_grad():
                student(ir_images)
            student.train()
            s_logits = s_hook.output
            s_hook.remove()
        else:
            s_logits = None

        if s_logits is not None and t_logits is not None:
            total_loss = cls_kl_loss(s_logits, t_logits, kl_temperature) * config.ir_ir_teacher_weight
            log["ir_kl_loss"] = total_loss.item()
        else:
            total_loss = ir_images.sum() * 0.0
            log["ir_kl_loss"] = 0.0

        log["ir_total_loss"] = total_loss.item()
        return total_loss, log

    # Hard pseudo-label path (original behaviour)
    with torch.no_grad():
        ir_preds = ir_teacher(t_images)
    ir_pseudo = filter_pseudo_labels(ir_preds, conf_thresh)

    loss_dict  = student(ir_images, ir_pseudo)
    total_loss = _sum_loss_dict(loss_dict) * config.ir_ir_teacher_weight

    log["ir_ir_teacher_loss"] = total_loss.item()
    log["ir_total_loss"]      = total_loss.item()
    return total_loss, log

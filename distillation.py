"""
Soft KL distillation loss between student and teacher classification heads.

Replaces hard pseudo-label loss with feature-level knowledge distillation:
  - Teacher produces soft probability maps from raw cls logits
  - Student is trained to match these distributions
  - Optional conf_thresh mask: only locations where teacher is confident
    (max class prob >= threshold) are included in the loss

FCOS:         sigmoid binary-CE per (location, class) across FPN levels
Faster-RCNN:  KL divergence on per-ROI softmax distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

ThreshType = Union[float, Dict[int, float]]


# ---------------------------------------------------------------------------
# Hook utility
# ---------------------------------------------------------------------------

class _ClsHook:
    """Captures the raw output of the classification head via a forward hook."""

    def __init__(self, module: nn.Module) -> None:
        self.output = None
        self._handle = module.register_forward_hook(self._fn)

    def _fn(self, module, inp, out):
        # FCOS:         out = List[Tensor[B, C, H_l, W_l]] per FPN level
        # Faster-RCNN:  out = Tensor[N, num_classes + 1]
        self.output = out

    def remove(self) -> None:
        self._handle.remove()


def get_cls_hook(model: nn.Module) -> Optional[_ClsHook]:
    """
    Attach a hook to the classification head of a supported detector.

    FCOS (torchvision):    hooks model.head.classification_head
    Faster-RCNN:           hooks roi_heads.box_predictor.cls_score

    Returns None if the architecture is not recognised.
    """
    if hasattr(model, 'head') and hasattr(model.head, 'classification_head'):
        return _ClsHook(model.head.classification_head)
    if hasattr(model, 'roi_heads'):
        pred = getattr(model.roi_heads, 'box_predictor', None)
        if pred is not None and hasattr(pred, 'cls_score'):
            return _ClsHook(pred.cls_score)
    return None


# ---------------------------------------------------------------------------
# Confidence mask
# ---------------------------------------------------------------------------

def _build_conf_mask(
    teacher_probs: torch.Tensor,    # [B, C, H, W] — already sigmoid-ed
    conf_thresh: ThreshType,
) -> torch.Tensor:
    """
    Build a [B, H, W] boolean mask.
    True = at least one class at this location exceeds its confidence threshold.

    float threshold : keep if max_class_prob >= conf_thresh
    dict threshold  : keep if any class c has prob[:, c] >= conf_thresh[c]
    """
    if isinstance(conf_thresh, dict):
        B, C, H, W = teacher_probs.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=teacher_probs.device)
        for cls_id, thresh in conf_thresh.items():
            if cls_id < C:
                mask = mask | (teacher_probs[:, cls_id] >= thresh)
    else:
        mask = teacher_probs.max(dim=1).values >= conf_thresh
    return mask  # [B, H, W]


# ---------------------------------------------------------------------------
# KL / BCE distillation loss
# ---------------------------------------------------------------------------

def cls_kl_loss(
    student_logits: Union[List[torch.Tensor], torch.Tensor],
    teacher_logits: Union[List[torch.Tensor], torch.Tensor],
    temperature: float = 1.0,
    conf_thresh: Optional[ThreshType] = None,
) -> torch.Tensor:
    """
    Soft distillation loss between student and teacher cls head logits.

    FCOS — binary sigmoid BCE (multi-label per location):
        t_prob = sigmoid(t_logits / T)
        loss   = mean_over_levels [ BCE(s_logits / T, t_prob)[mask] ] * T²

    Faster-RCNN — multinomial KL per ROI:
        t_prob = softmax(t_logits / T)
        loss   = KL( log_softmax(s_logits / T) ‖ t_prob )[mask] * T²

    Args:
        student_logits : raw cls logits from student (requires_grad on leaf)
        teacher_logits : raw cls logits from teacher (detached internally)
        temperature    : > 1 softens teacher distribution (range 1.0–4.0)
        conf_thresh    : float or Dict[int, float] — spatial locations where
                         teacher max-class prob < threshold are excluded from
                         the loss. None = keep all locations (dense distillation).

    Returns:
        scalar Tensor with gradient attached to the student path.
        Returns zero-grad tensor if conf_thresh filters out all locations.
    """
    if isinstance(student_logits, (list, tuple)):
        # FCOS: per-level binary classification (sigmoid)
        total = student_logits[0].new_zeros(())
        n_valid_levels = 0

        for s_l, t_l in zip(student_logits, teacher_logits):
            t_prob = torch.sigmoid(t_l.detach() / temperature)  # [B, C, H, W]

            if conf_thresh is not None:
                mask = _build_conf_mask(t_prob, conf_thresh)         # [B, H, W]
                if not mask.any():
                    continue
                mask_exp = mask.unsqueeze(1).expand_as(s_l)          # [B, C, H, W]
                level_loss = F.binary_cross_entropy_with_logits(
                    (s_l / temperature)[mask_exp],
                    t_prob[mask_exp],
                    reduction='mean',
                ) * (temperature ** 2)
            else:
                level_loss = F.binary_cross_entropy_with_logits(
                    s_l / temperature,
                    t_prob,
                    reduction='mean',
                ) * (temperature ** 2)

            total = total + level_loss
            n_valid_levels += 1

        if n_valid_levels == 0:
            return student_logits[0].sum() * 0.0
        return total / n_valid_levels

    else:
        # Faster-RCNN: softmax KL per ROI
        t_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)  # [N, C+1]

        if conf_thresh is not None:
            # Foreground classes only (skip background index 0)
            fg_probs = t_prob[:, 1:]    # [N, C]
            if isinstance(conf_thresh, dict):
                mask = torch.zeros(t_prob.shape[0], dtype=torch.bool,
                                   device=t_prob.device)
                for cls_id, thresh in conf_thresh.items():
                    if cls_id < fg_probs.shape[1]:
                        mask = mask | (fg_probs[:, cls_id] >= thresh)
            else:
                mask = fg_probs.max(dim=1).values >= conf_thresh

            if not mask.any():
                return student_logits.sum() * 0.0

            student_logits = student_logits[mask]
            t_prob = t_prob[mask]

        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            t_prob,
            reduction='batchmean',
        ) * (temperature ** 2)

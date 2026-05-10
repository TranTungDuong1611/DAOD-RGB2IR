"""
Soft KL distillation loss between student and teacher classification heads.

Replaces hard pseudo-label loss with feature-level knowledge distillation:
  - Teacher produces soft probability maps from raw cls logits
  - Student is trained to match these distributions (no box filtering needed)
  - Teacher low-confidence outputs contribute proportionally less → fewer FP
  - Dense spatial supervision across all FPN locations

FCOS:         sigmoid binary-CE per (location, class) across FPN levels
Faster-RCNN:  KL divergence on per-ROI softmax distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


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
# KL / BCE distillation loss
# ---------------------------------------------------------------------------

def cls_kl_loss(
    student_logits: Union[List[torch.Tensor], torch.Tensor],
    teacher_logits: Union[List[torch.Tensor], torch.Tensor],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Soft distillation loss between student and teacher cls head logits.

    FCOS — binary sigmoid BCE (multi-label per location):
        t_prob = sigmoid(t_logits / T)
        loss   = mean_over_levels [ BCE(s_logits / T, t_prob) ] * T²

    Faster-RCNN — multinomial KL per ROI:
        t_prob = softmax(t_logits / T)
        loss   = KL( log_softmax(s_logits / T) ‖ t_prob ) * T²

    Args:
        student_logits : raw cls logits from student (requires_grad on leaf)
        teacher_logits : raw cls logits from teacher (detached internally)
        temperature    : > 1 softens the teacher distribution — locations with
                         moderate teacher confidence contribute less, reducing
                         gradient from potential FP predictions.

    Returns:
        scalar Tensor with gradient attached to the student path
    """
    if isinstance(student_logits, (list, tuple)):
        # FCOS: per-level binary classification (sigmoid)
        total = student_logits[0].new_zeros(())
        for s_l, t_l in zip(student_logits, teacher_logits):
            t_prob = torch.sigmoid(t_l.detach() / temperature)
            total = total + F.binary_cross_entropy_with_logits(
                s_l / temperature,
                t_prob,
                reduction='mean',
            ) * (temperature ** 2)
        return total / len(student_logits)
    else:
        # Faster-RCNN: softmax KL
        t_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            t_prob,
            reduction='batchmean',
        ) * (temperature ** 2)

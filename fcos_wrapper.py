"""
FCOSDetector — torchvision FCOS wrapped to match the trainer's detector API.

Trainer expects:
  train mode : model(images: Tensor[B,C,H,W], targets: List[Dict]) → Dict[str, Tensor]
  eval  mode : model(images: Tensor[B,C,H,W])                       → List[Dict]

torchvision FCOS expects:
  train mode : model(images: List[Tensor], targets: List[Dict])     → Dict[str, Tensor]
  eval  mode : model(images: List[Tensor])                           → List[Dict]

This wrapper bridges the batched-tensor API and the list-of-tensors API.

Target dict format (same as torchvision FCOS):
  {
    "boxes":  FloatTensor[N, 4]   # xyxy absolute pixel coords
    "labels": LongTensor[N]       # 0-indexed foreground class indices: 0..num_classes-1
  }

IMPORTANT — class indexing convention:
  torchvision FCOS  → 0-indexed foreground: labels in {0, 1, ..., num_classes-1}
  torchvision RCNN  → 1-indexed foreground: labels in {1, ..., num_classes}, 0=background
  Always use 0-indexed labels with FCOSDetector.

IR images:
  If IR images have 1 channel (true grayscale), use `ir_to_rgb=True` to replicate
  to 3 channels before passing to FCOS (which uses a 3-channel backbone).
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS, fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead
from custom_fcos import build_custom_fcos, CustomFCOS

try:
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
    _HAS_NEW_WEIGHTS_API = True
except ImportError:
    _HAS_NEW_WEIGHTS_API = False


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Trio factory — builds (student, rgb_teacher, ir_teacher)
# ---------------------------------------------------------------------------

def build_fcos_trio(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    ir_to_rgb: bool = True,
    from_coco: bool = False,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0
) -> Tuple["FCOSDetector", "FCOSDetector", "FCOSDetector"]:
    """
    Create (student, rgb_teacher, ir_teacher) — all sharing the same
    architecture.  Teachers are deep copies of student so they start
    with identical weights.  Caller is responsible for freezing/EMA.

    Args:
        num_classes              : foreground classes (1-indexed)
        pretrained_backbone      : ImageNet pretrained ResNet50+FPN
        trainable_backbone_layers: backbone layers to unfreeze
        min_size / max_size      : detection resize range
        ir_to_rgb                : expand 1-ch IR images to 3 channels
        from_coco                : start from COCO pretrained FCOS head

    Returns:
        (student, rgb_teacher, ir_teacher)
    """
    student_core = build_custom_fcos(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        from_coco=from_coco,
        vfl_alpha=vfl_alpha,
        vfl_gamma=vfl_gamma,
        vfl_loss_weight=vfl_loss_weight,
    )

    student = FCOSDetector(student_core, ir_to_rgb=ir_to_rgb)

    rgb_teacher = copy.deepcopy(student)
    ir_teacher  = copy.deepcopy(student)

    return student, rgb_teacher, ir_teacher


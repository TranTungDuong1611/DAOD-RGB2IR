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

try:
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
    _HAS_NEW_WEIGHTS_API = True
except ImportError:
    _HAS_NEW_WEIGHTS_API = False


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class FCOSDetector(nn.Module):
    """
    Thin wrapper around torchvision.models.detection.FCOS.

    Accepts:
      images  : Tensor[B, C, H, W]  (float32, range 0–1 before FCOS transform)
      targets : List[Dict] with "boxes" [N,4] and "labels" [N]

    Internally converts to List[Tensor] expected by torchvision.
    """

    def __init__(self, fcos_model: FCOS, ir_to_rgb: bool = True) -> None:
        """
        Args:
            fcos_model : a configured torchvision FCOS model
            ir_to_rgb  : if True, 1-channel images are replicated to 3 channels
                         before forwarding (needed when IR is stored as grayscale)
        """
        super().__init__()
        self.model = fcos_model
        self.ir_to_rgb = ir_to_rgb

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        image_list = self._to_image_list(images)
        return self.model(image_list, targets)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_image_list(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert [B, C, H, W] batch tensor to List[Tensor[C, H, W]].
        If C==1 and ir_to_rgb=True, replicate to 3 channels.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4-D tensor [B,C,H,W], got shape {tuple(images.shape)}")

        if self.ir_to_rgb and images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)

        return list(images.unbind(dim=0))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scratch(
        cls,
        num_classes: int,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        min_size: int = 600,
        max_size: int = 1000,
        ir_to_rgb: bool = True,
        **fcos_kwargs,
    ) -> "FCOSDetector":
        """
        Build FCOS with ImageNet-pretrained backbone, randomly-initialized head.

        Args:
            num_classes              : number of foreground classes (background excluded)
            pretrained_backbone      : load ImageNet weights for ResNet50+FPN
            trainable_backbone_layers: how many FPN stages to unfreeze (0–5)
            min_size / max_size      : FCOS GeneralizedRCNNTransform resize range
            ir_to_rgb                : replicate 1-channel IR images to 3 channels
        """
        backbone_weights = "DEFAULT" if pretrained_backbone else None
        model = fcos_resnet50_fpn(
            weights=None,                       # head not pretrained
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            **fcos_kwargs,
        )
        return cls(model, ir_to_rgb=ir_to_rgb)

    @classmethod
    def from_coco_pretrained(
        cls,
        num_classes: int,
        min_size: int = 600,
        max_size: int = 1000,
        ir_to_rgb: bool = True,
        **fcos_kwargs,
    ) -> "FCOSDetector":
        """
        Load COCO-pretrained FCOS (91 classes) and replace the classification
        head for `num_classes` foreground classes. Useful for fine-tuning.

        If num_classes == 91, the head is kept as-is.
        """
        if not _HAS_NEW_WEIGHTS_API:
            raise RuntimeError(
                "FCOS_ResNet50_FPN_Weights requires torchvision >= 0.13. "
                "Use FCOSDetector.from_scratch() instead."
            )
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
            **fcos_kwargs,
        )
        if num_classes != 91:
            model = _replace_classification_head(model, num_classes)
        return cls(model, ir_to_rgb=ir_to_rgb)


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
    if from_coco:
        student = FCOSDetector.from_coco_pretrained(
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            ir_to_rgb=ir_to_rgb,
        )
    else:
        student = FCOSDetector.from_scratch(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            ir_to_rgb=ir_to_rgb,
        )

    rgb_teacher = copy.deepcopy(student)
    ir_teacher  = copy.deepcopy(student)

    return student, rgb_teacher, ir_teacher


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replace_classification_head(model: FCOS, num_classes: int) -> FCOS:
    """
    Replace FCOS classification head with one sized for `num_classes`.
    Regression head (bbox + centerness) is kept with pretrained weights.
    """
    old_head = model.head.classification_head
    # Infer in_channels from the existing conv layers
    in_channels = old_head.conv[0][0].in_channels
    num_anchors = old_head.num_anchors

    model.head.classification_head = FCOSClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=torch.nn.GroupNorm,
    )
    return model

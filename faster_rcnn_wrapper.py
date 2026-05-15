"""
FasterRCNNDetector — torchvision Faster RCNN wrapped to match the trainer's detector API.

Label convention mismatch and how it's handled:
  Datasets / trainer  → 0-indexed foreground: labels in {0 .. num_classes-1}
  Faster RCNN         → 1-indexed foreground: labels in {1 .. num_classes}, 0=background

This wrapper converts in both directions automatically:
  train : labels += 1  before passing targets to the model
  eval  : labels -= 1  on model output, background (label==0) predictions are dropped

COCO weight transfer (coco_src_indices):
  Same [1, 3, 2] list as FCOS — these are the 1-indexed COCO class positions:
    0=__background__, 1=person, 2=bicycle, 3=car, ...
  Background (index 0) is always included automatically in _replace_box_predictor.
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

try:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    _HAS_NEW_WEIGHTS_API = True
except ImportError:
    _HAS_NEW_WEIGHTS_API = False


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class FasterRCNNDetector(nn.Module):
    """
    Thin wrapper around torchvision.models.detection.FasterRCNN.

    Accepts:
      images  : Tensor[B, C, H, W]  (float32, range 0-1)
      targets : List[Dict] with "boxes" [N,4] and "labels" [N]  — 0-indexed

    Internally converts labels to 1-indexed for the model and back to 0-indexed
    on output so the rest of the pipeline (losses, evaluator) stays unchanged.
    """

    def __init__(self, rcnn_model: FasterRCNN, ir_to_rgb: bool = True) -> None:
        super().__init__()
        self.model    = rcnn_model
        self.ir_to_rgb = ir_to_rgb

    def forward(
        self,
        images:  torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        image_list = self._to_image_list(images)
        if targets is not None:
            # 0-indexed → 1-indexed for Faster RCNN
            rcnn_targets = [{**t, "labels": t["labels"] + 1} for t in targets]
            return self.model(image_list, rcnn_targets)
        # No targets → inference. Torchvision GeneralizedRCNN gates on
        # self.training, so force eval for this call regardless of caller state.
        was_training = self.model.training
        self.model.eval()
        try:
            preds = self.model(image_list)
        finally:
            if was_training:
                self.model.train()
        return [self._to_0indexed(p) for p in preds]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_0indexed(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """1-indexed RCNN output → 0-indexed, background detections dropped."""
        labels = pred["labels"]
        fg     = labels > 0
        return {
            "boxes":  pred["boxes"][fg],
            "labels": labels[fg] - 1,
            "scores": pred["scores"][fg],
        }

    def _to_image_list(self, images: torch.Tensor) -> List[torch.Tensor]:
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
        **rcnn_kwargs,
    ) -> "FasterRCNNDetector":
        """
        Build Faster RCNN with ImageNet-pretrained backbone, randomly-initialized head.

        Args:
            num_classes : number of foreground classes (background excluded)
        """
        backbone_weights = "DEFAULT" if pretrained_backbone else None
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes + 1,          # +1 for background
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            **rcnn_kwargs,
        )
        return cls(model, ir_to_rgb=ir_to_rgb)

    @classmethod
    def from_coco_pretrained(
        cls,
        num_classes: int,
        min_size: int = 600,
        max_size: int = 1000,
        ir_to_rgb: bool = True,
        coco_src_indices: Optional[List[int]] = None,
        **rcnn_kwargs,
    ) -> "FasterRCNNDetector":
        """
        Load COCO-pretrained Faster RCNN (91 classes) and replace the box predictor
        for `num_classes` foreground classes.

        Args:
            coco_src_indices : 1-indexed COCO class positions for each new foreground
                               class, same convention as FCOS wrapper.
                               Example (FLIR person/car/bicycle): [1, 3, 2]
                               Background (0) is always included automatically.
        """
        if not _HAS_NEW_WEIGHTS_API:
            raise RuntimeError(
                "FasterRCNN_ResNet50_FPN_Weights requires torchvision >= 0.13. "
                "Use FasterRCNNDetector.from_scratch() instead."
            )
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
            **rcnn_kwargs,
        )
        if num_classes != 90:     # COCO has 90 foreground classes
            model = _replace_box_predictor(model, num_classes, coco_src_indices)
        return cls(model, ir_to_rgb=ir_to_rgb)


# ---------------------------------------------------------------------------
# Trio factory
# ---------------------------------------------------------------------------

def build_faster_rcnn_trio(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    ir_to_rgb: bool = True,
    from_coco: bool = False,
    coco_src_indices: Optional[List[int]] = None,
) -> Tuple["FasterRCNNDetector", "FasterRCNNDetector", "FasterRCNNDetector"]:
    """
    Create (student, rgb_teacher, ir_teacher) with Faster RCNN backbone.
    Teachers are deep copies of student. Caller is responsible for freezing/EMA.

    Args:
        coco_src_indices : only used when from_coco=True. See from_coco_pretrained.
    """
    if from_coco:
        student = FasterRCNNDetector.from_coco_pretrained(
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            ir_to_rgb=ir_to_rgb,
            coco_src_indices=coco_src_indices,
        )
    else:
        student = FasterRCNNDetector.from_scratch(
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

def _replace_box_predictor(
    model: FasterRCNN,
    num_classes: int,
    coco_src_indices: Optional[List[int]] = None,
) -> FasterRCNN:
    """
    Replace Faster RCNN box predictor for `num_classes` foreground classes.
    Backbone and RPN are kept with pretrained weights.

    Args:
        coco_src_indices : 1-indexed COCO class positions for each new foreground
                           class (background at 0 is added automatically).
                           When set:
                             cls_score weight/bias  → rows sliced from COCO predictor
                             bbox_pred weight/bias  → 4 rows per class sliced
    """
    old_pred    = model.roi_heads.box_predictor
    in_features = old_pred.cls_score.in_features
    new_pred    = FastRCNNPredictor(in_features, num_classes + 1)  # +1 background

    if coco_src_indices is not None:
        # Build row indices for the new predictor in COCO 91-class space:
        #   [0 (bg), coco_src_indices[0], coco_src_indices[1], ...]
        cls_rows  = [0] + list(coco_src_indices)

        # bbox_pred has 4 rows per class: class c → rows [c*4 : (c+1)*4]
        bbox_rows = [r for c in cls_rows for r in range(c * 4, (c + 1) * 4)]

        with torch.no_grad():
            new_pred.cls_score.weight.copy_(old_pred.cls_score.weight[cls_rows])
            new_pred.cls_score.bias.copy_(  old_pred.cls_score.bias[cls_rows])
            new_pred.bbox_pred.weight.copy_(old_pred.bbox_pred.weight[bbox_rows])
            new_pred.bbox_pred.bias.copy_(  old_pred.bbox_pred.bias[bbox_rows])

    model.roi_heads.box_predictor = new_pred
    return model

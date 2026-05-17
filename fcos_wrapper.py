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

    Harmony support (call enable_harmony() to activate, matching HT paper):
      Eval  mode: runs inference twice — once normally (post-NMS pseudo-labels),
                  once with nms_thresh=1.0 (dense pre-NMS predictions for Eq.9).
                  "pre_nms_boxes" key is added to each output dict so that
                  compute_localization_quality_u() can compute meaningful u_i
                  using dense overlapping FCOS predictions (as in the HT paper).
      Train mode: strips "harmony_weights" from targets before passing to FCOS,
                  then scales all loss components by the batch-mean harmony weight.
                  (Batch-mean is an image-level approximation of HT's per-location
                  weighting, which requires access to FCOS assignment internals.)
    """

    def __init__(self, fcos_model: FCOS, ir_to_rgb: bool = True) -> None:
        super().__init__()
        self.model    = fcos_model
        self.ir_to_rgb = ir_to_rgb
        self._harmony_enabled: bool = False

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        image_list = self._to_image_list(images)

        if self._harmony_enabled:
            if self.training and targets is not None:
                return self._forward_train_harmony(image_list, targets)
            if not self.training:
                return self._forward_eval_harmony(image_list)

        return self.model(image_list, targets)

    # ------------------------------------------------------------------
    # Harmony helpers
    # ------------------------------------------------------------------

    def _forward_train_harmony(
        self,
        image_list: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Strip "harmony_weights" from targets (unknown key crashes FCOS), run
        the standard FCOS loss, then scale every component by the batch-mean
        harmony weight.

        GT targets never carry "harmony_weights" → standard loss, no scaling.
        Pseudo targets carry "harmony_weights" → scaled loss.
        """
        harmony_weights: List[Optional[torch.Tensor]] = []
        clean_targets:   List[Dict[str, torch.Tensor]] = []
        for t in targets:
            hw = t.get("harmony_weights", None)
            harmony_weights.append(hw)
            clean_targets.append({k: v for k, v in t.items() if k != "harmony_weights"})

        loss_dict: Dict[str, torch.Tensor] = self.model(image_list, clean_targets)

        valid_hw = [hw for hw in harmony_weights if hw is not None and hw.numel() > 0]
        if valid_hw:
            mean_h = torch.cat(valid_hw).mean()
            loss_dict = {k: v * mean_h for k, v in loss_dict.items()}

        return loss_dict

    def _forward_eval_harmony(
        self,
        image_list: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Eval forward: post-NMS detections + pre-NMS boxes for Eq.9.

        Runs FCOS twice:
          1. Normal inference  (nms_thresh=0.6) → post-NMS pseudo-labels.
          2. NMS suppressed    (nms_thresh=1.0) → all decoded boxes (pre-NMS).

        "pre_nms_boxes" attached to each output dict for use in
        compute_localization_quality_u().  The second pass is teacher-only
        (eval+no_grad) so the overhead is one extra forward without backward.
        """
        post_nms: List[Dict[str, torch.Tensor]] = self.model(image_list)
        pre_nms:  List[Dict[str, torch.Tensor]] = self._get_pre_nms_boxes(image_list)
        for d, pnms in zip(post_nms, pre_nms):
            d["pre_nms_boxes"]  = pnms["boxes"].detach()
            d["pre_nms_scores"] = pnms["scores"].detach()
        return post_nms

    def _get_pre_nms_boxes(
        self,
        image_list: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Run FCOS with NMS disabled → all decoded boxes per image."""
        orig_nms = self.model.nms_thresh
        self.model.nms_thresh = 1.0
        try:
            result: List[Dict[str, torch.Tensor]] = self.model(image_list)
        finally:
            self.model.nms_thresh = orig_nms
        return result

    def enable_harmony(self) -> "FCOSDetector":
        """
        Activate harmony mode (call on student AND both teachers).

        Student : scales pseudo-label loss by batch-mean harmony weight.
        Teachers: return pre-NMS boxes alongside post-NMS pseudo-labels so
                  that compute_localization_quality_u() can use dense FCOS
                  predictions for Eq.9 — matching the Harmonious Teacher paper.

        Returns self for chaining:
            student = FCOSDetector.from_scratch(...).enable_harmony()
        """
        self._harmony_enabled = True
        return self

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
        coco_src_indices: Optional[List[int]] = None,
        **fcos_kwargs,
    ) -> "FCOSDetector":
        """
        Load COCO-pretrained FCOS (91 classes) and replace the classification
        head for `num_classes` foreground classes. Useful for fine-tuning.

        If num_classes == 91, the head is kept as-is.

        Args:
            coco_src_indices : COCO 91-class output indices to slice into the new head,
                               one per new class. When provided, the class-agnostic conv
                               layers and the matching cls_logits rows are transferred
                               instead of random init.
                               Example (FLIR person/car/bicycle): [1, 3, 2]
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
            model = _replace_classification_head(model, num_classes, coco_src_indices)
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
    coco_src_indices: Optional[List[int]] = None,
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
        coco_src_indices         : COCO 91-class indices to slice into the new head
                                   (only used when from_coco=True). See
                                   FCOSDetector.from_coco_pretrained for details.

    Returns:
        (student, rgb_teacher, ir_teacher)
    """
    if from_coco:
        student = FCOSDetector.from_coco_pretrained(
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            ir_to_rgb=ir_to_rgb,
            coco_src_indices=coco_src_indices,
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

def _replace_classification_head(
    model: FCOS,
    num_classes: int,
    coco_src_indices: Optional[List[int]] = None,
) -> FCOS:
    """
    Replace FCOS classification head with one sized for `num_classes`.
    Regression head (bbox + centerness) is kept with pretrained weights.

    Args:
        coco_src_indices : if provided, transfer weights from the COCO head instead of
                           random init. List of length `num_classes` — each entry is the
                           0-based output index in the COCO 91-class cls_logits for the
                           i-th new class. Example (FLIR): [1, 3, 2] → person=1, car=3,
                           bicycle=2. When set, class-agnostic conv layers are copied and
                           cls_logits weight/bias are sliced from the COCO head.
    """
    old_head = model.head.classification_head
    # Infer in_channels from the existing conv layers.
    # torchvision < 0.13 : conv is ModuleList of Sequential → conv[0][0] is Conv2d
    # torchvision >= 0.13 : conv is flat Sequential         → conv[0]    is Conv2d
    first = old_head.conv[0]
    in_channels = (first[0] if isinstance(first, torch.nn.Sequential) else first).in_channels
    num_anchors = old_head.num_anchors

    # norm_layer wrapper: handles both torchvision calling conventions
    #   old (< 0.13): norm_layer(num_groups, num_channels)
    #   new (>= 0.13): norm_layer(num_channels)
    # args[-1] is always num_channels regardless of convention.
    def _group_norm(*args):
        return torch.nn.GroupNorm(32, args[-1])

    new_head = FCOSClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=_group_norm,
    )

    if coco_src_indices is not None:
        # Transfer class-agnostic conv stack (same architecture, same channels)
        new_head.conv.load_state_dict(old_head.conv.state_dict())

        # Slice cls_logits for the target classes.
        # cls_logits.weight : [old_num_classes * num_anchors, in_channels, 1, 1]
        # cls_logits.bias   : [old_num_classes * num_anchors]
        # Each class c occupies rows [c*num_anchors : (c+1)*num_anchors].
        src_rows = [
            c * num_anchors + a
            for c in coco_src_indices
            for a in range(num_anchors)
        ]
        with torch.no_grad():
            new_head.cls_logits.weight.copy_(old_head.cls_logits.weight[src_rows])
            new_head.cls_logits.bias.copy_(old_head.cls_logits.bias[src_rows])

    model.head.classification_head = new_head
    return model

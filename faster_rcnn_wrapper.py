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
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

try:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    _HAS_NEW_WEIGHTS_API = True
except ImportError:
    _HAS_NEW_WEIGHTS_API = False


# ---------------------------------------------------------------------------
# Harmony-weighted loss helpers
# ---------------------------------------------------------------------------

def weighted_fastrcnn_loss(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: List[torch.Tensor],
    regression_targets: List[torch.Tensor],
    proposal_harmony_weights: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like torchvision's fastrcnn_loss but multiplies each proposal's loss by its
    harmony weight.  Background proposals get neg_proposal_weight (set in
    _build_proposal_harmony_weights before calling here).

    Normalisation matches the standard: classification loss is mean over all
    sampled proposals; box loss is sum-over-positives / total-sampled.
    This keeps the gradient scale comparable to the unweighted baseline.
    """
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)
    hw_cat = torch.cat(proposal_harmony_weights, dim=0)  # [N_sampled]

    # --- Classification loss (all sampled proposals) ---
    log_p = F.log_softmax(class_logits, dim=-1)
    nll = -log_p[torch.arange(len(labels_cat), device=labels_cat.device), labels_cat]
    classification_loss = (hw_cat * nll).mean()

    # --- Box regression loss (positive proposals only) ---
    sampled_pos = torch.where(labels_cat > 0)[0]
    if sampled_pos.numel() == 0:
        box_loss = class_logits.sum() * 0.0
    else:
        labels_pos = labels_cat[sampled_pos]
        N = class_logits.shape[0]
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        pos_hw = hw_cat[sampled_pos]  # [N_pos]
        per_box = F.smooth_l1_loss(
            box_regression[sampled_pos, labels_pos],
            regression_targets_cat[sampled_pos],
            beta=1.0 / 9,
            reduction="none",
        ).sum(dim=-1)   # [N_pos] — sum over 4 coords, mirrors torchvision reduction="sum"
        box_loss = (pos_hw * per_box).sum() / labels_cat.numel()

    return classification_loss, box_loss


# ---------------------------------------------------------------------------
# Harmony-aware RoI head
# ---------------------------------------------------------------------------

class HarmonyRoIHeads(RoIHeads):
    """
    Drop-in replacement for torchvision's RoIHeads that supports per-box
    harmony weights in the pseudo-label loss.

    Install via FasterRCNNDetector.enable_harmony() — this changes the
    roi_heads __class__ in-place (all weights are preserved).

    How it works
    ------------
    During training, if any target dict contains "harmony_weights" [N_gt]:
      1. Strip "harmony_weights" from targets before select_training_samples.
      2. Use matched_idxs (after fg/bg sampling) to look up the harmony weight
         for each sampled proposal from its matched GT box.
      3. Replace fastrcnn_loss with weighted_fastrcnn_loss.
    If no target has "harmony_weights", falls back to standard fastrcnn_loss.
    GT supervised targets never contain "harmony_weights", so the GT branch
    is completely unaffected.
    """

    neg_proposal_weight: float = 0.0  # set by enable_harmony()

    def forward(self, features, proposals, image_shapes, targets=None):
        # ---- Training path ------------------------------------------------
        if self.training:
            # Extract harmony weights before mutating targets
            harmony_weights_list: Optional[List[Optional[torch.Tensor]]] = None
            if targets is not None:
                if any("harmony_weights" in t for t in targets):
                    harmony_weights_list = [t.get("harmony_weights", None)
                                            for t in targets]
                    # Strip so torchvision internals see only boxes/labels
                    targets = [{k: v for k, v in t.items() if k != "harmony_weights"}
                               for t in targets]

            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)

            if harmony_weights_list is not None:
                proposal_hw = self._build_proposal_harmony_weights(
                    harmony_weights_list, matched_idxs, labels
                )
                loss_cls, loss_box = weighted_fastrcnn_loss(
                    class_logits, box_regression, labels,
                    regression_targets, proposal_hw,
                )
            else:
                loss_cls, loss_box = fastrcnn_loss(
                    class_logits, box_regression, labels, regression_targets
                )

            return [], {"loss_classifier": loss_cls, "loss_box_reg": loss_box}

        # ---- Inference path (unchanged) -----------------------------------
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        result = [{"boxes": b, "labels": l, "scores": s}
                  for b, l, s in zip(boxes, labels, scores)]
        return result, {}

    def _build_proposal_harmony_weights(
        self,
        harmony_weights_list: List[Optional[torch.Tensor]],
        matched_idxs: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Build per-proposal harmony weight tensors.

        matched_idxs[i] : [N_sampled_i] — GT-box index for each sampled proposal
                           (clamped to [0, N_gt-1] by select_training_samples).
        labels[i]        : [N_sampled_i] — 0=background, >0=foreground class.
        """
        out = []
        for hw, midxs, lbls in zip(harmony_weights_list, matched_idxs, labels):
            device = midxs.device
            fg_mask = lbls > 0

            if hw is None or hw.numel() == 0:
                # No per-box weights: fg→1.0, bg→neg_proposal_weight
                w = torch.where(
                    fg_mask,
                    torch.ones(len(lbls), device=device),
                    torch.full((len(lbls),), self.neg_proposal_weight, device=device),
                )
            else:
                hw = hw.to(device)
                # Look up harmony weight for each proposal via its matched GT index
                per_prop = hw[midxs.clamp(0, len(hw) - 1)]
                w = torch.where(
                    fg_mask,
                    per_prop,
                    torch.full_like(per_prop, self.neg_proposal_weight),
                )
            out.append(w)
        return out


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
    # Harmony support
    # ------------------------------------------------------------------

    def enable_harmony(self, neg_proposal_weight: float = 0.0) -> "FasterRCNNDetector":
        """
        Replace the stock RoI head with HarmonyRoIHeads in-place.

        This is a lightweight __class__ swap — all trained weights are
        preserved.  Call once before training when use_harmony_weight=True.

        Args:
            neg_proposal_weight : harmony weight assigned to background
                                  proposals in the pseudo-label ROI loss.
                                  0.0 = ignore background (recommended).
                                  0.25 = mild background supervision.

        Returns self for chaining:
            student = FasterRCNNDetector.from_scratch(...).enable_harmony()
        """
        roi = self.model.roi_heads
        if not isinstance(roi, HarmonyRoIHeads):
            roi.__class__ = HarmonyRoIHeads
        roi.neg_proposal_weight = neg_proposal_weight
        return self

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

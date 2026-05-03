import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchvision.models.detection import FCOS as TV_FCOS
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import (
    FCOSClassificationHead,
    FCOSRegressionHead,
    FCOSHead,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou_loss

try:
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
    _HAS_WEIGHTS_API = True
except ImportError:
    _HAS_WEIGHTS_API = False

from loss import HMfocalLoss, QFLv2, giou_loss_ltrb


def _box_iou_for_target(
    pred_boxes: Tensor,   # [N, 4]  xyxy
    gt_boxes:   Tensor,   # [N, 4]  xyxy
) -> Tensor:
    """Calculate IoU for target assignment."""
    inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_gt   = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * \
                (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)

    union = area_pred + area_gt - inter_area
    eps = torch.finfo(pred_boxes.dtype).eps
    return inter_area / union.clamp(min=eps)


class HMFocalClassificationHead(FCOSClassificationHead):
    """
    FCOSClassificationHead with HMfocalLoss. This class helps to change the loss function
    used for classification from binary to iou. It maintains the exact same network architecture (layers, convolutions) as the 
    original FCOS head, only overriding the loss computation logic.
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        prior_probability: float = 0.01,
        norm_layer=None,
        # HMfocalLoss params
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        vfl_weight_type: str = "iou",
        vfl_loss_weight: float = 1.0,
    ):
        if norm_layer is None:
            norm_layer = nn.GroupNorm

        super().__init__(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            prior_probability=prior_probability,
            norm_layer=norm_layer,
        )

        self.hm_focal_loss = HMfocalLoss(
            use_sigmoid=True,
            alpha=vfl_alpha,
            gamma=vfl_gamma,
            weight_type=vfl_weight_type,
            loss_weight=vfl_loss_weight,
        )

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        matched_idxs: List[Tensor],

        decoded_boxes_per_level: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Calculates the classification loss using IoU-weighted soft targets.

        Args:
            targets              : Ground truth data for the current batch, list of {"boxes": [N,4] 
                                and this is absolute coordinates, "labels": [N], foreground class indices}
            head_outputs         : The raw predictions from the forward pass of the head. 
                                  {"cls_logits": [B, HWA, num_classes]}
            matched_idxs         : list length B, each [HWA] matched gt index (-1 = bg)
            decoded_boxes_per_level: The predicted bounding boxes that have been transformed from offsets (ltrb) 
                                    to absolute coordinates (xyxy) shape list[list[Tensor[HW, 4]]]
                                     
                                   
        """
        cls_logits = head_outputs["cls_logits"]  # [B, HWA, C]

        all_cls_iou_targets = []

        for img_i, (targets_per_img, matched_per_img) in enumerate(
            zip(targets, matched_idxs)
        ):
            gt_boxes_i  = targets_per_img["boxes"]   # [N_gt, 4]
            gt_labels_i = targets_per_img["labels"]  # [N_gt]
            num_anchors = matched_per_img.shape[0]

            # fg / bg masks
            fg_mask = matched_per_img >= 0
            bg_mask = ~fg_mask

            # Build one-hot class target
            cls_target = torch.zeros(
                num_anchors, self.num_classes,
                dtype=cls_logits.dtype,
                device=cls_logits.device,
            )

            if fg_mask.any():
                fg_idxs     = matched_per_img[fg_mask]          # gt indices for fg anchors
                fg_labels   = gt_labels_i[fg_idxs]              # class labels

                # --- IoU-weighted target (Varifocal style) ---
                if decoded_boxes_per_level is not None:
                    pred_boxes_i = torch.cat(decoded_boxes_per_level[img_i], dim=0)  # [HWA, 4]
                    pred_boxes_fg = pred_boxes_i[fg_mask]   # [Nfg, 4]
                    gt_boxes_fg   = gt_boxes_i[fg_idxs]     # [Nfg, 4]
                    with torch.no_grad():
                        iou_weights = _box_iou_for_target(
                            pred_boxes_fg.detach(), gt_boxes_fg
                        ).clamp(min=0.0)  # [Nfg]
                    cls_target[fg_mask, fg_labels] = iou_weights
                else:
                    # Fallback: one-hot
                    cls_target[fg_mask, fg_labels] = 1.0

            all_cls_iou_targets.append(cls_target)

        cls_targets = torch.stack(all_cls_iou_targets, dim=0)   # [B, HWA, C]

        num_fg = sum((m >= 0).sum() for m in matched_idxs)
        num_fg = max(1, num_fg.item() if isinstance(num_fg, Tensor) else num_fg)

        loss = self.hm_focal_loss(
            cls_logits.reshape(-1, self.num_classes),
            cls_targets.reshape(-1, self.num_classes),
            avg_factor=num_fg,
        )
        return loss


class CustomFCOS(TV_FCOS):
    """
    Custom FCOS implementation that integrates HM-Focal Loss and 
    provides raw feature extraction for Semi-supervised Distillation.
    """
    def __init__(
        self,
        backbone,
        num_classes: int,
        # Transform parameters
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # Head parameters
        anchor_generator=None,
        head=None,
        # Detection parameters
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.6,
        detections_per_img: int = 100,
        topk_candidates: int = 1000,
        # HMfocalLoss parameters
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
        vfl_weight_type: str = "iou",
        vfl_loss_weight: float = 1.0,
    ):
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            anchor_generator=anchor_generator,
            head=head,
            center_sampling_radius=center_sampling_radius,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            topk_candidates=topk_candidates,
        )

        # Replace the default Torchvision classification head with HMFocalClassificationHead
        old_cls_head = self.head.classification_head
        in_channels  = old_cls_head.conv[0][0].in_channels
        num_anchors  = old_cls_head.num_anchors

        self.head.classification_head = HMFocalClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=nn.GroupNorm,
            vfl_alpha=vfl_alpha,
            vfl_gamma=vfl_gamma,
            vfl_weight_type=vfl_weight_type,
            vfl_loss_weight=vfl_loss_weight,
        )

    def forward_for_distill(self, images: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Returns raw per-level FPN tensors for the distillation loss (Teacher-Student).
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            logits_levels: Raw classification scores per FPN level.
            deltas_levels: Raw box regression offsets (ltrb) per FPN level.
            quality_levels: Raw centerness scores per FPN level.
            box_xyxy_per_image: Decoded absolute coordinates [xyxy] for UHL loss.
        """
        # 1. Preprocess images using torchvision's internal transform (resize, normalize)
        image_list = list(images.unbind(dim=0))
        transformed_images, _ = self.transform(image_list, None)

        # 2. Extract features from Backbone + FPN
        features = self.backbone(transformed_images.tensors)
        features_list = list(features.values())

        # 3. Manually pass features through subnets to keep per-level outputs
        logits_levels, deltas_levels, quality_levels = [], [], []

        for feature in features_list:
            cls_f, reg_f = feature, feature
            
            # Process Classification Subnet
            for layer in self.head.classification_head.conv:
                cls_f = layer(cls_f)
            logits_levels.append(self.head.classification_head.cls_logits(cls_f))

            # Process Regression Subnet
            for layer in self.head.regression_head.conv:
                reg_f = layer(reg_f)
            deltas_levels.append(self.head.regression_head.bbox_reg(reg_f))
            quality_levels.append(self.head.regression_head.bbox_ctrness(reg_f))

        # 4. Decode bounding boxes into xyxy format for per-image processing
        # Generate anchor points (centers) based on FPN strides
        anchors = self.anchor_generator(transformed_images, features_list)
        
        box_xyxy_per_image = []
        B = images.shape[0]
        
        # Flatten deltas across all levels for decoding
        all_deltas_flat = []
        for d in deltas_levels:
            all_deltas_flat.append(d.permute(0, 2, 3, 1).reshape(B, -1, 4))
        all_deltas = torch.cat(all_deltas_flat, dim=1) # [B, HWA_total, 4]

        for i in range(B):
            anch_i = anchors[i]  # [HWA, 4]
            delt_i = all_deltas[i]
            
            # Anchor coordinates are represented as [x1, y1, x2, y2] where x1=x2 and y1=y2 (points)
            cx = (anch_i[:, 0] + anch_i[:, 2]) / 2.0
            cy = (anch_i[:, 1] + anch_i[:, 3]) / 2.0
            
            # Convert ltrb (distance to 4 sides) to xyxy (absolute coordinates)
            box_xyxy_per_image.append(torch.stack([
                cx - delt_i[:, 0], cy - delt_i[:, 1],
                cx + delt_i[:, 2], cy + delt_i[:, 3]
            ], dim=-1))

        return logits_levels, deltas_levels, quality_levels, box_xyxy_per_image

    def compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Calculates loss by overriding the default classification loss with HM-Focal logic. 
        Used for supervised training
        """
        bbox_regression     = head_outputs["bbox_regression"]
        bbox_ctrness        = head_outputs["bbox_ctrness"]
        cls_logits          = head_outputs["cls_logits"]

        # Decode predicted boxes to use as targets for IoU-weighted classification
        decoded_boxes_per_image = self._decode_boxes_for_iou(
            bbox_regression, anchors
        )

        # 1. Classification loss: HM-Focal with soft targets (IoU)
        loss_cls = self.head.classification_head.compute_loss(
            targets,
            head_outputs,
            matched_idxs,
            decoded_boxes_per_level=decoded_boxes_per_image,
        )

        # 2. Regression and Centerness losses: Maintain original GIoU and BCE logic
        loss_bbox_reg, loss_bbox_ctrness = self._regression_centerness_loss(
            targets, head_outputs, anchors, matched_idxs
        )

        return {
            "classification": loss_cls,
            "bbox_regression": loss_bbox_reg,
            "bbox_ctrness": loss_bbox_ctrness,
        }

    def _decode_boxes_for_iou(
        self,
        bbox_regression: Tensor,  # [HWA_total, 4] (ltrb deltas)
        anchors: List[Tensor],    # list[Tensor[HWA_i, 4]] per image
    ) -> List[List[Tensor]]:
        """
        Decodes LTRB regression deltas into absolute XYXY boxes for IoU calculation.
        """
        # Split flattened regression output back into per-image tensors
        splits = [len(a) for a in anchors]
        bbox_reg_per_img = bbox_regression.split(splits, dim=0)

        decoded_per_img = []
        for bbox_reg_i, anchors_i in zip(bbox_reg_per_img, anchors):
            # Calculate centers from anchor points
            cx = (anchors_i[:, 0] + anchors_i[:, 2]) / 2.0
            cy = (anchors_i[:, 1] + anchors_i[:, 3]) / 2.0

            l, t, r, b = bbox_reg_i.unbind(dim=-1)

            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b

            pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [HWA, 4]
            # Wrap in a list to keep level-compatibility even if flattened
            decoded_per_img.append([pred_boxes])

        return decoded_per_img

    def _regression_centerness_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates Regression loss (GIoU) and Centerness loss (BCE).
        """
        bbox_regression = head_outputs["bbox_regression"]   # [HWA_total, 4]
        bbox_ctrness    = head_outputs["bbox_ctrness"]      # [HWA_total, 1]

        splits = [len(a) for a in anchors]
        bbox_reg_split   = bbox_regression.split(splits, dim=0)
        bbox_ctrn_split  = bbox_ctrness.split(splits, dim=0)

        losses_bbox_reg   = []
        losses_bbox_ctrns = []

        for targets_i, matched_i, bbox_reg_i, bbox_ctrn_i, anchors_i in zip(
            targets, matched_idxs, bbox_reg_split, bbox_ctrn_split, anchors
        ):
            fg_mask = matched_i >= 0

            # Handle cases with no foreground objects
            if not fg_mask.any():
                losses_bbox_reg.append(bbox_reg_i.sum() * 0.0)
                losses_bbox_ctrns.append(bbox_ctrn_i.sum() * 0.0)
                continue

            # Get matched Ground Truth boxes for foreground anchors
            fg_idxs = matched_i[fg_mask]
            gt_boxes_fg = targets_i["boxes"][fg_idxs]   # [Nfg, 4] (xyxy)

            # Decode predicted boxes for foreground only
            cx = (anchors_i[fg_mask, 0] + anchors_i[fg_mask, 2]) / 2.0
            cy = (anchors_i[fg_mask, 1] + anchors_i[fg_mask, 3]) / 2.0
            pred_reg_fg = bbox_reg_i[fg_mask]            # [Nfg, 4] (ltrb)

            pred_boxes_fg = torch.stack([
                cx - pred_reg_fg[:, 0],
                cy - pred_reg_fg[:, 1],
                cx + pred_reg_fg[:, 2],
                cy + pred_reg_fg[:, 3],
            ], dim=-1)                                   # [Nfg, 4] (xyxy)

            # Calculate GIoU loss
            reg_loss = generalized_box_iou_loss(
                pred_boxes_fg,
                gt_boxes_fg,
                reduction="sum",
            ) / max(1, fg_mask.sum().item())
            losses_bbox_reg.append(reg_loss)

            # Calculate Centerness target: geometric mean of min/max ratios of l,t,r,b
            lt = torch.stack([cx - gt_boxes_fg[:, 0], cy - gt_boxes_fg[:, 1]], dim=-1)
            rb = torch.stack([gt_boxes_fg[:, 2] - cx, gt_boxes_fg[:, 3] - cy], dim=-1)
            lt = lt.clamp(min=0.0)
            rb = rb.clamp(min=0.0)
            
            ctrness_target = torch.sqrt(
                (lt.min(dim=-1).values / lt.max(dim=-1).values.clamp(min=1e-6))
                * (rb.min(dim=-1).values / rb.max(dim=-1).values.clamp(min=1e-6))
            )                                            # [Nfg]

            # Binary Cross Entropy for centerness prediction
            pred_ctrness_fg = bbox_ctrn_i[fg_mask].squeeze(1)  # [Nfg]
            ctrness_loss = F.binary_cross_entropy_with_logits(
                pred_ctrness_fg,
                ctrness_target,
                reduction="sum",
            ) / max(1, fg_mask.sum().item())
            losses_bbox_ctrns.append(ctrness_loss)

        return sum(losses_bbox_reg), sum(losses_bbox_ctrns)

class FCOSDetector(nn.Module):
    """
    Wrapper to bridge Batch-Tensor API and List-of-Tensors API.
    Also handles IR 1-ch to 3-ch conversion.
    """
    def __init__(self, fcos_model: CustomFCOS, ir_to_rgb: bool = True) -> None:
        super().__init__()
        self.model = fcos_model  
        self.ir_to_rgb = ir_to_rgb

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None):
        # Automatic conversion from [B, C, H, W] to List[Tensor]
        image_list = self._to_image_list(images)
        return self.model(image_list, targets)
    
    def forward_for_distill(self, images: torch.Tensor):
        # Automatic conversion from [B, C, H, W] to List[Tensor]
        image_list = self._to_image_list(images)
        return self.model.forward_for_distill(image_list)

    def _to_image_list(self, images: torch.Tensor) -> List[torch.Tensor]:
        if self.ir_to_rgb and images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        return list(images.unbind(dim=0))

def build_custom_fcos(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    from_coco: bool = False,
    # HM-Focal Loss hyperparameters
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0,
    **fcos_kwargs,
) -> CustomFCOS:
    """
    Factory function to create a CustomFCOS model with HM-Focal Loss.

    Args:
        num_classes: Number of foreground classes (0-indexed).
        pretrained_backbone: Load ImageNet weights (ignored if from_coco=True).
        trainable_backbone_layers: Number of trainable FPN stages (0–5).
        min_size / max_size: Resize range for GeneralizedRCNNTransform.
        from_coco: If True, load COCO pretrained weights (91 classes) then replace head.
        vfl_alpha/gamma: HM-Focal Loss focal hyperparameters.
        vfl_weight_type: "iou" or other weighting types supported by HMfocalLoss.
        vfl_loss_weight: Scaling factor for the classification loss.
        **fcos_kwargs: Extra arguments for the FCOS model (e.g., nms_thresh, score_thresh).

    Returns:
        An instance of CustomFCOS.
    """
    
    # 1. Instantiate the base Torchvision FCOS model to extract components
    if from_coco:
        if not _HAS_WEIGHTS_API:
            raise RuntimeError(
                "FCOS_ResNet50_FPN_Weights requires torchvision >= 0.13. "
                "Update torchvision or set from_coco=False."
            )
        
        # Load COCO weights (91 classes)
        base_model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
            **fcos_kwargs,
        )
    else:
        # Load ImageNet weights for backbone only, head initialized from scratch
        backbone_weights = "DEFAULT" if pretrained_backbone else None
        base_model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            **fcos_kwargs,
        )

    # 2. Extract standard FCOS parameters from base_model or kwargs
    center_sampling_radius = fcos_kwargs.get("center_sampling_radius", 1.5)

    # 3. Create the CustomFCOS instance
    # CustomFCOS.__init__ will automatically handle the replacement of the 
    # classification head with HMFocalClassificationHead.
    custom_model = CustomFCOS(
        backbone=base_model.backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        image_mean=base_model.transform.image_mean,
        image_std=base_model.transform.image_std,
        anchor_generator=base_model.anchor_generator,
        head=base_model.head,
        center_sampling_radius=center_sampling_radius,
        score_thresh=fcos_kwargs.get("score_thresh", base_model.score_thresh),
        nms_thresh=fcos_kwargs.get("nms_thresh", base_model.nms_thresh),
        detections_per_img=fcos_kwargs.get("detections_per_img", base_model.detections_per_img),
        topk_candidates=fcos_kwargs.get("topk_candidates", base_model.topk_candidates),
        vfl_alpha=vfl_alpha,
        vfl_gamma=vfl_gamma,
        vfl_weight_type=vfl_weight_type,
        vfl_loss_weight=vfl_loss_weight,
    )

    # 4. Synchronize the transform (preprocessing) layer
    custom_model.transform = base_model.transform

    return custom_model

def build_fcos_trio(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    ir_to_rgb: bool = True,
    from_coco: bool = False,
    # HM-Focal Loss hyperparameters
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0,
    **fcos_kwargs
) -> Tuple[FCOSDetector, FCOSDetector, FCOSDetector]:
    """
    Builds the (Student, RGB Teacher, IR Teacher) trio.
    Each core model is a CustomFCOS, wrapped inside an FCOSDetector.
    """

    # 1. Build the core CustomFCOS model
    # This core handles architecture, HM-Focal Loss, and raw feature extraction
    student_core = build_custom_fcos(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        from_coco=from_coco,
        vfl_alpha=vfl_alpha,
        vfl_gamma=vfl_gamma,
        vfl_weight_type=vfl_weight_type,
        vfl_loss_weight=vfl_loss_weight,
        **fcos_kwargs
    )

    # 2. Wrap the core model with FCOSDetector
    # This wrapper handles Batch Tensor -> List conversion and IR 1-ch to 3-ch expansion
    student = FCOSDetector(student_core, ir_to_rgb=ir_to_rgb)

    # 3. Create Teachers by deep-copying the Student
    # Deep copy ensures teachers start with identical weights but are independent objects
    rgb_teacher = copy.deepcopy(student)
    ir_teacher  = copy.deepcopy(student)

    # 4. Note: The caller is responsible for freezing teacher weights 
    # and setting them to eval() mode during training.
    
    return student, rgb_teacher, ir_teacher

class FCOSDistillAdapter:
    """
    Adapter for FCOS Knowledge Distillation. 
    Manages the interaction between a Student model and dual Teachers (RGB and IR).
    """
    def __init__(self, student, rgb_teacher, ir_teacher, sched, config):
        """
        Args:
            student: The FCOSDetector instance being trained.
            rgb_teacher: The FCOSDetector instance acting as the RGB expert.
            ir_teacher: The FCOSDetector instance acting as the Thermal/IR expert.
            sched: The adaptive threshold scheduler.
            config: Configuration object containing alpha, beta, ratio, etc.
        """
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.config = config # Contains hyperparameters: alpha, beta, ratio...
        self.sched = sched 
        self.vfl_loss = HMfocalLoss(use_sigmoid=True)

    def distill_step(self, student_images, rgb_images, ir_images, global_step: int):
        """
        Performs one distillation step across all teachers.
        
        Returns:
            Dict containing distillation losses for both RGB and IR teachers.
        """
        phase = self.config.get_phase(global_step)

        rgb_ratio, rgb_min_hm = self.config.distill.rgb_teacher.get_params(phase)
        ir_ratio, ir_min_hm = self.config.distill.ir_teacher.get_params(phase)
        # 1. Get Student raw outputs (requires gradient for backpropagation)
        s_res = self.student.forward_for_distill(student_images)
        
        # 2. Get Teacher raw outputs (no-grad context to save memory/computation)
        with torch.no_grad():
            r_res = self.rgb_teacher.forward_for_distill(rgb_images)
            
            # Note: ir_images should be pre-processed (1-ch to 3-ch) 
            # if the model backbone expects 3 channels.
            i_res = self.ir_teacher.forward_for_distill(ir_images)
            
        loss_dict = {}
        
        # 3. Calculate Knowledge Distillation (KD) losses for each modality
        loss_dict.update(self._calculate_kd(s_res, r_res, rgb_ratio, rgb_min_hm, name="_rgb"))
        loss_dict.update(self._calculate_kd(s_res, i_res, ir_ratio, ir_min_hm, name="_ir"))
        
        return loss_dict

    def _calculate_kd(self, s_res, t_res, ratio, min_hm, name=""):
        """
        Internal logic to compute KD losses between a Student and a Teacher.
        """
        s_logits, s_deltas, s_quality, s_boxes = s_res
        t_logits, t_deltas, t_quality, t_boxes = t_res
        
        # Helper to flatten per-level FPN tensors into a single continuous tensor
        def flatten(t_list, K):
            return torch.cat([t.permute(0, 2, 3, 1).reshape(-1, K) for t in t_list], dim=0)

        num_cls = s_logits[0].shape[1]
        sl_f, tl_f = flatten(s_logits, num_cls), flatten(t_logits, num_cls)
        sd_f, td_f = flatten(s_deltas, 4), flatten(t_deltas, 4)
        sq_f, tq_f = flatten(s_quality, 1), flatten(t_quality, 1)

        # 1. Uncertainty Region Selection (Harmony Measure - HM)
        with torch.no_grad():
            t_probs = tl_f.sigmoid()
            cls_p, _ = t_probs.max(dim=1)
            iou_p = tq_f.sigmoid().squeeze()
            
            # Calculate Harmony Measure to gauge teacher reliability
            # Default hyperparameters: alpha=1.0, beta=1.0
            alpha = self.config.distill.hm_alpha
            beta = self.config.distill.hm_beta
            hm = (cls_p ** alpha) * (iou_p ** beta)
            
            # Select Top-K most reliable regions based on HM (e.g., Top 1%)
            count = int(tl_f.size(0) * ratio) # ratio=0.01
            _, top_inds = torch.topk(hm, count)

            hm_mask = hm > min_hm
            
            final_mask = torch.zeros_like(hm)
            final_mask[top_inds] = 1.0
            final_mask = final_mask * hm_mask.float() 

            fg_num = final_mask.sum().clamp(min=1.0)

        # 2. Compute distillation weights based on uncertainty/reliability
        un_alpha = self.config.distill.un_regular_alpha
        loss_weight = torch.exp(-(1 - hm.detach()) / un_alpha)

        # 3. Classification Distillation (using Quality Focal Loss v2)
        l_logits = QFLv2(sl_f.sigmoid(), tl_f.sigmoid(), weight=loss_weight * final_mask) / fg_num

        # 4. Box Regression Distillation (using GIoU Loss for LTRB format)
        pos_idx = final_mask > 0
        if pos_idx.any():
            # GIoU Distillation
            l_deltas = (giou_loss_ltrb(sd_f[pos_idx], td_f[pos_idx]) * loss_weight[pos_idx]).mean()
            
            # Centerness/Quality Distillation (BCE)
            l_quality = F.binary_cross_entropy(
                sq_f.sigmoid()[pos_idx], 
                tq_f.sigmoid()[pos_idx], 
                weight=loss_weight[pos_idx].unsqueeze(1)
            )
        else:
            # Fallback if no points pass the filters
            l_deltas = sd_f.sum() * 0.0
            l_quality = sq_f.sum() * 0.0
        
        return {
            f"kd_logits{name}": l_logits,
            f"kd_deltas{name}": l_deltas,
            f"kd_quality{name}": l_quality
        }


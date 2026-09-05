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

from losses import HMfocalLoss

def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction="mean"):
    """
    Use to calculate the loss logits. 
    With the input is the probability of Student and pseudo from teacher, and the weight. 
    Output is a scalar.
    """
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss

def _giou_loss_ltrb(pred, target, weight=None, reduction="mean"):
    """
    GIoU loss cho format LTRB. Input are the prediction of students in LTRB format
    and the prediction of teachers in LTRB format. Shape [N, 4]
    pred/target shape: [N, 4]
    """
    pred_l, pred_t, pred_r, pred_b = pred.unbind(dim=-1)
    tgt_l, tgt_t, tgt_r, tgt_b = target.unbind(dim=-1)

    # Area(Width = L+R, Height = T+B)
    pred_area = (pred_l + pred_r).clamp(min=0) * (pred_t + pred_b).clamp(min=0)
    target_area = (tgt_l + tgt_r).clamp(min=0) * (tgt_t + tgt_b).clamp(min=0)

    # Intersection 
    inter_w = (torch.min(pred_l, tgt_l) + torch.min(pred_r, tgt_r)).clamp(min=0)
    inter_h = (torch.min(pred_t, tgt_t) + torch.min(pred_b, tgt_b)).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union 
    union = pred_area + target_area - inter_area
    iou = inter_area / union.clamp(min=1e-7)

    # Enclosing box 
    enc_w = (torch.max(pred_l, tgt_l) + torch.max(pred_r, tgt_r)).clamp(min=0)
    enc_h = (torch.max(pred_t, tgt_t) + torch.max(pred_b, tgt_b)).clamp(min=0)
    enc_area = enc_w * enc_h
    
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-7)
    loss = 1.0 - giou

    if weight is not None:
        loss = loss * weight
        
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

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
    FCOSClassificationHead with HMfocalLoss.
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
        Override compute_loss to use HMfocalLoss with IoU-weighted target.

        Args:
            targets              : list of {"boxes": [N,4], "labels": [N]}
            head_outputs         : {"cls_logits": [B, HWA, num_classes]}
            matched_idxs         : list length B, each [HWA] matched gt index (-1 = bg)
            decoded_boxes_per_level: optional, decoded pred boxes per level per image
                                     shape list[list[Tensor[HW, 4]]]
                                     Nếu None → dùng one-hot target (fallback)
        """
        cls_logits = head_outputs["cls_logits"]  # [B, HWA, C]

        all_gt_classes_targets = []
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
    def __init__(
        self,
        backbone,
        num_classes: int,
        # transform params
        min_size: int = 800,
        max_size: int = 1333,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        # head params
        anchor_generator=None,
        head=None,
        # detection params
        center_sampling_radius: float = 1.5,
        score_thresh: float = 0.2,
        nms_thresh: float = 0.6,
        detections_per_img: int = 100,
        topk_candidates: int = 1000,
        # HMfocalLoss params
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
        Trả về các Tensor thô theo từng tầng FPN để đưa vào get_distill_loss.
        """
        # 1. Preprocess ảnh (đưa qua torchvision transform)
        # Chú ý: images truyền vào đây thường đã là Tensor [B, 3, H, W] từ FCOSDetector
        image_list = list(images.unbind(dim=0))
        transformed_images, _ = self.transform(image_list, None)

        # 2. Extract Features từ Backbone + FPN
        features = self.backbone(transformed_images.tensors)
        features_list = list(features.values())

        # 3. Chạy qua Head để lấy per-level outputs (logits, deltas, quality)
        logits_levels, deltas_levels, quality_levels = [], [], []

        for feature in features_list:
            cls_f, reg_f = feature, feature
            
            # Subnet Phân loại
            for layer in self.head.classification_head.conv:
                cls_f = layer(cls_f)
            logits_levels.append(self.head.classification_head.cls_logits(cls_f))

            # Subnet Hồi quy
            for layer in self.head.regression_head.conv:
                reg_f = layer(reg_f)
            deltas_levels.append(self.head.regression_head.bbox_reg(reg_f))
            quality_levels.append(self.head.regression_head.bbox_ctrness(reg_f))

        # 4. Giải mã tọa độ hộp (box_xyxy) cho từng ảnh phục vụ UHL Loss
        # torchvision ShiftGenerator trả về tọa độ trung tâm (anchors)
        anchors = self.anchor_generator(transformed_images, features_list)
        
        box_xyxy_per_image = []
        B = images.shape[0]
        
        # Flatten deltas để dễ decode
        all_deltas_flat = []
        for d in deltas_levels:
            all_deltas_flat.append(d.permute(0, 2, 3, 1).reshape(B, -1, 4))
        all_deltas = torch.cat(all_deltas_flat, dim=1) # [B, HWA_total, 4]

        for i in range(B):
            anch_i = anchors[i]  # [HWA, 4]
            delt_i = all_deltas[i]
            
            cx = (anch_i[:, 0] + anch_i[:, 2]) / 2.0
            cy = (anch_i[:, 1] + anch_i[:, 3]) / 2.0
            
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
        Override compute_loss to use HMfocalLoss with IoU-weighted target.
        """
        bbox_regression     = head_outputs["bbox_regression"]
        bbox_ctrness        = head_outputs["bbox_ctrness"]
        cls_logits          = head_outputs["cls_logits"]

        decoded_boxes_per_image = self._decode_boxes_for_iou(
            bbox_regression, anchors
        )

        # Classification loss (HMfocal với IoU-weighted target)
        loss_cls = self.head.classification_head.compute_loss(
            targets,
            head_outputs,
            matched_idxs,
            decoded_boxes_per_level=decoded_boxes_per_image,
        )

        # Regression + centerness losses (giữ nguyên torchvision logic)
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
        bbox_regression: Tensor,  # [HWA_total, 4]  — ltrb deltas
        anchors: List[Tensor],    # list[Tensor[HWA_i, 4]] per image
    ) -> List[List[Tensor]]:
        """
        Decode LTRB regression deltas to xyxy boxes, per image per level.
        """
        # Split regression per image
        splits = [len(a) for a in anchors]
        bbox_reg_per_img = bbox_regression.split(splits, dim=0)

        decoded_per_img = []
        for bbox_reg_i, anchors_i in zip(bbox_reg_per_img, anchors):
            # centers = (x1+x2)/2, (y1+y2)/2 = x1 = x2 (vì anchor là điểm)
            cx = (anchors_i[:, 0] + anchors_i[:, 2]) / 2.0
            cy = (anchors_i[:, 1] + anchors_i[:, 3]) / 2.0

            l = bbox_reg_i[:, 0]
            t = bbox_reg_i[:, 1]
            r = bbox_reg_i[:, 2]
            b = bbox_reg_i[:, 3]

            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + b

            pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [HWA, 4]
            # Flatten thành 1 tensor (không chia theo level ở đây vì
            # matched_idxs cũng được flatten)
            decoded_per_img.append([pred_boxes])  # single "level" flattened

        return decoded_per_img

    def _regression_centerness_loss(
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate regression loss (GIoU) và centerness loss (BCE).
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

            if not fg_mask.any():
                losses_bbox_reg.append(bbox_reg_i.sum() * 0.0)
                losses_bbox_ctrns.append(bbox_ctrn_i.sum() * 0.0)
                continue

            fg_idxs = matched_i[fg_mask]
            gt_boxes_fg = targets_i["boxes"][fg_idxs]   # [Nfg, 4]  xyxy

            # Decode predicted boxes (fg only)
            cx = (anchors_i[fg_mask, 0] + anchors_i[fg_mask, 2]) / 2.0
            cy = (anchors_i[fg_mask, 1] + anchors_i[fg_mask, 3]) / 2.0
            pred_reg_fg = bbox_reg_i[fg_mask]            # [Nfg, 4]  ltrb

            pred_boxes_fg = torch.stack([
                cx - pred_reg_fg[:, 0],
                cy - pred_reg_fg[:, 1],
                cx + pred_reg_fg[:, 2],
                cy + pred_reg_fg[:, 3],
            ], dim=-1)                                   # [Nfg, 4]  xyxy

            # GIoU loss
            reg_loss = generalized_box_iou_loss(
                pred_boxes_fg,
                gt_boxes_fg,
                reduction="sum",
            ) / max(1, fg_mask.sum().item())
            losses_bbox_reg.append(reg_loss)

            # Centerness target
            lt = torch.stack([
                cx - gt_boxes_fg[:, 0],
                cy - gt_boxes_fg[:, 1],
            ], dim=-1)
            rb = torch.stack([
                gt_boxes_fg[:, 2] - cx,
                gt_boxes_fg[:, 3] - cy,
            ], dim=-1)
            lt = lt.clamp(min=0.0)
            rb = rb.clamp(min=0.0)
            ctrness_target = torch.sqrt(
                (lt.min(dim=-1).values / lt.max(dim=-1).values.clamp(min=1e-6))
                * (rb.min(dim=-1).values / rb.max(dim=-1).values.clamp(min=1e-6))
            )                                            # [Nfg]

            pred_ctrness_fg = bbox_ctrn_i[fg_mask].squeeze(1)  # [Nfg]
            ctrness_loss = F.binary_cross_entropy_with_logits(
                pred_ctrness_fg,
                ctrness_target,
                reduction="sum",
            ) / max(1, fg_mask.sum().item())
            losses_bbox_ctrns.append(ctrness_loss)

        return sum(losses_bbox_reg), sum(losses_bbox_ctrns)


def build_custom_fcos(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    from_coco: bool = False,
    # HMfocalLoss params
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0,
    **fcos_kwargs,
) -> CustomFCOS:
    """
    Tạo CustomFCOS với HMfocalLoss.

    Args:
        num_classes              : số foreground classes (0-indexed)
        pretrained_backbone      : load ImageNet weights (chỉ dùng khi from_coco=False)
        trainable_backbone_layers: số FPN layers được unfreeze (0–5)
        min_size / max_size      : resize range cho GeneralizedRCNNTransform
        from_coco                : True → load COCO pretrained FCOS, thay head
        vfl_alpha/gamma          : HMfocalLoss hyperparams
        vfl_weight_type          : "iou" hoặc loại khác theo HMfocalLoss
        vfl_loss_weight          : weight của classification loss

    Returns:
        CustomFCOS instance
    """
    if from_coco:
        if not _HAS_WEIGHTS_API:
            raise RuntimeError(
                "FCOS_ResNet50_FPN_Weights requires torchvision >= 0.13."
            )
        # Load torchvision COCO pretrained (91 classes)
        base_model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
            **fcos_kwargs,
        )
    else:
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

    # Tạo CustomFCOS từ backbone + các thành phần của base_model
    # Cần trích xuất backbone (đã có FPN) từ base_model
    custom = CustomFCOS(
        backbone=base_model.backbone,
        num_classes=num_classes if not from_coco else 91,
        min_size=min_size,
        max_size=max_size,
        anchor_generator=base_model.anchor_generator,
        head=base_model.head,
        center_sampling_radius=base_model.head.classification_head.num_anchors,  # placeholder, see note
        vfl_alpha=vfl_alpha,
        vfl_gamma=vfl_gamma,
        vfl_weight_type=vfl_weight_type,
        vfl_loss_weight=vfl_loss_weight,
    )

    # Copy transform từ base model
    custom.transform = base_model.transform

    # Nếu from_coco và num_classes != 91: thay classification head cho đúng num_classes
    if from_coco and num_classes != 91:
        old_cls_head = custom.head.classification_head
        in_channels  = old_cls_head.conv[0][0].in_channels
        num_anchors  = old_cls_head.num_anchors
        custom.head.classification_head = HMFocalClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=nn.GroupNorm,
            vfl_alpha=vfl_alpha,
            vfl_gamma=vfl_gamma,
            vfl_weight_type=vfl_weight_type,
            vfl_loss_weight=vfl_loss_weight,
        )
        custom.num_classes = num_classes
    elif not from_coco:
        # head đã đúng num_classes nhưng vẫn cần thay bằng HMFocal head
        # (đã được thay trong __init__ của CustomFCOS)
        pass

    return custom


def build_custom_fcos_simple(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0,
) -> TV_FCOS:
    """
    Cách đơn giản hơn: khởi tạo torchvision FCOS chuẩn, sau đó
    monkey-patch compute_loss để dùng HMfocalLoss.

    Phù hợp khi muốn giữ nguyên kiến trúc torchvision hoàn toàn.

    Returns:
        torchvision FCOS với compute_loss được patch
    """
    backbone_weights = "DEFAULT" if pretrained_backbone else None
    model = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    # Tạo HMfocalLoss instance
    hm_loss = HMfocalLoss(
        use_sigmoid=True,
        alpha=vfl_alpha,
        gamma=vfl_gamma,
        weight_type=vfl_weight_type,
        loss_weight=vfl_loss_weight,
    )
    # Gắn vào model để compute_loss có thể truy cập
    model._hm_focal_loss = hm_loss
    model._num_classes   = num_classes

    # Patch compute_loss
    import types
    model.compute_loss = types.MethodType(_patched_compute_loss, model)

    return model


def _patched_compute_loss(
    self: TV_FCOS,
    targets: List[Dict[str, Tensor]],
    head_outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    matched_idxs: List[Tensor],
) -> Dict[str, Tensor]:
    """
    Monkey-patch cho torchvision FCOS.compute_loss.
    Thay classification loss bằng HMfocalLoss với IoU-weighted target.
    """
    cls_logits      = head_outputs["cls_logits"]      # [HWA_total, C]
    bbox_regression = head_outputs["bbox_regression"] # [HWA_total, 4]
    bbox_ctrness    = head_outputs["bbox_ctrness"]    # [HWA_total, 1]
    num_classes     = self._num_classes

    splits = [len(a) for a in anchors]
    cls_split   = cls_logits.split(splits, dim=0)
    bbox_split  = bbox_regression.split(splits, dim=0)
    ctrn_split  = bbox_ctrness.split(splits, dim=0)

    all_cls_targets  = []
    all_reg_losses   = []
    all_ctrn_losses  = []
    total_fg = 0

    for img_i, (targets_i, matched_i, cls_i, bbox_i, ctrn_i, anch_i) in enumerate(
        zip(targets, matched_idxs, cls_split, bbox_split, ctrn_split, anchors)
    ):
        gt_boxes_i  = targets_i["boxes"]   # [N_gt, 4]
        gt_labels_i = targets_i["labels"]  # [N_gt]
        fg_mask = matched_i >= 0
        num_fg  = fg_mask.sum().item()
        total_fg += num_fg

        # --- Classification target ---
        cls_target = torch.zeros(
            matched_i.shape[0], num_classes,
            dtype=cls_i.dtype, device=cls_i.device,
        )
        if num_fg > 0:
            fg_gt_idxs = matched_i[fg_mask]
            fg_labels  = gt_labels_i[fg_gt_idxs]

            # Decode pred boxes (fg)
            cx = (anch_i[fg_mask, 0] + anch_i[fg_mask, 2]) / 2.0
            cy = (anch_i[fg_mask, 1] + anch_i[fg_mask, 3]) / 2.0
            pred_reg_fg = bbox_i[fg_mask]
            pred_boxes_fg = torch.stack([
                cx - pred_reg_fg[:, 0], cy - pred_reg_fg[:, 1],
                cx + pred_reg_fg[:, 2], cy + pred_reg_fg[:, 3],
            ], dim=-1)
            gt_boxes_fg = gt_boxes_i[fg_gt_idxs]
            with torch.no_grad():
                iou_w = _box_iou_for_target(
                    pred_boxes_fg.detach(), gt_boxes_fg
                ).clamp(min=0.0)
            cls_target[fg_mask, fg_labels] = iou_w

        all_cls_targets.append(cls_target)

        # --- Regression loss ---
        if num_fg > 0:
            fg_gt_idxs = matched_i[fg_mask]
            gt_boxes_fg = gt_boxes_i[fg_gt_idxs]
            cx = (anch_i[fg_mask, 0] + anch_i[fg_mask, 2]) / 2.0
            cy = (anch_i[fg_mask, 1] + anch_i[fg_mask, 3]) / 2.0
            pred_reg_fg = bbox_i[fg_mask]
            pred_boxes_fg = torch.stack([
                cx - pred_reg_fg[:, 0], cy - pred_reg_fg[:, 1],
                cx + pred_reg_fg[:, 2], cy + pred_reg_fg[:, 3],
            ], dim=-1)
            reg_loss = generalized_box_iou_loss(
                pred_boxes_fg, gt_boxes_fg, reduction="sum"
            ) / max(1, num_fg)
        else:
            reg_loss = bbox_i.sum() * 0.0
        all_reg_losses.append(reg_loss)

        # --- Centerness loss ---
        if num_fg > 0:
            fg_gt_idxs = matched_i[fg_mask]
            gt_boxes_fg = gt_boxes_i[fg_gt_idxs]
            cx = (anch_i[fg_mask, 0] + anch_i[fg_mask, 2]) / 2.0
            cy = (anch_i[fg_mask, 1] + anch_i[fg_mask, 3]) / 2.0
            lt = torch.stack([cx - gt_boxes_fg[:, 0], cy - gt_boxes_fg[:, 1]], dim=-1).clamp(min=0)
            rb = torch.stack([gt_boxes_fg[:, 2] - cx, gt_boxes_fg[:, 3] - cy], dim=-1).clamp(min=0)
            ctrness_target = torch.sqrt(
                (lt.min(-1).values / lt.max(-1).values.clamp(min=1e-6))
                * (rb.min(-1).values / rb.max(-1).values.clamp(min=1e-6))
            )
            pred_ctrn_fg = ctrn_i[fg_mask].squeeze(1)
            ctrn_loss = F.binary_cross_entropy_with_logits(
                pred_ctrn_fg, ctrness_target, reduction="sum"
            ) / max(1, num_fg)
        else:
            ctrn_loss = ctrn_i.sum() * 0.0
        all_ctrn_losses.append(ctrn_loss)

    # Stack targets for HMfocalLoss
    cls_targets_all = torch.cat(all_cls_targets, dim=0)   # [HWA_total, C]
    num_fg_total = max(1, total_fg)

    loss_cls = self._hm_focal_loss(
        cls_logits,
        cls_targets_all,
        avg_factor=num_fg_total,
    )

    return {
        "classification": loss_cls,
        "bbox_regression": sum(all_reg_losses),
        "bbox_ctrness":    sum(all_ctrn_losses),
    }

def build_custom_fcos_trio(
    num_classes: int,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 600,
    max_size: int = 1000,
    from_coco: bool = False,
    use_simple_patch: bool = True,
    vfl_alpha: float = 0.75,
    vfl_gamma: float = 2.0,
    vfl_weight_type: str = "iou",
    vfl_loss_weight: float = 1.0,
) -> Tuple[TV_FCOS, TV_FCOS, TV_FCOS]:
    """
    Tạo (student, rgb_teacher, ir_teacher) với HMfocalLoss.

    Args:
        use_simple_patch : True → dùng monkey-patch (đơn giản, an toàn hơn)
                           False → dùng CustomFCOS subclass (cần kiểm tra kỹ)

    Returns:
        (student, rgb_teacher, ir_teacher)  — teachers là deep copy của student
    """
    if use_simple_patch:
        student = build_custom_fcos_simple(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            min_size=min_size,
            max_size=max_size,
            vfl_alpha=vfl_alpha,
            vfl_gamma=vfl_gamma,
            vfl_weight_type=vfl_weight_type,
            vfl_loss_weight=vfl_loss_weight,
        )
    else:
        student = build_custom_fcos(
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
        )

    rgb_teacher = copy.deepcopy(student)
    ir_teacher  = copy.deepcopy(student)

    return student, rgb_teacher, ir_teacher

class FCOSDistillAdapter:
    def __init__(self, student, rgb_teacher, ir_teacher, config):
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.config = config # Chứa alpha, beta, ratio...
        
        self.vfl_loss = HMfocalLoss(use_sigmoid=True)

    def distill_step(self, student_images, rgb_images, ir_images):
        # Lấy outputs từ Student (có grad)
        s_res = self.student.model.forward_for_distill(student_images)
        
        # Lấy từ Teachers (không grad)
        with torch.no_grad():
            r_res = self.rgb_teacher.model.forward_for_distill(rgb_images)
            # ir_images có thể cần xử lý 1-ch -> 3-ch nếu truyền bare tensor
            i_res = self.ir_teacher.model.forward_for_distill(ir_images)
            
        loss_dict = {}
        loss_dict.update(self._calculate_kd(s_res, r_res, name="_rgb"))
        loss_dict.update(self._calculate_kd(s_res, i_res, name="_ir"))
        return loss_dict

    def _calculate_kd(self, s_res, t_res, name=""):
        s_logits, s_deltas, s_quality, s_boxes = s_res
        t_logits, t_deltas, t_quality, t_boxes = t_res
        
        # Flatten per-level tensors
        def flatten(t_list, K):
            return torch.cat([t.permute(0, 2, 3, 1).reshape(-1, K) for t in t_list], dim=0)

        num_cls = s_logits[0].shape[1]
        sl_f, tl_f = flatten(s_logits, num_cls), flatten(t_logits, num_cls)
        sd_f, td_f = flatten(s_deltas, 4), flatten(t_deltas, 4)
        sq_f, tq_f = flatten(s_quality, 1), flatten(t_quality, 1)

        # 1. Uncertainty Region Selection (Harmony Measure)
        with torch.no_grad():
            t_probs = tl_f.sigmoid()
            cls_p, _ = t_probs.max(dim=1)
            iou_p = tq_f.sigmoid().squeeze()
            hm = (cls_p ** 1.0) * (iou_p ** 1.0) # alpha=1, beta=1
            
            count = int(tl_f.size(0) * 0.01) # ratio=0.01
            _, top_inds = torch.topk(hm, count)
            mask = torch.zeros_like(hm)
            mask[top_inds] = 1.0
            fg_num = mask.sum().clamp(min=1.0)

        # 2. Weights for distillation
        loss_weight = torch.exp(-(1 - hm.detach()) / 4.0)

        # 3. Logits Distill (QFLv2)
        l_logits = QFLv2(sl_f.sigmoid(), tl_f.sigmoid(), weight=loss_weight) / fg_num

        # 4. Deltas Distill (GIoU)
        l_deltas = (_giou_loss_ltrb(sd_f, td_f) * loss_weight).mean()

        # 5. Quality Distill (BCE)
        l_quality = F.binary_cross_entropy(sq_f.sigmoid(), tq_f.sigmoid(), weight=loss_weight.unsqueeze(1))

        # 6. UHL (Unified Harmony Learning) - Inter-box consistency
        # Tạm lược bớt logic UHL phức tạp để tối ưu VRAM, chỉ dùng logic distill cơ bản ở trên
        
        return {
            f"kd_logits{name}": l_logits,
            f"kd_deltas{name}": l_deltas,
            f"kd_quality{name}": l_quality
        }
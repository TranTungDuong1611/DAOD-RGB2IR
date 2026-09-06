"""Torchvision FCOS adapter for detector-neutral D3T training.

Only this module knows Torchvision's FCOS head layout, anchor assignment, box
coder, and native detection postprocessing.  The public output is decoded XYXY
boxes plus foreground/quality logits.
"""

from dataclasses import dataclass
from enum import Enum
import copy
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSHead

from .d3t_adapter import (
    AdapterOutput,
    DetectorAdapter,
    DistillationPair,
    Predictions,
    SupervisedBatch,
    Targets,
)


class ClassificationInitMode(str, Enum):
    """Which classification tower is retained from the COCO FCOS model."""

    COCO_TOWER = "coco_tower"
    RANDOM_HEAD = "random_head"


class _RegressionHeadWithoutCenterness(nn.Module):
    """Torchvision regression tower and predictor without the old ctrness head."""

    def __init__(self, conv: nn.Module, bbox_reg: nn.Module, num_anchors: int):
        super().__init__()
        self.conv = conv
        self.bbox_reg = bbox_reg
        self.num_anchors = num_anchors

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        outputs = []
        for feature in features:
            regression_feature = self.conv(feature)
            regression = F.relu(self.bbox_reg(regression_feature))
            batch, _, height, width = regression.shape
            regression = regression.view(
                batch, self.num_anchors, 4, height, width
            )
            regression = regression.permute(0, 3, 4, 1, 2).reshape(
                batch, -1, 4
            )
            outputs.append(regression)
        return torch.cat(outputs, dim=1)


def _new_quality_predictor(in_channels: int, num_anchors: int) -> nn.Conv2d:
    predictor = nn.Conv2d(
        in_channels, num_anchors, kernel_size=3, stride=1, padding=1
    )
    nn.init.normal_(predictor.weight, mean=0.0, std=0.01)
    nn.init.zeros_(predictor.bias)
    return predictor


def _flatten_level(output: Tensor, channels_per_anchor: int) -> Tensor:
    batch, _, height, width = output.shape
    output = output.view(
        batch, -1, channels_per_anchor, height, width
    )
    return output.permute(0, 3, 4, 1, 2).reshape(
        batch, -1, channels_per_anchor
    )


class FCOSIoUHead(nn.Module):
    """FCOS head with a three-class predictor and an IoU quality branch.

    The regression tower and bbox predictor are copied from ``base_head`` in
    both modes.  In ``COCO_TOWER`` the classification convolution tower is
    copied too; ``RANDOM_HEAD`` obtains a freshly initialized tower from a new
    Torchvision classification head.  Neither mode retains the COCO
    classification predictor or centerness predictor.
    """

    def __init__(
        self,
        base_head: FCOSHead,
        num_classes: int,
        classification_init_mode: ClassificationInitMode,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not isinstance(base_head, FCOSHead):
            raise TypeError("base_head must be a torchvision FCOSHead")
        mode = ClassificationInitMode(classification_init_mode)

        base_classification = base_head.classification_head
        base_regression = base_head.regression_head
        first_conv = next(
            (module for module in base_classification.conv if isinstance(module, nn.Conv2d)),
            None,
        )
        if first_conv is None:
            raise ValueError("FCOS classification head has no convolution")
        in_channels = first_conv.in_channels
        num_anchors = base_classification.num_anchors
        num_convs = sum(
            isinstance(module, nn.Conv2d)
            for module in base_classification.conv
        )

        if mode is ClassificationInitMode.COCO_TOWER:
            classification_head = copy.deepcopy(base_classification)
            classification_head.num_classes = num_classes
            classification_head.cls_logits = nn.Conv2d(
                in_channels,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            nn.init.normal_(classification_head.cls_logits.weight, std=0.01)
            prior_probability = 0.01
            nn.init.constant_(
                classification_head.cls_logits.bias,
                -torch.log(
                    torch.tensor((1.0 - prior_probability) / prior_probability)
                ).item(),
            )
        else:
            classification_head = FCOSClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
                num_convs=num_convs,
            )

        self.classification_head = classification_head
        self.regression_head = _RegressionHeadWithoutCenterness(
            conv=copy.deepcopy(base_regression.conv),
            bbox_reg=copy.deepcopy(base_regression.bbox_reg),
            num_anchors=num_anchors,
        )
        self.quality_predictor = _new_quality_predictor(
            in_channels, num_anchors
        )
        self.classification_init_mode = mode
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    @property
    def quality_logits(self) -> nn.Conv2d:
        """Alias retained for callers that name the output branch directly."""

        return self.quality_predictor

    @property
    def quality_head(self) -> nn.Conv2d:
        """Readable alias for the standalone IoU quality predictor."""

        return self.quality_predictor

    def forward(self, features: Sequence[Tensor]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("FCOSIoUHead requires at least one feature level")
        cls_outputs = []
        quality_outputs = []
        for feature in features:
            cls_feature = self.classification_head.conv(feature)
            cls_outputs.append(
                _flatten_level(
                    self.classification_head.cls_logits(cls_feature),
                    self.num_classes,
                )
            )
            quality_outputs.append(
                _flatten_level(
                    self.quality_predictor(cls_feature),
                    1,
                )
            )

        return {
            "cls_logits": torch.cat(cls_outputs, dim=1),
            "bbox_regression": self.regression_head(features),
            "quality_logits": torch.cat(quality_outputs, dim=1),
        }


@dataclass(frozen=True)
class FCOSAdapterContext:
    transformed_images: Any
    transformed_targets: Optional[Tuple[Mapping[str, Any], ...]]
    original_image_sizes: Tuple[Tuple[int, int], ...]
    features: Tuple[Tensor, ...]
    anchors_per_image: Tuple[Tensor, ...]
    anchors_per_level: Tuple[Tuple[Tensor, ...], ...]
    num_anchors_per_level: Tuple[int, ...]
    raw_head_outputs: Mapping[str, Tensor]
    sample_ids: Optional[Tuple[str, ...]] = None
    class_names: Tuple[str, ...] = ("person", "car", "bicycle")

    @property
    def transformed_image_sizes(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(tuple(size) for size in self.transformed_images.image_sizes)


def _aligned_box_iou(pred_boxes: Tensor, target_boxes: Tensor, eps: float = 1e-7) -> Tensor:
    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("aligned boxes must both have shape [N, 4]")
    top_left = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    bottom_right = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp_min(0)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp_min(0)
    union = (
        pred_wh[:, 0] * pred_wh[:, 1]
        + target_wh[:, 0] * target_wh[:, 1]
        - intersection_area
    )
    return intersection_area / union.clamp_min(eps)


class TorchvisionFCOSAdapter(DetectorAdapter):
    """Run Torchvision FCOS once and expose D3T-compatible tensors."""

    def __init__(
        self,
        detector: nn.Module,
        class_names: Sequence[str] = ("person", "car", "bicycle"),
    ) -> None:
        super().__init__()
        required = (
            "transform",
            "backbone",
            "anchor_generator",
            "box_coder",
            "postprocess_detections",
        )
        missing = [name for name in required if not hasattr(detector, name)]
        if missing:
            raise TypeError(f"FCOS detector is missing required members: {missing}")
        if not isinstance(getattr(detector, "head", None), FCOSIoUHead):
            raise TypeError("detector.head must be an FCOSIoUHead")
        names = tuple(class_names)
        if len(names) != detector.head.num_classes:
            raise ValueError("class_names must match the FCOS predictor class count")
        if len(set(names)) != len(names):
            raise ValueError("class_names must be unique")

        self.detector = detector
        self.class_names = names
        self.num_classes = len(names)

    @staticmethod
    def _normalize_images(images: Sequence[Tensor] | Tensor) -> Tuple[Tensor, ...]:
        if isinstance(images, Tensor):
            if images.ndim != 4:
                raise ValueError("batched images must have shape [B, C, H, W]")
            images = tuple(images.unbind(dim=0))
        else:
            images = tuple(images)
        if not images:
            raise ValueError("images must contain at least one image")
        if any(not isinstance(image, Tensor) or image.ndim != 3 for image in images):
            raise ValueError("each image must have shape [C, H, W]")
        return images

    def _resolve_sample_ids(
        self,
        sample_ids: Optional[Sequence[str]],
        targets: Optional[Targets],
        batch_size: int,
    ) -> Optional[Tuple[str, ...]]:
        if sample_ids is None and targets is not None:
            stems = tuple(target.get("stem") for target in targets)
            if all(isinstance(stem, str) and stem.strip() for stem in stems):
                sample_ids = stems
        return self.validate_sample_ids(sample_ids, batch_size)

    def forward(
        self,
        images: Sequence[Tensor] | Tensor,
        targets: Optional[Targets] = None,
        sample_ids: Optional[Tuple[str, ...]] = None,
    ) -> AdapterOutput:
        image_list = self._normalize_images(images)
        target_list = None if targets is None else list(targets)
        if target_list is not None and len(target_list) != len(image_list):
            raise ValueError("targets must match the image batch length")
        resolved_ids = self._resolve_sample_ids(
            sample_ids, target_list, len(image_list)
        )

        original_sizes = tuple(
            (int(image.shape[-2]), int(image.shape[-1])) for image in image_list
        )
        transformed_images, transformed_targets = self.detector.transform(
            list(image_list), target_list
        )
        features = self.detector.backbone(transformed_images.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([("0", features)])
        feature_list = tuple(features.values())
        head_outputs = self.detector.head(feature_list)
        expected_keys = {"cls_logits", "bbox_regression", "quality_logits"}
        if set(head_outputs) != expected_keys:
            raise ValueError(
                f"FCOSIoUHead output keys must be {sorted(expected_keys)}"
            )

        anchors_list = tuple(
            self.detector.anchor_generator(transformed_images, list(feature_list))
        )
        anchors_per_location = tuple(
            self.detector.anchor_generator.num_anchors_per_location()
        )
        if len(anchors_per_location) != len(feature_list):
            raise ValueError("anchor generator levels do not match FPN levels")
        if any(value != 1 for value in anchors_per_location):
            raise ValueError("FCOS D3T adapter requires one anchor per location")
        num_anchors_per_level = tuple(
            int(feature.shape[-2] * feature.shape[-1] * anchors_per_location[level])
            for level, feature in enumerate(feature_list)
        )
        total_anchors = sum(num_anchors_per_level)
        for image_anchors in anchors_list:
            if image_anchors.shape != (total_anchors, 4):
                raise ValueError("anchor/head layout mismatch in FCOS adapter")

        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        quality_logits = head_outputs["quality_logits"]
        if cls_logits.shape[1] != total_anchors:
            raise ValueError("classification/head anchor row count mismatch")
        if bbox_regression.shape[:2] != cls_logits.shape[:2]:
            raise ValueError("regression/head anchor row count mismatch")
        if quality_logits.shape[:2] != cls_logits.shape[:2]:
            raise ValueError("quality/head anchor row count mismatch")

        predictions = []
        anchors_per_level = []
        for image_index, anchors in enumerate(anchors_list):
            levels = tuple(anchors.split(num_anchors_per_level, dim=0))
            anchors_per_level.append(levels)
            decoded_boxes = self.detector.box_coder.decode(
                bbox_regression[image_index], anchors
            )
            predictions.append(
                Predictions(
                    class_logits=cls_logits[image_index],
                    boxes=decoded_boxes,
                    quality_logits=quality_logits[image_index].squeeze(-1),
                )
            )

        context = FCOSAdapterContext(
            transformed_images=transformed_images,
            transformed_targets=(
                None
                if transformed_targets is None
                else tuple(transformed_targets)
            ),
            original_image_sizes=original_sizes,
            features=feature_list,
            anchors_per_image=anchors_list,
            anchors_per_level=tuple(anchors_per_level),
            num_anchors_per_level=num_anchors_per_level,
            raw_head_outputs=head_outputs,
            sample_ids=resolved_ids,
            class_names=self.class_names,
        )
        return AdapterOutput(
            tuple(predictions), context=context, sample_ids=resolved_ids
        )

    def _match_anchors_to_targets(
        self,
        anchors: Tensor,
        target: Mapping[str, Any],
        num_anchors_per_level: Sequence[int],
    ) -> Tensor:
        """Port Torchvision 0.24.1 FCOS assignment into an isolated method."""

        if anchors.ndim != 2 or anchors.shape[-1] != 4:
            raise ValueError("anchors must have shape [N, 4]")
        if sum(num_anchors_per_level) != anchors.shape[0]:
            raise ValueError("anchor levels do not cover all anchors")
        gt_boxes = target["boxes"].to(device=anchors.device, dtype=anchors.dtype)
        if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
            raise ValueError("target boxes must have shape [M, 4]")
        if gt_boxes.numel() == 0:
            return torch.full(
                (anchors.shape[0],),
                -1,
                dtype=torch.int64,
                device=anchors.device,
            )

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchor_sizes = anchors[:, 2] - anchors[:, 0]
        radius = float(self.detector.center_sampling_radius)
        center_match = (
            anchor_centers[:, None, :] - gt_centers[None, :, :]
        ).abs().amax(dim=2) < radius * anchor_sizes[:, None]

        x, y = anchor_centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes.unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack(
            [x - x0, y - y0, x1 - x, y1 - y], dim=2
        )
        inside_match = pairwise_dist.min(dim=2).values > 0

        lower_bound = anchor_sizes * 4
        lower_bound = lower_bound.clone()
        lower_bound[: num_anchors_per_level[0]] = 0
        upper_bound = anchor_sizes * 8
        upper_bound = upper_bound.clone()
        upper_bound[-num_anchors_per_level[-1] :] = float("inf")
        max_regression = pairwise_dist.max(dim=2).values
        level_match = (
            (max_regression > lower_bound[:, None])
            & (max_regression < upper_bound[:, None])
        )

        candidate = center_match & inside_match & level_match
        gt_areas = (
            (gt_boxes[:, 2] - gt_boxes[:, 0])
            * (gt_boxes[:, 3] - gt_boxes[:, 1])
        )
        scores = candidate.to(torch.float32) * (1e8 - gt_areas[None, :])
        max_values, matched = scores.max(dim=1)
        matched[max_values < 1e-5] = -1
        return matched.to(torch.int64)

    def prepare_supervised(
        self, output: AdapterOutput, targets: Targets
    ) -> SupervisedBatch:
        if not isinstance(output.context, FCOSAdapterContext):
            raise ValueError("FCOS supervised preparation requires FCOS context")
        context = output.context
        if context.transformed_targets is None:
            raise ValueError(
                "raw output was created without targets; run raw(images, targets)"
            )
        if len(targets) != len(output.predictions):
            raise ValueError("targets must match the output batch length")

        prepared_targets = []
        for prediction, transformed_target, anchors in zip(
            output.predictions,
            context.transformed_targets,
            context.anchors_per_image,
        ):
            matched = self._match_anchors_to_targets(
                anchors,
                transformed_target,
                context.num_anchors_per_level,
            )
            foreground = matched >= 0
            class_targets = torch.zeros(
                prediction.class_logits.shape,
                dtype=prediction.class_logits.dtype,
                device=prediction.class_logits.device,
            )
            box_targets = torch.zeros_like(prediction.boxes)
            quality_targets = torch.zeros_like(prediction.quality_logits)
            if foreground.any():
                matched_fg = matched[foreground]
                gt_boxes = transformed_target["boxes"].to(
                    device=prediction.boxes.device,
                    dtype=prediction.boxes.dtype,
                )
                gt_labels = transformed_target["labels"].to(
                    device=prediction.class_logits.device,
                    dtype=torch.long,
                )
                labels_fg = gt_labels[matched_fg]
                if (
                    labels_fg.numel()
                    and (
                        labels_fg.min() < 0
                        or labels_fg.max() >= self.num_classes
                    )
                ):
                    raise ValueError("FCOS target labels must be zero-based")
                matched_boxes = gt_boxes[matched_fg]
                box_targets[foreground] = matched_boxes.detach()
                with torch.no_grad():
                    iou = _aligned_box_iou(
                        prediction.boxes[foreground].detach(),
                        matched_boxes,
                    ).clamp(0.0, 1.0)
                quality_targets[foreground] = iou.detach()
                class_targets[foreground, labels_fg] = iou.detach().to(
                    class_targets.dtype
                )

            prepared_targets.append(
                {
                    "class_targets": class_targets.detach(),
                    "box_targets": box_targets.detach(),
                    "quality_targets": quality_targets.detach(),
                    "foreground": foreground.detach(),
                    "matched_idxs": matched.detach(),
                }
            )
        return SupervisedBatch(
            predictions=output.predictions,
            targets=tuple(prepared_targets),
            native_losses={},
        )

    def prepare_distillation(
        self, student: AdapterOutput, teacher: AdapterOutput
    ) -> DistillationPair:
        if not isinstance(student.context, FCOSAdapterContext) or not isinstance(
            teacher.context, FCOSAdapterContext
        ):
            raise ValueError("FCOS distillation requires FCOS adapter contexts")
        s_context = student.context
        t_context = teacher.context
        if s_context.sample_ids is None or t_context.sample_ids is None:
            raise ValueError("sample_ids are required for position-wise FCOS KD")
        if s_context.sample_ids != t_context.sample_ids:
            raise ValueError("student and teacher sample IDs do not correspond")
        if s_context.transformed_image_sizes != t_context.transformed_image_sizes:
            raise ValueError("student and teacher transformed image sizes differ")
        if s_context.num_anchors_per_level != t_context.num_anchors_per_level:
            raise ValueError("student and teacher FPN anchor counts differ")
        if s_context.class_names != t_context.class_names:
            raise ValueError("student and teacher class order differs")
        if len(s_context.anchors_per_image) != len(t_context.anchors_per_image):
            raise ValueError("student and teacher image counts differ")
        for s_anchors, t_anchors in zip(
            s_context.anchors_per_image, t_context.anchors_per_image
        ):
            if s_anchors.shape != t_anchors.shape or not torch.equal(
                s_anchors, t_anchors
            ):
                raise ValueError("student and teacher anchor geometry differs")
        if len(student.predictions) != len(teacher.predictions):
            raise ValueError("student and teacher image counts differ")
        for s_prediction, t_prediction in zip(
            student.predictions, teacher.predictions
        ):
            if s_prediction.class_logits.shape != t_prediction.class_logits.shape:
                raise ValueError("student and teacher class layouts differ")
            if s_prediction.boxes.shape != t_prediction.boxes.shape:
                raise ValueError("student and teacher decoded layouts differ")
        return DistillationPair(
            tuple(student.predictions), tuple(teacher.predictions)
        )

    def postprocess(
        self, output: AdapterOutput
    ) -> Sequence[Mapping[str, Tensor]]:
        if not isinstance(output.context, FCOSAdapterContext):
            raise ValueError("FCOS postprocessing requires FCOS context")
        context = output.context
        per_level_outputs = {
            key: list(value.split(context.num_anchors_per_level, dim=1))
            for key, value in context.raw_head_outputs.items()
        }
        native_outputs = self.detector.postprocess_detections(
            {
                "cls_logits": per_level_outputs["cls_logits"],
                "bbox_regression": per_level_outputs["bbox_regression"],
                # Torchvision's native postprocessor uses this third tensor in
                # sqrt(sigmoid(class) * sigmoid(quality)).
                "bbox_ctrness": per_level_outputs["quality_logits"],
            },
            [list(levels) for levels in context.anchors_per_level],
            list(context.transformed_images.image_sizes),
        )
        return self.detector.transform.postprocess(
            native_outputs,
            list(context.transformed_images.image_sizes),
            list(context.original_image_sizes),
        )


__all__ = [
    "ClassificationInitMode",
    "FCOSAdapterContext",
    "FCOSIoUHead",
    "TorchvisionFCOSAdapter",
]

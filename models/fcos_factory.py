"""Factories for the Torchvision FCOS D3T student/teacher models."""

from dataclasses import fields, is_dataclass
from enum import Enum
import copy
from typing import Any, Callable, Optional, Tuple

from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
)

from config import FCOSModelConfig, TrainingConfig
from loss.d3t_criterion import D3TLossCriterion
from .d3t_wrapper import D3TWrapper
from .torchvision_fcos_adapter import (
    ClassificationInitMode,
    FCOSIoUHead,
    TorchvisionFCOSAdapter,
)


CHECKPOINT_SCHEMA_VERSION = 1


def _resolve_fcos_weights(value):
    if value is None:
        return None
    if isinstance(value, FCOS_ResNet50_FPN_Weights):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"DEFAULT", "COCO_V1", "FCOS_RESNET50_FPN_COCO_V1"}:
            return FCOS_ResNet50_FPN_Weights.DEFAULT
        if normalized in {"NONE", ""}:
            return None
    raise ValueError(
        "weights must be None, 'DEFAULT', or a supported FCOS weight identifier"
    )


def _serialize(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def serialize_effective_config(config: TrainingConfig) -> dict[str, Any]:
    return _serialize(config)


def build_checkpoint_metadata(config: TrainingConfig) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "num_classes": config.model.num_classes,
        "class_names": list(config.model.class_names),
        "classification_init_mode": config.model.classification_init_mode.value,
        "teacher_mode": config.teacher_mode,
        "config": serialize_effective_config(config),
    }


def validate_checkpoint_metadata(
    metadata: dict[str, Any], config: TrainingConfig
) -> None:
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema; expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {metadata.get('schema_version')}"
        )
    if metadata.get("num_classes") != config.model.num_classes:
        raise ValueError("Checkpoint class count does not match the effective config")
    if tuple(metadata.get("class_names", ())) != tuple(config.model.class_names):
        raise ValueError("Checkpoint class order does not match the effective config")
    if metadata.get("classification_init_mode") != config.model.classification_init_mode.value:
        raise ValueError(
            "Checkpoint classification initialization mode does not match the effective config"
        )
    if metadata.get("teacher_mode") != config.teacher_mode:
        raise ValueError("Checkpoint teacher mode does not match the effective config")


def _make_base_detector(
    config: TrainingConfig,
    detector_builder: Callable[..., nn.Module],
) -> nn.Module:
    model_config: FCOSModelConfig = config.model
    weights = _resolve_fcos_weights(model_config.weights)
    if weights is not None:
        weights_backbone = None
        base_num_classes = len(weights.meta.get("categories", ())) or 91
    else:
        weights_backbone = (
            ResNet50_Weights.DEFAULT if model_config.pretrained_backbone else None
        )
        base_num_classes = 91
    return detector_builder(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=base_num_classes,
        trainable_backbone_layers=model_config.trainable_backbone_layers,
        min_size=model_config.min_size,
        max_size=model_config.max_size,
        center_sampling_radius=model_config.center_sampling_radius,
        score_thresh=model_config.score_thresh,
        nms_thresh=model_config.nms_thresh,
        topk_candidates=model_config.topk_candidates,
        detections_per_img=model_config.detections_per_img,
    )


def build_fcos_d3t_model(
    config: TrainingConfig,
    *,
    base_detector: Optional[nn.Module] = None,
    detector_builder: Optional[Callable[..., nn.Module]] = None,
) -> D3TWrapper:
    """Build one student wrapper from the effective FCOS configuration."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if base_detector is None:
        base_detector = _make_base_detector(
            config, detector_builder or fcos_resnet50_fpn
        )
    if not hasattr(base_detector, "head"):
        raise TypeError("base_detector must expose a Torchvision FCOS head")
    base_detector.head = FCOSIoUHead(
        base_detector.head,
        num_classes=config.model.num_classes,
        classification_init_mode=config.model.classification_init_mode,
    )
    # The injected sentinel path is used by tests and still receives the same
    # effective transform/detection settings as a freshly constructed model.
    transform = getattr(base_detector, "transform", None)
    if transform is not None:
        transform.min_size = [config.model.min_size]
        transform.max_size = config.model.max_size
    for name in (
        "center_sampling_radius",
        "score_thresh",
        "nms_thresh",
        "topk_candidates",
        "detections_per_img",
    ):
        if hasattr(base_detector, name):
            setattr(base_detector, name, getattr(config.model, name))

    adapter = TorchvisionFCOSAdapter(
        base_detector,
        class_names=config.model.class_names,
    )
    criterion = D3TLossCriterion(
        alpha=config.model.vfl_alpha,
        gamma=config.model.vfl_gamma,
        weight_type=config.model.vfl_weight_type,
    )
    wrapper = D3TWrapper(adapter, criterion)
    wrapper.effective_config = config
    wrapper.classification_init_mode = config.model.classification_init_mode
    return wrapper


def build_fcos_d3t_trio(
    config: TrainingConfig,
    *,
    base_detector: Optional[nn.Module] = None,
    detector_builder: Optional[Callable[..., nn.Module]] = None,
) -> Tuple[D3TWrapper, Optional[D3TWrapper], Optional[D3TWrapper]]:
    """Build student plus independent RGB/IR teachers."""

    student = build_fcos_d3t_model(
        config,
        base_detector=base_detector,
        detector_builder=detector_builder,
    )
    rgb_teacher = (
        copy.deepcopy(student)
        if config.teacher_mode in {"rgb", "two_teacher"}
        else None
    )
    ir_teacher = (
        copy.deepcopy(student)
        if config.teacher_mode in {"ir", "two_teacher"}
        else None
    )
    for teacher in (rgb_teacher, ir_teacher):
        if teacher is None:
            continue
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    return student, rgb_teacher, ir_teacher


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "ClassificationInitMode",
    "build_checkpoint_metadata",
    "build_fcos_d3t_model",
    "build_fcos_d3t_trio",
    "serialize_effective_config",
    "validate_checkpoint_metadata",
]

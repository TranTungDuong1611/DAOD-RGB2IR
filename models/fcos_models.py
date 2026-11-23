"""Build FCOS models and manage their teacher forward and EMA operations."""

import copy
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
)

from config import FCOSModelConfig, TrainingConfig
from loss.detection_criterion import DetectionLossCriterion
from .detector_wrapper import DetectorWrapper
from .fcos_adapter import (
    ClassificationInitMode,
    FCOSIoUHead,
    TorchvisionFCOSAdapter,
)


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
        min_size=model_config.min_sizes,
        max_size=model_config.max_size,
        center_sampling_radius=model_config.center_sampling_radius,
        score_thresh=model_config.score_thresh,
        nms_thresh=model_config.nms_thresh,
        topk_candidates=model_config.topk_candidates,
        detections_per_img=model_config.detections_per_img,
    )


def build_fcos_model(
    config: TrainingConfig,
    *,
    base_detector: Optional[nn.Module] = None,
    detector_builder: Optional[Callable[..., nn.Module]] = None,
) -> DetectorWrapper:
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
        transform.min_size = config.model.min_sizes
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
    criterion = DetectionLossCriterion(
        alpha=config.model.vfl_alpha,
        gamma=config.model.vfl_gamma,
        weight_type=config.model.vfl_weight_type,
    )
    wrapper = DetectorWrapper(adapter, criterion)
    wrapper.effective_config = config
    wrapper.classification_init_mode = config.model.classification_init_mode
    return wrapper


def build_fcos_trio(
    config: TrainingConfig,
    *,
    base_detector: Optional[nn.Module] = None,
    detector_builder: Optional[Callable[..., nn.Module]] = None,
) -> Tuple[DetectorWrapper, Optional[DetectorWrapper], Optional[DetectorWrapper]]:
    """Build student plus independent RGB/IR teachers."""

    student = build_fcos_model(
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


def initialize_teachers(student, teachers, *, names):
    """Hard-copy selected teachers when the caller reaches the EMA start step."""
    for name in names:
        teacher = teachers[name]
        if teacher is None:
            raise ValueError(f"EMA start requires the '{name}' teacher")
        copy_student_to_teacher(teacher, student)


def get_teacher(teachers, name):
    """Resolve a configured teacher and restore inference mode before use."""
    teacher = teachers.get(name)
    if teacher is None:
        raise ValueError(f"teacher '{name}' is not configured")
    teacher.eval()
    return teacher


def forward_teacher(teacher, images, *, sample_ids, resize_short_edge):
    """Reuse student-selected geometry and keep teacher forward outside autograd."""
    with torch.no_grad():
        kwargs = {"sample_ids": sample_ids}
        if resize_short_edge is not None:
            kwargs["resize_short_edge"] = resize_short_edge
        return teacher.raw(images, **kwargs)


def copy_student_to_teacher(teacher: nn.Module, student: nn.Module) -> None:
    """Hard-copy student parameters and buffers into a teacher."""
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.copy_(s_p.data)
        for t_b, s_b in zip(teacher.buffers(), student.buffers()):
            t_b.data.copy_(s_b.data)


def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    alpha: float = 0.9996,
    global_step: Optional[int] = None,
) -> None:
    """Update teacher parameters and buffers with the existing warmup decay."""
    if global_step is not None:
        warmup_alpha = (1.0 + global_step) / (10.0 + global_step)
        alpha = min(alpha, warmup_alpha)

    with torch.no_grad():
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())
        for name, t_p in t_params.items():
            if name in s_params:
                t_p.data.mul_(alpha).add_(s_params[name].data, alpha=1.0 - alpha)

        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())
        for name, t_b in t_bufs.items():
            if name not in s_bufs:
                continue
            s_b = s_bufs[name]
            if t_b.is_floating_point():
                t_b.data.mul_(alpha).add_(s_b.data, alpha=1.0 - alpha)
            else:
                # Integral counters follow the student without interpolation.
                t_b.data.copy_(s_b.data)


__all__ = [
    "ClassificationInitMode",
    "build_fcos_model",
    "build_fcos_trio",
    "initialize_teachers",
    "get_teacher",
    "forward_teacher",
    "copy_student_to_teacher",
    "ema_update",
]

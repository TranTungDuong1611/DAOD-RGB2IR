"""FCOS construction, transforms and detection settings."""

from dataclasses import dataclass
from typing import Optional, Tuple

from .types import ClassificationInitMode


@dataclass
class FCOSModelConfig:
    """Effective Torchvision FCOS model settings."""
    num_classes: int = 3
    class_names: Tuple[str, ...] = ("person", "car", "bicycle")
    weights: Optional[str] = "DEFAULT"
    classification_init_mode: ClassificationInitMode = ClassificationInitMode.COCO_TOWER
    pretrained_backbone: Optional[bool] = None
    trainable_backbone_layers: int = 3
    min_sizes: Tuple[int, ...] = (640, 672, 704, 736, 768, 800)
    max_size: int = 1333
    center_sampling_radius: float = 0.0
    score_thresh: float = 0.05
    nms_thresh: float = 0.6
    topk_candidates: int = 1000
    detections_per_img: int = 100

    # HM-Focal Loss (VFL) Hyperparameters
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
    vfl_weight_type: str = "iou"

    def __post_init__(self):
        if isinstance(self.weights, str) and self.weights.strip().lower() in {"none", ""}:
            self.weights = None
        if self.pretrained_backbone is None:
            self.pretrained_backbone = self.weights is not None
        self.class_names = tuple(self.class_names)
        self.min_sizes = tuple(self.min_sizes)
        self.classification_init_mode = ClassificationInitMode(
            self.classification_init_mode
        )
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match class_names")
        if self.class_names != ("person", "car", "bicycle"):
            raise ValueError("class_names must be ('person', 'car', 'bicycle')")
        if not 0 <= self.trainable_backbone_layers <= 5:
            raise ValueError("trainable_backbone_layers must be in [0, 5]")
        if not self.min_sizes or any(size <= 0 for size in self.min_sizes):
            raise ValueError("min_sizes must contain positive values")
        if self.min_sizes != tuple(sorted(self.min_sizes)):
            raise ValueError("min_sizes must be increasing")
        if self.max_size < max(self.min_sizes):
            raise ValueError("max_size must be at least the largest min_size")
        if self.center_sampling_radius < 0:
            raise ValueError("center_sampling_radius must be non-negative")
        if not 0 <= self.score_thresh <= 1 or not 0 <= self.nms_thresh <= 1:
            raise ValueError("score_thresh and nms_thresh must be in [0, 1]")
        if self.topk_candidates <= 0 or self.detections_per_img <= 0:
            raise ValueError("detection top-k values must be positive")


__all__ = ['FCOSModelConfig']

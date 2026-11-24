"""Shared enums for configuration and detector implementations."""

from enum import Enum


class Phase(Enum):
    """Enumeration of Curriculum Learning stages."""
    PHASE1_RGB_WARMUP = 1
    PHASE2_TRANSITION = 2  # Đảm bảo tên này tồn tại
    PHASE3_ADAPTATION = 3  # Đảm bảo tên này tồn tại
    PHASE4_IR_FOCUS   = 4


class ClassificationInitMode(str, Enum):
    """Which classification tower is retained from the COCO FCOS model."""

    COCO_TOWER = "coco_tower"
    RANDOM_HEAD = "random_head"


__all__ = ['Phase', 'ClassificationInitMode']

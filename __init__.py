"""
Block Domain Adaptation: RGB -> MID(GAN) -> IR, dense distillation (D3T-style).
"""

from batch_types import IRBatch, RGBBatch
from config import (
    BlockSchedulerConfig,
    DisjointSplitConfig,
    DistillConfig,
    EMAConfig,
    IRAugConfig,
    LambdaConfig,
    RGBAugConfig,
    TeacherWeightConfig,
    TrainingConfig,
)
from ema import copy_student_to_teacher, ema_update
from losses import distill, dense_forward, get_distill_loss
from scheduler import BlockDomainScheduler, DomainStep, parse_ratio_schedule
from trainer import BlockDomainAdaptationTrainer
from fcos_wrapper import FCOSDetector, build_fcos_trio
from evaluator import DetectionEvaluator, PhaseEvaluator
from visualize import visualize_eval_samples

__all__ = [
    # Trainer (main entry point)
    "BlockDomainAdaptationTrainer",
    # Config
    "TrainingConfig",
    "EMAConfig",
    "RGBAugConfig",
    "IRAugConfig",
    "BlockSchedulerConfig",
    "DistillConfig",
    "TeacherWeightConfig",
    "DisjointSplitConfig",
    "LambdaConfig",
    # Scheduler
    "BlockDomainScheduler",
    "parse_ratio_schedule",
    "DomainStep",
    # EMA
    "ema_update",
    "copy_student_to_teacher",
    # Losses (dense distillation)
    "dense_forward",
    "get_distill_loss",
    "distill",
    # Batch types
    "RGBBatch",
    "IRBatch",
    # FCOS
    "FCOSDetector",
    "build_fcos_trio",
    # Evaluation
    "DetectionEvaluator",
    "PhaseEvaluator",
    # Visualization
    "visualize_eval_samples",
]

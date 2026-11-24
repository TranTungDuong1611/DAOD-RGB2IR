"""Optimizer, monitoring and effective top-level training configuration."""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

from .curriculum import CurriculumConfig, MidRoutingConfig
from .data import AugConfig, DataConfig, DataLoaderConfig, SoftSAGAConfig
from .distillation import DistillConfig, EMAConfig, LossConfig
from .gan_bridge import BridgeScheduleConfig, GANBridgeConfig
from .model import FCOSModelConfig
from .types import Phase


@dataclass
class OptimizerConfig:
    """SGD and iteration-based LR schedule derived from the D3T solver."""

    base_lr: Optional[float] = None
    reference_lr: float = 0.016
    reference_batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: Optional[int] = None
    warmup_max_iters: int = 1000
    warmup_factor: float = 1.0 / 1000.0
    milestones: Tuple[int, ...] = (70000,)
    gamma: float = 0.1

    def __post_init__(self):
        self.milestones = tuple(self.milestones)
        positive_values = (self.reference_lr, self.momentum, self.gamma)
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("reference_lr, momentum, and gamma must be positive")
        if self.base_lr is not None and (
            not math.isfinite(self.base_lr) or self.base_lr <= 0
        ):
            raise ValueError("base_lr must be None or positive")
        if self.reference_batch_size <= 0:
            raise ValueError("reference_batch_size must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_iters is not None and self.warmup_iters < 0:
            raise ValueError("warmup_iters must be None or non-negative")
        if not math.isfinite(self.warmup_factor) or not 0 < self.warmup_factor <= 1:
            raise ValueError("warmup_factor must be in (0, 1]")
        if any(step < 0 for step in self.milestones):
            raise ValueError("LR milestones must be non-negative")
        if self.milestones != tuple(sorted(self.milestones)):
            raise ValueError("LR milestones must be increasing")


@dataclass
class PseudoLabelEvalConfig:
    """Audit cadence, fixed population, and scoring; these settings never control KD."""

    enabled: bool = False
    interval: int = 500
    iou_threshold: float = 0.5
    sample_size: int = 256
    subset_seed: int = 0

    def __post_init__(self):
        if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval <= 0:
            raise ValueError("pseudo-label evaluation interval must be a positive integer")
        if not math.isfinite(self.iou_threshold) or not 0. <= self.iou_threshold <= 1.:
            raise ValueError("pseudo-label evaluation IoU threshold must be in [0, 1]")
        if isinstance(self.sample_size, bool) or not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError("pseudo-label evaluation sample size must be a positive integer")
        if isinstance(self.subset_seed, bool) or not isinstance(self.subset_seed, int) or self.subset_seed < 0:
            raise ValueError("pseudo-label evaluation subset seed must be a non-negative integer")


@dataclass
class TrainingConfig:
    """Master config for CurriculumDomainAdaptationTrainer."""
    model: FCOSModelConfig = field(default_factory=FCOSModelConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    ema: EMAConfig = field(default_factory=EMAConfig)
    soft_saga: SoftSAGAConfig = field(default_factory=SoftSAGAConfig)
    mid_routing: MidRoutingConfig = field(default_factory=MidRoutingConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    bridge_schedule: BridgeScheduleConfig = field(default_factory=BridgeScheduleConfig)
    gan_bridge: GANBridgeConfig = field(default_factory=GANBridgeConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    data: DataConfig = field(default_factory=DataConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    pseudo_label_eval: PseudoLabelEvalConfig = field(default_factory=PseudoLabelEvalConfig)

    total_iters: int = 10000
    grad_clip: float = 10.0
    device: str = "cuda"
    log_interval: int = 50
    output_dir: str = "./output_flir"
    workflow: str = "curriculum"
    teacher_mode: str = "two_teacher"
    wandb: bool = False
    eval_period: int = 500
    eval_iou_thresholds: Tuple[float, ...] = (0.5,)
    eval_interp: str = "voc11"
    checkpoint_period: int = 5000

    def __post_init__(self):
        if self.total_iters <= 0:
            raise ValueError("total_iters must be positive")
        self.curriculum = self.curriculum.resolved(self.total_iters)
        if self.grad_clip < 0 or min(self.log_interval, self.eval_period, self.checkpoint_period) <= 0:
            raise ValueError("training intervals and grad_clip are invalid")
        if self.workflow not in {"curriculum", "rgb_baseline", "gan_bridge"}:
            raise ValueError("workflow must be 'curriculum', 'rgb_baseline' or 'gan_bridge'")
        if self.teacher_mode not in {"rgb", "ir", "two_teacher"}:
            raise ValueError("teacher_mode must be 'rgb', 'ir', or 'two_teacher'")
        if self.optimizer.base_lr is None:
            self.optimizer.base_lr = (
                self.optimizer.reference_lr
                * self.loader.train.batch_size
                / self.optimizer.reference_batch_size
            )
        warmup_end = self.curriculum.phase1_end
        if self.workflow == "gan_bridge":
            if not isinstance(self.gan_bridge.checkpoint, str) or not self.gan_bridge.checkpoint.strip():
                raise ValueError("gan_bridge requires a pretrained GAN checkpoint (--gan-weights)")
            if self.teacher_mode != "two_teacher":
                raise ValueError("gan_bridge requires teacher_mode='two_teacher'")
            warmup_end = self.bridge_schedule.phase_ends(self.total_iters)[0]
            if self.ema.start_steps is not None and self.ema.start_steps != warmup_end:
                raise ValueError("gan_bridge EMA start must match the end of RGB warmup")
        if self.optimizer.warmup_iters is None:
            self.optimizer.warmup_iters = min(self.optimizer.warmup_max_iters, warmup_end)
        if self.ema.start_steps is None:
            self.ema.start_steps = warmup_end

    def get_phase(self, global_step: int) -> Phase:
        """Determines the phase using CurriculumConfig boundaries."""
        if self.workflow == "gan_bridge":
            stage = self.bridge_schedule.stage_for_step(global_step, self.total_iters)
            return (
                Phase.PHASE1_RGB_WARMUP, Phase.PHASE2_TRANSITION,
                Phase.PHASE3_ADAPTATION, Phase.PHASE3_ADAPTATION, Phase.PHASE4_IR_FOCUS,
            )[stage]
        cfg = self.curriculum
        if global_step < cfg.phase1_end:
            return Phase.PHASE1_RGB_WARMUP
        elif global_step < cfg.phase2_end:
            return Phase.PHASE2_TRANSITION
        elif global_step < cfg.phase3_end:
            return Phase.PHASE3_ADAPTATION
        else:
            return Phase.PHASE4_IR_FOCUS


__all__ = ['OptimizerConfig', 'PseudoLabelEvalConfig', 'TrainingConfig']

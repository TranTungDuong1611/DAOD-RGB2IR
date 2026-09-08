"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

Training flow:  RGB → MID(SAGA) → IR
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple
from enum import Enum

from models.torchvision_fcos_adapter import ClassificationInitMode

class Phase(Enum):
    """Enumeration of Curriculum Learning stages."""
    PHASE1_RGB_WARMUP = 1
    PHASE2_TRANSITION = 2  # Đảm bảo tên này tồn tại
    PHASE3_ADAPTATION = 3  # Đảm bảo tên này tồn tại
    PHASE4_IR_FOCUS   = 4

@dataclass
class TeacherSchedule:
    """Holds (ratio, min_hm) pairs for each curriculum phase."""
    # Format: (ratio, min_hm)
    phase1: Tuple[float, float]
    phase2: Tuple[float, float]
    phase3: Tuple[float, float]
    phase4: Tuple[float, float]

    def __post_init__(self):
        for phase_name in ("phase1", "phase2", "phase3", "phase4"):
            ratio, min_hm = getattr(self, phase_name)
            if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
                raise ValueError(f"{phase_name} top ratio must be in (0, 1]")
            if not math.isfinite(min_hm) or not 0.0 <= min_hm <= 1.0:
                raise ValueError(f"{phase_name} min_hm must be in [0, 1]")

    def get_params(self, phase: Phase) -> Tuple[float, float]:
        mapping = {
            Phase.PHASE1_RGB_WARMUP: self.phase1,
            Phase.PHASE2_TRANSITION:    self.phase2,
            Phase.PHASE3_ADAPTATION:     self.phase3,
            Phase.PHASE4_IR_FOCUS:   self.phase4,
        }
        return mapping.get(phase, self.phase2)
    
@dataclass
class DistillConfig:
    # Adaptive Threshold Schedules for RGB and IR Teachers
    # RGB Teacher: Learns faster, more reliable early on
    rgb_teacher: TeacherSchedule = field(default_factory=lambda: TeacherSchedule(
        phase1=(0.010, 0.45), phase2=(0.020, 0.35), 
        phase3=(0.040, 0.25), phase4=(0.060, 0.20)
    ))
    
    # IR Teacher: Noisier, requires stricter filtering in early phases
    ir_teacher: TeacherSchedule = field(default_factory=lambda: TeacherSchedule(
        phase1=(0.002, 0.60), phase2=(0.005, 0.50), 
        phase3=(0.015, 0.35), phase4=(0.030, 0.25)
    ))

    # Harmony Measure (HM) parameters: (prob^alpha) * (iou^beta)
    hm_alpha: float = 0.5
    hm_beta: float = 0.5
    
    # Uncertainty Weighting: weight = exp(-(1-HM) / un_regular_alpha)
    un_regular_alpha: float = 1.0

    def __post_init__(self):
        if self.hm_alpha < 0 or self.hm_beta < 0 or self.un_regular_alpha <= 0:
            raise ValueError("invalid HM/uncertainty settings")
    

@dataclass
class FCOSModelConfig:
    """Effective Torchvision FCOS model settings."""
    num_classes: int = 3
    class_names: Tuple[str, ...] = ("person", "car", "bicycle")
    weights: Optional[str] = "DEFAULT"
    classification_init_mode: ClassificationInitMode = ClassificationInitMode.COCO_TOWER
    pretrained_backbone: bool = True
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

@dataclass
class EMAConfig:
    """Exponential Moving Average settings for teacher update."""
    alpha: float = 0.996          # EMA decay factor (higher = slower teacher update)
    start_steps: Optional[int] = None

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("EMA alpha must be in [0, 1]")
        if self.start_steps is not None and (
            not isinstance(self.start_steps, int) or self.start_steps < 0
        ):
            raise ValueError("EMA start_steps must be None or a non-negative integer")


@dataclass
class SoftSAGAConfig:
    """SoftSAGA alpha per MID level (1.0=pure RGB, 0.0=pure gray)."""
    alpha_near_rgb:     float = 0.70   # objects mostly RGB
    alpha_intermediate: float = 0.50   # half RGB, half gray
    alpha_near_ir:      float = 0.25   # objects mostly gray


@dataclass
class StepRouting:
    """
    Defines the behavior for a specific training step.
    """
    ema_target: str = "none"      # Which teacher to update: "rgb" | "ir" | "none"
    use_gt: bool = True           # Whether to include Supervised Ground Truth loss
    student_saga_level: str = "rgb"  # "rgb" | "weak" | "mid" | "high" | "ir"
    teacher_saga_level: str = "rgb"  # "rgb" | "weak" | "mid" | "high" | "ir"
    teacher_names: Tuple[str, ...] = ()

@dataclass
class MidRoutingConfig:
    """
    Routes the data flow and EMA updates based on the DomainStep 
    returned by the CurriculumScheduler.
    """
    # PHASE 1: Supervised Warmup
    p1_rgb_supervised: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="none", use_gt=True, student_saga_level="rgb", teacher_saga_level="rgb",
        teacher_names=(),
    ))

    # PHASE 2: Transition (RGB Dominant)
    p2_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="weak", teacher_saga_level="rgb",
        teacher_names=("rgb",),
    ))
    p2_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=True, student_saga_level="high", teacher_saga_level="mid",
        teacher_names=("ir",),
    ))

    # PHASE 3: Adaptation (IR Dominant)
    p3_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="mid", teacher_saga_level="weak",
        teacher_names=("rgb",),
    ))
    p3_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="high",
        teacher_names=("ir",),
    ))

    # PHASE 4: IR Focus
    p4_ir_focus: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="ir",
        teacher_names=("ir",),
    ))

    def get_routing(self, step_name: str) -> StepRouting:
        """Helper to fetch routing params using the step name string."""
        return getattr(self, step_name, self.p4_ir_focus)


@dataclass
class AugConfig:
    """Augmentation config for student (applied on top of DataLoader transforms)."""
    # Geometric
    hflip_prob:              float = 0.5   # horizontal flip
    # Photometric
    blur_prob:               float = 0.5
    blur_sigma_max:          float = 1.0
    brightness_prob:         float = 0.3
    brightness_mag:          float = 0.2   # ±20% brightness
    contrast_prob:           float = 0.3
    contrast_mag:            float = 0.2   # ±20% contrast


@dataclass
class CurriculumConfig:
    """Phase boundaries and step ratios for alternating domain training."""
    phase1_end: int = 2000
    phase2_end: int = 5000
    phase3_end: int = 8000

    phase2_rgb_sampling_ratio: float = 0.7 
    phase3_rgb_sampling_ratio: float = 0.3  

    def __post_init__(self):
        boundaries = (self.phase1_end, self.phase2_end, self.phase3_end)
        if any(not isinstance(value, int) or value < 0 for value in boundaries):
            raise ValueError("phase boundaries must be non-negative integers")
        if tuple(boundaries) != tuple(sorted(boundaries)):
            raise ValueError("phase boundaries must be increasing")
        for name in ("phase2_rgb_sampling_ratio", "phase3_rgb_sampling_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")



@dataclass
class LossConfig:
    """
    Phase-based loss weights to control the balance between 
    Supervised (Ground Truth) and Unsupervised (Knowledge Distillation) learning.
    """
    weight_logits: float = 2.0
    weight_deltas: float = 1.0
    weight_quality: float = 1.0
    weight_uhl: float = 1.0

    # Phase 1: Pure Supervised Warmup (RGB + GT)
    p1_sup_weight: float = 1.0
    p1_distill_weight: float = 0.0

    # Phase 2: Transition (RGB/IR flows with GT + Distill)
    p2_sup_weight: float = 1.0
    p2_distill_weight: float = 0.3  # Start trusting teachers slightly less than GT

    # Phase 3: Adaptation (Shift weight towards Distill)
    p3_sup_weight: float = 0.8      # Reduce GT reliance (only used in RGB flow)
    p3_distill_weight: float = 1.0

    # Phase 4: IR Focus (Pure Distill)
    p4_sup_weight: float = 0.0      # No GT used
    p4_distill_weight: float = 1.0

    def get_phase_weights(self, phase: Phase) -> Tuple[float, float]:
        """Helper to retrieve (supervised_weight, distillation_weight) for a given phase."""
        mapping = {
            Phase.PHASE1_RGB_WARMUP: (self.p1_sup_weight, self.p1_distill_weight),
            Phase.PHASE2_TRANSITION: (self.p2_sup_weight, self.p2_distill_weight),
            Phase.PHASE3_ADAPTATION: (self.p3_sup_weight, self.p3_distill_weight),
            Phase.PHASE4_IR_FOCUS:   (self.p4_sup_weight, self.p4_distill_weight),
        }
        return mapping.get(phase, (1.0, 1.0))


@dataclass
class OptimizerConfig:
    """SGD and iteration-based LR schedule derived from the D3T solver."""

    base_lr: Optional[float] = None
    reference_lr: float = 0.016
    reference_batch_size: int = 16
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: Optional[int] = None
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
class DataConfig:
    root: str = "/home/duongtt/ws/DA/datasets/flir_data/align"

@dataclass
class TrainLoaderConfig:
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class EvalLoaderConfig:
    batch_size: int = 4
    num_workers: int = 2
    shuffle: bool = False


@dataclass
class DataLoaderConfig:
    train: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    eval: EvalLoaderConfig   = field(default_factory=EvalLoaderConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

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
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    data: DataConfig = field(default_factory=DataConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    max_iter: int = 10000
    total_iters: Optional[int] = None
    grad_clip: float = 10.0
    device: str = "cuda"
    log_interval: int = 50
    output_dir: str = "outputs"
    workflow: str = "curriculum"
    teacher_mode: str = "two_teacher"
    wandb: bool = False       
    eval_period: int = 500       

    def __post_init__(self):
        if self.total_iters is not None:
            if self.total_iters <= 0:
                raise ValueError("total_iters must be positive")
            self.max_iter = self.total_iters
        else:
            self.total_iters = self.max_iter
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.grad_clip < 0 or self.log_interval <= 0 or self.eval_period <= 0:
            raise ValueError("training intervals and grad_clip are invalid")
        if self.workflow not in {"curriculum", "rgb_baseline"}:
            raise ValueError("workflow must be 'curriculum' or 'rgb_baseline'")
        if self.teacher_mode not in {"rgb", "ir", "two_teacher"}:
            raise ValueError("teacher_mode must be 'rgb', 'ir', or 'two_teacher'")
        if self.optimizer.base_lr is None:
            self.optimizer.base_lr = (
                self.optimizer.reference_lr
                * self.loader.train.batch_size
                / self.optimizer.reference_batch_size
            )
        if self.optimizer.warmup_iters is None:
            self.optimizer.warmup_iters = min(1000, self.curriculum.phase1_end)
        if self.ema.start_steps is None:
            self.ema.start_steps = self.curriculum.phase1_end

    def get_phase(self, global_step: int) -> Phase:
        """Determines the phase using CurriculumConfig boundaries."""
        cfg = self.curriculum
        if global_step < cfg.phase1_end:
            return Phase.PHASE1_RGB_WARMUP
        elif global_step < cfg.phase2_end:
            return Phase.PHASE2_TRANSITION
        elif global_step < cfg.phase3_end:
            return Phase.PHASE3_ADAPTATION
        else:
            return Phase.PHASE4_IR_FOCUS

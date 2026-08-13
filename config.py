"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

Training flow:  RGB → MID(SAGA) → IR
"""

from dataclasses import dataclass, field
from typing import Tuple
from enum import Enum

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
    # Phase boundaries (iterations)
    phase_boundaries: Tuple[int, int, int] = (12500, 15000, 17500)
    
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
    hm_alpha: float = 1.0
    hm_beta: float = 1.0
    
    # Uncertainty Weighting: weight = exp(-(1-HM) / un_regular_alpha)
    un_regular_alpha: float = 4.0
    

@dataclass
class FCOSModelConfig:
    """Settings for build_custom_fcos and HMFocalClassificationHead."""
    num_classes: int = 3
    pretrained_backbone: bool = True
    trainable_backbone_layers: int = 3
    min_size: int = 600
    max_size: int = 1000
    from_coco: bool = True
    
    # HM-Focal Loss (VFL) Hyperparameters
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0
    vfl_weight_type: str = "iou"
    vfl_loss_weight: float = 1.0

@dataclass
class EMAConfig:
    """Exponential Moving Average settings for teacher update."""
    alpha: float = 0.996          # EMA decay factor (higher = slower teacher update)
    start_steps: int = 6000      # ramp alpha up during early training


@dataclass
class SAGAConfig:
    """SemanticAwareGrayAugmentation settings (hard SAGA, legacy)."""
    apply_prob: float = 0.5       # probability of applying SAGA per image


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

@dataclass
class MidRoutingConfig:
    """
    Routes the data flow and EMA updates based on the DomainStep 
    returned by the CurriculumScheduler.
    """
    # PHASE 1: Supervised Warmup
    p1_rgb_supervised: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="none", use_gt=True, student_saga_level="rgb", teacher_saga_level="rgb"
    ))

    # PHASE 2: Transition (RGB Dominant)
    p2_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="weak", teacher_saga_level="rgb"
    ))
    p2_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=True, student_saga_level="high", teacher_saga_level="mid"
    ))

    # PHASE 3: Adaptation (IR Dominant)
    p3_rgb_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="rgb", use_gt=True, student_saga_level="mid", teacher_saga_level="weak"
    ))
    p3_ir_flow: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="high"
    ))

    # PHASE 4: IR Focus
    p4_ir_focus: StepRouting = field(default_factory=lambda: StepRouting(
        ema_target="ir", use_gt=False, student_saga_level="ir", teacher_saga_level="ir"
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



@dataclass
class LossConfig:
    """
    Phase-based loss weights to control the balance between 
    Supervised (Ground Truth) and Unsupervised (Knowledge Distillation) learning.
    """
    weight_logits: float = 2.0
    weight_deltas: float = 1.0
    weight_quality: float = 1.0

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
class TeacherUpdateConfig:
    """Flags determining which teacher is updated via EMA in each step."""
    update_rgb: bool = True
    update_ir: bool = True


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
    saga: SAGAConfig = field(default_factory=SAGAConfig)
    soft_saga: SoftSAGAConfig = field(default_factory=SoftSAGAConfig)
    mid_routing: MidRoutingConfig = field(default_factory=MidRoutingConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)

    data: DataConfig = field(default_factory=DataConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    step2_start: int = 2000
    max_iter: int = 10000
    grad_clip: float = 10.0
    device: str = "cuda"
    log_interval: int = 50
    output_dir: str = "outputs"
    wandb: bool = False       
    eval_period: int = 500       

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

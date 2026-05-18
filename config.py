"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

Training flow:  RGB → MID(SAGA 100%) → IR
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EMAConfig:
    """Exponential Moving Average settings for teacher update."""
    alpha: float = 0.999          # EMA decay factor (higher = slower teacher update)
    use_warmup: bool = True       # ramp alpha up during early training


@dataclass
class SAGAConfig:
    """SemanticAwareGrayAugmentation settings — always applied (apply_prob=1.0)."""
    apply_prob: float = 1.0       # 1.0 = deterministic, always apply SAGA


@dataclass
class RGBAugConfig:
    """Augmentation for RGB and MID (SAGA) images."""
    # --- Geometric (weak aug — applied to BOTH teacher and student) ---
    hflip_prob:                  float = 0.5

    # Multi-scale: resize to [scale_min, scale_max] × original, then crop/pad to fixed size
    multiscale_min:              float = 0.5
    multiscale_max:              float = 1.5
    multiscale_target_h:         int   = 512
    multiscale_target_w:         int   = 640

    # --- Photometric (strong aug — applied to student only) ---
    blur_prob:                   float = 0.5
    blur_sigma_max:              float = 1.0

    color_jitter_prob:           float = 0.5
    cj_brightness:               float = 0.2
    cj_contrast:                 float = 0.2
    cj_saturation:               float = 0.3
    cj_hue:                      float = 0.05

    random_erasing_prob:         float = 0.3
    random_erasing_scale_min:    float = 0.02
    random_erasing_scale_max:    float = 0.10
    random_erasing_ratio_min:    float = 0.3
    random_erasing_ratio_max:    float = 3.3


@dataclass
class IRAugConfig:
    """Augmentation for IR (thermal) images."""
    # --- Geometric (weak aug — applied to BOTH teacher and student) ---
    hflip_prob:                  float = 0.5

    # Multi-scale
    multiscale_min:              float = 0.5
    multiscale_max:              float = 1.5
    multiscale_target_h:         int   = 512
    multiscale_target_w:         int   = 640

    # --- Photometric (strong aug — applied to student only) ---
    intensity_shift_prob:        float = 0.5
    intensity_shift_mag:         float = 0.1

    contrast_jitter_prob:        float = 0.5
    contrast_jitter_mag:         float = 0.2

    gamma_prob:                  float = 0.3
    gamma_min:                   float = 0.7
    gamma_max:                   float = 1.3

    gaussian_noise_prob:         float = 0.3
    gaussian_noise_std:          float = 0.02


@dataclass
class CurriculumConfig:
    """
    Phase boundaries (in global iterations).

    Phase 1: [0,           phase1_end)  → RGB only             (supervised warmup)
    Phase 2: [phase1_end,  phase2_end)  → RGB + MID alternating (rgb→rgb_teacher, mid→ir_teacher)
    Phase 3: [phase2_end,  ∞)           → IR only               (IR focus)

    phase2_rgb_ratio: fraction of Phase 2 steps that are RGB (rest = MID).
      0.5  → equal split  (rgb, mid, rgb, mid, ...)
      0.67 → more RGB     (rgb, rgb, mid, ...)
    """
    phase1_end: int = 3_000
    phase2_end: int = 17_000

    phase2_rgb_ratio: float = 0.5


@dataclass
class LossConfig:
    """Loss weights for each domain step."""
    # RGB step
    rgb_gt_weight: float = 1.0
    rgb_pseudo_weight: float = 0.0    # set > 0 to enable pseudo loss in RGB step

    # MID step — both teachers always active
    mid_rgb_weight: float = 0.5       # weight for rgb_teacher pseudo-labels
    mid_ir_weight:  float = 0.5       # weight for ir_teacher  pseudo-labels
    mid_gt_weight:  float = 1.0       # weight for GT loss on MID (set 0 to disable)

    # IR step
    ir_ir_teacher_weight: float = 1.0


@dataclass
class TeacherUpdateConfig:
    """Which teachers to update in each step."""
    # RGB step: update rgb_teacher (not during Phase 1)
    rgb_update_rgb_teacher: bool = True

    # IR step: update ir_teacher
    ir_update_ir_teacher: bool = True

    # MID step: always updates both teachers (not configurable)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Master config for CurriculumDomainAdaptationTrainer."""
    ema: EMAConfig = field(default_factory=EMAConfig)
    saga: SAGAConfig = field(default_factory=SAGAConfig)
    rgb_aug: RGBAugConfig = field(default_factory=RGBAugConfig)
    ir_aug: IRAugConfig = field(default_factory=IRAugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

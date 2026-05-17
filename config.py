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
    # Color jitter (brightness + contrast + saturation + hue in one transform)
    color_jitter_prob:       float = 0.5
    cj_brightness:           float = 0.2
    cj_contrast:             float = 0.2
    cj_saturation:           float = 0.3
    cj_hue:                  float = 0.05


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
    aug: AugConfig = field(default_factory=AugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

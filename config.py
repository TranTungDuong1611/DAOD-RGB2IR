"""
Configuration dataclasses for Block Domain Adaptation framework (D3T-style).

BlockDomainScheduler (see scheduler.py) mixes 3 domains — rgb / mid (GAN
bridge) / ir (real thermal) — whose ratio changes across a sequence of
phases; within each phase, short cycles are split into contiguous
rgb -> mid -> ir blocks. Training uses dense distillation (losses.py)
against two frozen teachers (rgb_teacher / ir_teacher) instead of hard
pseudo-labels.
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
class GANConfig:
    """GAN translator settings — RGB → MID via pretrained GAN generator."""
    checkpoint_path: str   = ""       # path tới file .pth của GAN generator
    input_nc:        int   = 3        # channels input GAN (RGB)
    output_nc:       int   = 1        # channels output GAN (grayscale)
    ngf:             int   = 64       # base filter count
    n_blocks:        int   = 9        # số ResNet blocks
    state_dict_key:  str   = ""       # key trong checkpoint dict ("" = state_dict thẳng)
    amp:             bool  = False    # dùng autocast khi GAN inference


@dataclass
class RGBAugConfig:
    """Augmentation for RGB and MID (GAN) images."""
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
class BlockSchedulerConfig:
    """
    BlockDomainScheduler config — 3 domains (rgb / mid / ir) whose mixing ratio
    changes across phases. Within each phase, iterations are grouped into short
    `cycle_iters`-long cycles; each cycle is split into three CONTIGUOUS blocks
    trained in order rgb -> mid -> ir, sized by the phase's ratio.

    ratio_schedule : "rgb:mid:ir" tuples per phase, comma-separated, e.g.
                     "7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8"
    cycle_iters    : length (in iters) of one rgb->mid->ir cycle within a phase.
    """
    ratio_schedule: str = "7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8"
    cycle_iters: int = 200


@dataclass
class DistillConfig:
    """Dense distillation (D3T-style) hyperparameters — FCOS only."""
    ratio:     float = 0.01   # top-ratio region selection by harmony measure
    hm_alpha:  float = 0.5    # hm = cls_prob^hm_alpha * iou^hm_beta
    hm_beta:   float = 0.5
    un_alpha:  float = 1.0    # uncertainty weighting: exp(-(1-hm)/un_alpha)
    w_logits:  float = 4.0    # QFLv2 loss weight
    w_deltas:  float = 1.0    # GIoU loss weight
    w_quality: float = 1.0    # centerness/quality BCE loss weight
    w_uhl:     float = 1.0    # VFL/UHL loss weight


@dataclass
class TeacherWeightConfig:
    """Two-teacher weighting for the unsupervised distillation loss.

    Each domain step distills against BOTH teachers: the one matching the
    image's domain (w_match, high) and the cross-domain one (w_cross, low).
    """
    w_match: float = 1.0
    w_cross: float = 0.3


@dataclass
class DisjointSplitConfig:
    """RGB-source / real-IR-target disjoint split of align_train (D3T paper 4.1)."""
    enabled: bool = True
    source_frac: float = 0.5
    split_seed: int = -1   # -1 = "first N" (paper); >=0 = deterministic shuffle


@dataclass
class LambdaConfig:
    """Ramp for the unsupervised (distillation) loss weight lambda."""
    lambda_max: float = 1.0
    ramp_iters: int = 6_000   # 0 = lambda fixed at lambda_max from the start


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Master config for BlockDomainAdaptationTrainer."""
    ema: EMAConfig = field(default_factory=EMAConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    rgb_aug: RGBAugConfig = field(default_factory=RGBAugConfig)
    ir_aug: IRAugConfig = field(default_factory=IRAugConfig)
    block_scheduler: BlockSchedulerConfig = field(default_factory=BlockSchedulerConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    teacher_weight: TeacherWeightConfig = field(default_factory=TeacherWeightConfig)
    disjoint_split: DisjointSplitConfig = field(default_factory=DisjointSplitConfig)
    lambda_cfg: LambdaConfig = field(default_factory=LambdaConfig)

    # Supervised GT loss weight, applied whenever a domain step has GT (rgb / mid)
    gt_weight: float = 1.0
    # Number of foreground classes — needed by dense distillation (losses.distill)
    distill_num_classes: int = 3

    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

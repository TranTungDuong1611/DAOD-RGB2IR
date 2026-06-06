"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

4-phase training flow with **mixed-batch** Phase 2 and Phase 3:

    Phase 1  → RGB only            (supervised warmup)
    Phase 2  → mixed batch [RGB | MID]   (rgb_part keeps RGB, mid_part = SAGA)
    Phase 3  → mixed batch [MID | IR]    (mid_part = SAGA, ir_part = unlabeled IR)
    Phase 4  → IR only             (IR focus, unsupervised)
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
    Phase boundaries (in global iterations) for the 4-phase curriculum.

    Phase 1: [0,          phase1_end)  → RGB only           (supervised warmup)
    Phase 2: [phase1_end, phase2_end)  → mixed [RGB | MID]  (in-batch split)
    Phase 3: [phase2_end, phase3_end)  → mixed [MID | IR]   (in-batch split)
    Phase 4: [phase3_end, ∞)           → IR only            (IR focus)

    In-batch split ratios:
      phase2_rgb_ratio : fraction of each Phase-2 batch that's RGB (rest is MID).
      phase3_mid_ratio : fraction of each Phase-3 batch that's MID (rest is IR).

    Example (batch_size=8, phase2_rgb_ratio=0.5):
      batch layout = [RGB, RGB, RGB, RGB | MID, MID, MID, MID]
    """
    phase1_end: int = 3_000
    phase2_end: int = 10_000
    phase3_end: int = 17_000

    phase2_rgb_ratio: float = 0.5
    phase3_mid_ratio: float = 0.5


@dataclass
class LossConfig:
    """Loss weights per phase."""
    # Phase 1 — RGB warmup (no teacher pseudo in pure warmup)
    p1_gt_weight:           float = 1.0
    p1_pseudo_weight:       float = 0.0   # optional rgb_teacher pseudo on RGB (kept 0 in warmup)

    # Phase 2 — mixed [RGB | MID]; GT applies to whole batch (both halves have GT)
    p2_gt_weight:           float = 1.0
    p2_rgb_teacher_weight:  float = 0.5
    p2_ir_teacher_weight:   float = 0.5

    # Phase 3 — mixed [MID | IR]; GT applies to MID slice ONLY (IR has no GT)
    p3_gt_weight:           float = 1.0
    p3_rgb_teacher_weight:  float = 0.5
    p3_ir_teacher_weight:   float = 0.5

    # Phase 4 — IR focus (unsupervised, ir_teacher only)
    p4_ir_teacher_weight:   float = 1.0


@dataclass
class AdvConfig:
    """
    Adversarial domain alignment via Gradient Reversal Layer (DANN-style).

    Phase 2: disc_rgb distinguishes RGB (0) vs MID (1)
    Phase 3: disc_ir  distinguishes MID (0) vs IR  (1)

    Set p2_adv_weight / p3_adv_weight = 0 to disable per-phase.
    """
    p2_adv_weight:   float = 0.2    # adversarial loss weight in Phase 2
    p3_adv_weight:   float = 0.2    # adversarial loss weight in Phase 3
    backbone_dim:    int   = 2048   # ResNet50 layer4 channels (input to discriminator)
    disc_hidden:     int   = 256    # discriminator hidden units (lightweight)
    disc_lr:         float = 1e-4   # discriminator optimizer LR (AdamW)
    label_smoothing: float = 0.1    # CrossEntropy label smoothing for discriminator
    grl_lambda:      float = 1.0    # max / fixed lambda for GRL
    use_schedule:    bool  = True   # True = DANN progressive schedule; False = fixed lambda


@dataclass
class TeacherUpdateConfig:
    """
    Which teachers to EMA-update in each phase step.

    Selected policy: only the "specialist" teacher updates per phase.
      Phase 1 (rgb step)      : no EMA update (warmup)
      Phase 2 (rgb_mid step)  : rgb_teacher only
      Phase 3 (mid_ir step)   : ir_teacher  only
      Phase 4 (ir step)       : ir_teacher  only
    """
    p2_update_rgb_teacher: bool = True
    p2_update_ir_teacher:  bool = False

    p3_update_rgb_teacher: bool = False
    p3_update_ir_teacher:  bool = True

    p4_update_ir_teacher:  bool = True


@dataclass
class ContrastiveConfig:
    """
    Object-level supervised contrastive loss (CMT-style).

    Applied in two places:
      Phase 2 : student (strong RGB/MID) vs rgb_teacher (weak RGB/MID), GT labels
      Phase 3 : student (strong MID)     vs rgb_teacher (weak MID),     GT labels
                student (strong IR)      vs ir_teacher  (weak IR),      pseudo-labels

    Contrastive uses a HIGHER confidence threshold than detection to limit
    false-positive contamination of the feature-space structure.

    p3_ir_start_offset: number of Phase-3 steps to skip before enabling the
    IR contrastive — gives ir_teacher time to adapt to IR before its features
    are used as contrastive anchors.

    feature_levels: FPN output keys to use for multi-scale object features.
      FCOS          : "0" = P3 (stride 8), "1" = P4 (stride 16), "2" = P5 (stride 32)
      Faster R-CNN  : same key convention
    """
    enabled: bool = False

    temperature: float = 0.07

    # Threshold for selecting pseudo-label boxes used in contrastive (IR slice, Phase 3).
    # Should be higher than pseudo_label_conf_thresh to keep only reliable labels.
    conf_thresh: float = 0.90

    # Loss weights per context
    p2_weight:     float = 0.05   # Phase 2 full batch (GT labels)
    p3_mid_weight: float = 0.05   # Phase 3 MID slice  (GT labels)
    p3_ir_weight:  float = 0.05   # Phase 3 IR  slice  (pseudo-labels)

    # Delay IR contrastive N steps into Phase 3 to let ir_teacher stabilise on IR
    p3_ir_start_offset: int = 1_000

    # FPN levels to compute contrastive loss on (multi-scale)
    feature_levels: tuple = ("0", "1", "2")


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
    adv: AdvConfig = field(default_factory=AdvConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

Training flow:  RGB → MID(SAGA) → IR
"""

from dataclasses import dataclass, field
from typing import Optional


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
    """SemanticAwareGrayAugmentation settings (hard SAGA, legacy)."""
    apply_prob: float = 0.5       # probability of applying SAGA per image


@dataclass
class SoftSAGAConfig:
    """SoftSAGA alpha per MID level (1.0=pure RGB, 0.0=pure gray)."""
    alpha_near_rgb:     float = 0.70   # objects mostly RGB
    alpha_intermediate: float = 0.50   # half RGB, half gray
    alpha_near_ir:      float = 0.25   # objects mostly gray


@dataclass
class MidRoutingConfig:
    """
    Per-MID-level routing: which teacher generates pseudo-labels and
    which teacher receives the EMA update.

    teacher_source: "rgb" | "ir" | "both"
    ema_target:     "rgb" | "ir" | "none"
    """
    near_rgb_teacher_source:      str   = "rgb"    # rgb_teacher infers
    near_rgb_ema_target:          str   = "rgb"    # rgb_teacher updated
    near_rgb_rgb_weight:          float = 1.0
    near_rgb_ir_weight:           float = 0.0

    intermediate_teacher_source:  str   = "both"
    intermediate_ema_target:      str   = "ir"     # ir_teacher updated (lighter)
    intermediate_ema_alpha:       float = 0.9998   # slower EMA for gentle start
    intermediate_rgb_weight:      float = 0.5
    intermediate_ir_weight:       float = 0.5

    near_ir_teacher_source:       str   = "ir"     # ir_teacher infers
    near_ir_ema_target:           str   = "ir"     # ir_teacher updated
    near_ir_rgb_weight:           float = 0.0
    near_ir_ir_weight:            float = 1.0


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
    Phase boundaries (in global iterations) and within-phase ratios.

    Phase 1: [0,           phase1_end)   → RGB only            (supervised warmup)
    Phase 2: [phase1_end,  phase2_end)   → RGB + mid_near_rgb  (bridge starts)
    Phase 3: [phase2_end,  phase3_end)   → mid_intermediate     (full bridge)
    Phase 4: [phase3_end,  phase4_end)   → mid_near_ir + IR    (IR adaptation)
    Phase 5: [phase4_end,  ∞)            → IR only             (IR focus)
    """
    phase1_end: int = 3_000
    phase2_end: int = 7_000
    phase3_end: int = 12_000
    phase4_end: int = 17_000

    # Ratio of RGB steps in Phase 2  (rest = mid_near_rgb)
    phase2_rgb_ratio: float = 0.67

    # Ratio of mid_near_ir steps in Phase 4  (rest = IR)
    phase4_mid_ratio: float = 0.67


@dataclass
class HarmonyConfig:
    """
    Harmony reweighting for pseudo-label loss (Harmonious Teacher, Eq. 5 & 9).

    h_i = p_i^beta * u_i^(1-beta)

    p_i  = teacher confidence score for pseudo-box i
    u_i  = localization quality proxy:
           • Supervised  (GT available): max IoU(pseudo_box_i, GT_boxes)      [exact]
           • Unsupervised (no GT):       max_{j≠i} IoU(pseudo_box_i, box_j)   [paper Eq.9]

    h_i is used ONLY as a per-proposal loss weight — class labels and box
    regression targets are unchanged.

    Ablation presets
    ----------------
    Baseline (no harmony):
        use_harmony_weight=False, conf_thresh=0.7
    Harmony safe:
        use_harmony_weight=True, conf_thresh=0.5
    Harmony aggressive:
        use_harmony_weight=True, conf_thresh=0.3, min_threshold=0.4
    """
    use_harmony_weight: bool = False
    beta: float = 0.5               # blend: 1.0 → h=p only, 0.0 → h=u only
    min_threshold: Optional[float] = None  # discard pseudo-box if h_i < min_threshold
    max_boxes_per_image: Optional[int] = None  # keep top-N by h_i (None = no limit)
    neg_proposal_weight: float = 0.0  # weight for bg ROI proposals in pseudo loss
    # RPN pseudo loss control
    use_rpn_pseudo: bool = True       # include RPN losses in pseudo-label loss
    rpn_pseudo_factor: float = 0.5    # relative weight of RPN pseudo vs ROI pseudo


@dataclass
class LossConfig:
    """Loss weights for each domain step."""
    # RGB step
    rgb_gt_weight: float = 1.0
    rgb_pseudo_weight: float = 0.0    # set > 0 to enable pseudo loss in RGB step

    # MID step (Phase 2/3: both teachers; Phase 4: ir_teacher only)
    mid_rgb_weight: float = 0.5       # weight for rgb_teacher pseudo-labels (Phase 2/3)
    mid_ir_weight: float = 0.5        # weight for ir_teacher  pseudo-labels
    mid_gt_weight: float = 1.0        # weight for GT loss on MID

    # IR step (Phase 4/5: ir_teacher only, no GT)
    ir_ir_teacher_weight: float = 1.0

    # Harmony reweighting (applied to all pseudo-label losses)
    harmony: HarmonyConfig = field(default_factory=HarmonyConfig)


@dataclass
class TeacherUpdateConfig:
    """Which teachers to update in each step (default = D3T-style)."""
    # RGB step: always update rgb_teacher
    rgb_update_rgb_teacher: bool = True

    # MID step: configurable
    mid_update_rgb_teacher: bool = False
    mid_update_ir_teacher: bool = False

    # IR step: always update ir_teacher
    ir_update_ir_teacher: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Master config for CurriculumDomainAdaptationTrainer."""
    ema: EMAConfig = field(default_factory=EMAConfig)
    saga: SAGAConfig = field(default_factory=SAGAConfig)
    soft_saga: SoftSAGAConfig = field(default_factory=SoftSAGAConfig)
    mid_routing: MidRoutingConfig = field(default_factory=MidRoutingConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

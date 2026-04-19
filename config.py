"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

Training flow:  RGB → MID(SAGA) → IR
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
    """SemanticAwareGrayAugmentation settings."""
    apply_prob: float = 0.5       # probability of applying SAGA per image


@dataclass
class CurriculumConfig:
    """
    Phase boundaries (in global iterations) and within-phase ratios.

    Phase 1: [0,           phase1_end)   → RGB only   (warmup)
    Phase 2: [phase1_end,  phase2_end)   → RGB + MID  (alternating)
    Phase 3: [phase2_end,  phase3_end)   → MID + IR   (alternating)
    Phase 4: [phase3_end,  ∞)            → IR focus   (with occasional MID)
    """
    phase1_end: int = 2_000       # end of RGB warmup
    phase2_end: int = 5_000       # end of RGB+MID phase
    phase3_end: int = 8_000       # end of MID+IR phase

    # Ratio of RGB steps in Phase 2  (rest = MID)
    phase2_rgb_ratio: float = 0.5

    # Ratio of MID steps in Phase 3  (rest = IR)
    phase3_mid_ratio: float = 0.5

    # In Phase 4: every N IR steps, insert 1 MID step for stability
    phase4_mid_every_n: int = 5


@dataclass
class LossConfig:
    """Loss weights for each domain step."""
    # RGB step
    rgb_gt_weight: float = 1.0
    rgb_pseudo_weight: float = 0.0    # set > 0 to enable pseudo loss in RGB step

    # MID step
    mid_rgb_weight: float = 0.5       # weight for rgb_teacher pseudo-labels on MID
    mid_ir_weight: float = 0.5        # weight for ir_teacher  pseudo-labels on MID
    mid_gt_weight: float = 0.0        # weight for GT loss on MID (optional)

    # IR step
    ir_rgb_teacher_weight: float = 0.5
    ir_ir_teacher_weight: float = 0.5


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
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations

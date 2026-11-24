"""Resolve iteration policy; the trainer owns execution and scheduler state."""

from dataclasses import dataclass

from config import Phase, StepRouting, TrainingConfig
from models.detector_adapter import DistillationSettings
from scheduler import BridgeDomainScheduler, CurriculumScheduler


@dataclass(frozen=True)
class StepPlan:
    """Resolved data, loss and teacher decisions for one zero-based iteration."""

    step_name: str
    phase: Phase
    source_domain: str
    use_gt: bool
    student_alpha: float | None
    teacher_alpha: float | None
    supervised_weight: float
    distill_weight: float
    teacher_names: tuple[str, ...]
    distillation: tuple[tuple[str, DistillationSettings], ...]
    initialize_teacher_names: tuple[str, ...]
    ema_target: str | None
    appearance_domain: str | None = None
    bridge_stage: int | None = None
    domain_ratio: tuple[float, ...] | None = None
    color_consistency: bool = False

    def log_metadata(self, *, did_step: bool, ema_initialized: bool) -> dict:
        """Keep bridge diagnostics aligned with the decisions actually executed."""
        if self.bridge_stage is None:
            return {}
        metadata = {
            "domain": self.appearance_domain,
            "source_domain": self.source_domain,
            "bridge_stage": self.bridge_stage + 1,
            "domain_ratio": self.domain_ratio,
            "teacher_names": ",".join(self.teacher_names) if ema_initialized else "none",
            "ema_target": self.ema_target if did_step and ema_initialized and self.ema_target else "none",
            "supervised_weight": self.supervised_weight if self.use_gt else 0.0,
            "distill_weight": self.distill_weight,
        }
        if self.appearance_domain == "saga":
            metadata["saga_student_alpha"] = self.student_alpha
            metadata["saga_teacher_alpha"] = self.teacher_alpha
        return metadata


def configured_teacher_names(config: TrainingConfig) -> tuple[str, ...]:
    """Use the same teacher-mode selection for initialization, KD and audits."""
    return ("rgb", "ir") if config.teacher_mode == "two_teacher" else (config.teacher_mode,)


def distillation_settings(config: TrainingConfig, teacher_name: str, phase: Phase):
    """Resolve teacher thresholds consistently for training and observation."""
    schedule = getattr(config.distill, f"{teacher_name}_teacher")
    top_ratio, min_hm = schedule.get_params(phase)
    return DistillationSettings(
        top_ratio=top_ratio,
        min_hm=min_hm,
        hm_alpha=config.distill.hm_alpha,
        hm_beta=config.distill.hm_beta,
        uncertainty_alpha=config.distill.un_regular_alpha,
    )


def bridge_route(config: TrainingConfig, stage: int, domain: str) -> StepRouting:
    if domain not in {"rgb", "saga", "ir_gan", "ir_real"}:
        raise ValueError(f"unknown bridge domain: {domain}")
    if stage == 0:
        if domain != "rgb":
            raise ValueError("bridge warmup is RGB-only")
        return StepRouting(use_gt=True)
    teacher = "rgb" if domain == "rgb" else "ir"
    if domain == "saga":
        teacher = config.gan_bridge.saga_teachers[stage]
    return StepRouting(ema_target=teacher, use_gt=domain != "ir_real", teacher_names=(teacher,))


def resolve_step(
    config: TrainingConfig,
    scheduler: BridgeDomainScheduler | CurriculumScheduler,
    global_step: int,
    *,
    ema_initialized: bool,
) -> StepPlan:
    """Select one route and resolve its policy without touching training objects.

    Route selection advances the supplied scheduler once (except for the fixed
    RGB baseline). The trainer retains ownership of checkpointing and rollback.
    Resolving a plan does not consume RNG or initialize/update any teacher.
    """
    stage = None
    domain = None
    domain_ratio = None
    if config.workflow == "gan_bridge":
        domain = scheduler.get_next_domain(global_step)
        stage = config.bridge_schedule.stage_for_step(global_step, config.total_iters)
        step_name = f"bridge_{domain}"
        route = bridge_route(config, stage, domain)
        source = "ir" if domain == "ir_real" else "rgb"
        student_alpha = config.gan_bridge.saga_student_alphas[stage] if domain == "saga" else None
        teacher_alpha = config.gan_bridge.saga_teacher_alphas[stage] if domain == "saga" else None
        domain_ratio = config.bridge_schedule.phase_ratios[stage]
    else:
        step_name = (
            "p1_rgb_supervised" if config.workflow == "rgb_baseline"
            else scheduler.get_next_step(global_step)
        )
        route = config.mid_routing.get_routing(step_name)
        source = scheduler.get_source_domain(step_name)
        alphas = {
            "rgb": 1.0,
            "weak": config.soft_saga.alpha_near_rgb,
            "mid": config.soft_saga.alpha_intermediate,
            "high": config.soft_saga.alpha_near_ir,
            "ir": 0.0,
        }
        student_alpha = alphas[route.student_saga_level] if source == "rgb" else None
        teacher_alpha = alphas[route.teacher_saga_level] if source == "rgb" else None

    phase = config.get_phase(global_step)
    supervised_weight, distill_weight = config.loss.get_phase_weights(phase)
    if config.workflow == "rgb_baseline":
        supervised_weight, distill_weight = 1.0, 0.0
    elif stage == 4 and route.use_gt:
        supervised_weight = config.gan_bridge.source_sup_weight

    configured_names = configured_teacher_names(config)
    teacher_names = tuple(name for name in route.teacher_names if name in configured_names)
    initialize_names = (
        configured_names
        if config.workflow != "rgb_baseline" and not ema_initialized
        and global_step >= config.ema.start_steps
        else ()
    )
    teachers_ready = ema_initialized or bool(initialize_names)
    distillation = (
        tuple((name, distillation_settings(config, name, phase)) for name in teacher_names)
        if config.workflow != "rgb_baseline" and teachers_ready and distill_weight > 0
        else ()
    )
    return StepPlan(
        step_name=step_name,
        phase=phase,
        source_domain=source,
        use_gt=route.use_gt,
        student_alpha=student_alpha,
        teacher_alpha=teacher_alpha,
        supervised_weight=supervised_weight,
        distill_weight=distill_weight,
        teacher_names=teacher_names,
        distillation=distillation,
        initialize_teacher_names=initialize_names,
        ema_target=route.ema_target if teachers_ready and route.ema_target in teacher_names else None,
        appearance_domain=domain,
        bridge_stage=stage,
        domain_ratio=domain_ratio,
        color_consistency=(
            config.loss.color_consistency_weight > 0
            and config.workflow != "rgb_baseline" and teachers_ready
            and "rgb" in configured_names and source == "rgb" and route.use_gt
            and student_alpha is not None and student_alpha < 1.0
        ),
    )

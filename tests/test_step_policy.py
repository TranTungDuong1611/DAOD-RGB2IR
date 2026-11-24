"""Step decisions stay consistent across data, losses, teachers and resume."""

import pytest
import torch

from config import (
    BridgeScheduleConfig, CurriculumConfig, EMAConfig, GANBridgeConfig,
    Phase, TrainingConfig, LossConfig, SoftSAGAConfig,
)
from scheduler import BridgeDomainScheduler, CurriculumScheduler
from training import step_policy as policy


@pytest.mark.parametrize(
    "step,ratio,route,phase,source,use_gt,alphas,teacher,weights",
    [
        (0, 1, "p1_rgb_supervised", Phase.PHASE1_RGB_WARMUP, "rgb", True, (1, 1), None, (1, 0)),
        (2, 1, "p2_rgb_flow", Phase.PHASE2_TRANSITION, "rgb", True, (.7, 1), "rgb", (1, .3)),
        (2, 0, "p2_ir_flow", Phase.PHASE2_TRANSITION, "rgb", True, (.25, .5), "ir", (1, .3)),
        (4, 1, "p3_rgb_flow", Phase.PHASE3_ADAPTATION, "rgb", True, (.5, .7), "rgb", (.8, 1)),
        (4, 0, "p3_ir_flow", Phase.PHASE3_ADAPTATION, "ir", False, (None, None), "ir", (.8, 1)),
        (6, 1, "p4_rgb_supervised", Phase.PHASE4_IR_FOCUS, "rgb", True, (1, 1), None, (1, 1)),
        (6, 0, "p4_ir_focus", Phase.PHASE4_IR_FOCUS, "ir", False, (None, None), "ir", (1, 1)),
    ],
)
def test_curriculum_resolves_physical_source_and_teacher_policy(
    step, ratio, route, phase, source, use_gt, alphas, teacher, weights,
):
    config = TrainingConfig(loss=LossConfig(p2_distill_weight=.3, p3_sup_weight=.8),
                            soft_saga=SoftSAGAConfig(alpha_near_rgb=.7), curriculum=CurriculumConfig(
        phase1_end=2, phase2_end=4, phase3_end=6,
        phase2_rgb_sampling_ratio=ratio, phase3_rgb_sampling_ratio=ratio,
        phase4_rgb_sampling_ratio=ratio,
    ))
    plan = policy.resolve_step(config, CurriculumScheduler(config), step, ema_initialized=True)

    assert (plan.step_name, plan.phase) == (route, phase)
    assert (plan.source_domain, plan.use_gt) == (source, use_gt)
    assert (plan.student_alpha, plan.teacher_alpha) == alphas
    assert (plan.supervised_weight, plan.distill_weight) == weights
    assert plan.ema_target == teacher
    assert tuple(name for name, _ in plan.distillation) == (() if teacher is None else (teacher,))
    assert plan.initialize_teacher_names == ()
    assert plan.log_metadata(did_step=True, ema_initialized=True) == {}


def test_baseline_remains_supervised_after_the_curriculum_warmup():
    config = TrainingConfig(workflow="rgb_baseline", ema=EMAConfig(start_steps=0))
    plan = policy.resolve_step(config, CurriculumScheduler(config), 9000, ema_initialized=False)

    assert plan.phase is Phase.PHASE4_IR_FOCUS
    assert plan.step_name == "p1_rgb_supervised"
    assert (plan.source_domain, plan.use_gt) == ("rgb", True)
    assert (plan.student_alpha, plan.teacher_alpha) == (1, 1)
    assert (plan.supervised_weight, plan.distill_weight) == (1, 0)
    assert plan.distillation == ()
    assert plan.initialize_teacher_names == ()
    assert plan.ema_target is None


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
@pytest.mark.parametrize("domain", ["rgb", "saga", "ir_gan", "ir_real"])
def test_bridge_resolves_each_domain_at_each_adaptation_stage(stage, domain):
    ratio = tuple(int(name == domain) for name in ("rgb", "saga", "ir_gan", "ir_real"))
    config = TrainingConfig(
        workflow="gan_bridge", total_iters=10,
        loss=LossConfig(p2_distill_weight=.3, p3_sup_weight=.8),
        gan_bridge=GANBridgeConfig(checkpoint="fixture.pth", source_sup_weight=.65),
        bridge_schedule=BridgeScheduleConfig(phase_ratios=((1, 0, 0, 0),) + (ratio,) * 4),
    )
    scheduler = BridgeDomainScheduler(config.bridge_schedule, config.total_iters)
    for step in range(stage * 2):
        scheduler.get_next_domain(step)
    plan = policy.resolve_step(config, scheduler, stage * 2, ema_initialized=True)

    teacher = "rgb" if domain == "rgb" or (domain == "saga" and stage == 1) else "ir"
    assert plan.step_name == f"bridge_{domain}"
    assert plan.source_domain == ("ir" if domain == "ir_real" else "rgb")
    assert plan.appearance_domain == domain
    assert plan.use_gt is (domain != "ir_real")
    assert plan.ema_target == teacher
    assert tuple(name for name, _ in plan.distillation) == (teacher,)
    assert scheduler.state_dict()["next_step"] == stage * 2 + 1
    if domain == "saga":
        assert plan.student_alpha == (1, .75, .5, .25, .25)[stage]
        assert plan.teacher_alpha == (1, .75, .5, .5, .25)[stage]
    else:
        assert (plan.student_alpha, plan.teacher_alpha) == (None, None)
    assert plan.supervised_weight == (
        .65 if stage == 4 and domain != "ir_real" else (1, 1, .8, .8, 1)[stage]
    )
    assert plan.distill_weight == (0, .3, 1, 1, 1)[stage]
    settings = plan.distillation[0][1]
    expected_thresholds = {
        "rgb": ((.02, .35), (.04, .25), (.04, .25), (.06, .20)),
        "ir": ((.005, .50), (.015, .35), (.015, .35), (.030, .25)),
    }
    assert (settings.top_ratio, settings.min_hm) == expected_thresholds[teacher][stage - 1]
    metadata = plan.log_metadata(did_step=True, ema_initialized=True)
    assert metadata["domain_ratio"] == ratio
    assert metadata["bridge_stage"] == stage + 1
    assert metadata["teacher_names"] == teacher
    assert metadata["ema_target"] == teacher
    assert metadata["supervised_weight"] == (plan.supervised_weight if plan.use_gt else 0)
    assert plan.log_metadata(did_step=False, ema_initialized=True)["ema_target"] == "none"


@pytest.mark.parametrize("step,initialized,names", [
    (1, False, ()), (2, False, ("rgb", "ir")), (2, True, ()),
])
def test_teacher_initialization_obeys_start_step_and_resume(step, initialized, names):
    config = TrainingConfig(
        curriculum=CurriculumConfig(phase1_end=2, phase2_end=4, phase3_end=6),
    )
    plan = policy.resolve_step(config, CurriculumScheduler(config), step, ema_initialized=initialized)
    assert plan.initialize_teacher_names == names
    assert tuple(name for name, _ in plan.distillation) == (() if step == 1 else ("rgb",))


def test_disabled_teacher_cannot_receive_distillation_or_ema():
    config = TrainingConfig(
        teacher_mode="rgb", ema=EMAConfig(start_steps=0),
        curriculum=CurriculumConfig(
            phase1_end=0, phase2_end=10, phase3_end=20, phase2_rgb_sampling_ratio=0,
        ),
    )
    plan = policy.resolve_step(config, CurriculumScheduler(config), 0, ema_initialized=False)
    assert plan.initialize_teacher_names == ("rgb",)
    assert plan.source_domain == "rgb"
    assert plan.use_gt
    assert plan.distillation == ()
    assert plan.ema_target is None


def test_policy_uses_restored_scheduler_without_consuming_training_rng():
    config = TrainingConfig(
        workflow="gan_bridge", total_iters=100,
        gan_bridge=GANBridgeConfig(checkpoint="fixture.pth"),
    )
    scheduler = BridgeDomainScheduler(config.bridge_schedule, config.total_iters)
    for step in range(53):
        policy.resolve_step(config, scheduler, step, ema_initialized=step >= 20)
    resumed = BridgeDomainScheduler(config.bridge_schedule, config.total_iters)
    resumed.load_state_dict(scheduler.state_dict())
    rng = torch.random.get_rng_state().clone()
    expected = [policy.resolve_step(config, scheduler, step, ema_initialized=True) for step in range(53, 100)]
    actual = [policy.resolve_step(config, resumed, step, ema_initialized=True) for step in range(53, 100)]
    assert actual == expected
    assert torch.equal(rng, torch.random.get_rng_state())

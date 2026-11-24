"""Exercise multi-level consistency through the criterion and real FCOS path."""

from dataclasses import replace
from unittest import mock

import pytest
import torch

from config import CurriculumConfig, IntermediateDomainConfig, LossConfig, TrainingConfig
from loss.detection_criterion import DetectionLossCriterion
from models.detector_adapter import ObjectFeaturePair
from models.detector_wrapper import DetectorWrapper
from scheduler import CurriculumScheduler
from training.step_policy import resolve_step
from test_fcos_adapter import build_adapter


def feature_pair(*, empty=False):
    student, teacher = [], []
    for size in (8, 4, 2):
        s = torch.zeros(1, 2, size, size)
        t = torch.zeros_like(s)
        s[:, 0] = 1
        t[:, 1] = 1
        student.append(s.requires_grad_())
        teacher.append(t.requires_grad_())
    return ObjectFeaturePair(
        tuple(student), tuple(teacher),
        ({"boxes": torch.empty(0, 4) if empty else torch.tensor([[0., 0., 64., 64.]])},),
        (1 / 8, 1 / 16, 1 / 32), ("p3", "p4", "p5"),
    )


def test_criterion_averages_levels_and_backpropagates_only_to_student():
    pair = feature_pair()
    result = DetectionLossCriterion().color_consistency(pair, level_weights=(1, 1, 1))
    loss = result.losses["loss_color_consistency"]
    torch.testing.assert_close(loss, torch.tensor(1.))
    assert set(result.metrics) == {"color_loss_p3", "color_loss_p4", "color_loss_p5", "color_objects"}
    loss.backward()
    for student, teacher in zip(pair.student, pair.teacher):
        assert student.grad is not None and student.grad.abs().sum() > 0
        assert teacher.grad is None


def test_criterion_uses_normalized_level_weights_and_object_embeddings():
    pair = feature_pair()
    teacher = (pair.teacher[0], pair.student[1].detach() * 7, pair.teacher[2])
    result = DetectionLossCriterion().color_consistency(replace(pair, teacher=teacher), level_weights=(1, 2, 3))
    torch.testing.assert_close(result.losses["loss_color_consistency"], torch.tensor(2 / 3))


def test_no_valid_objects_produces_finite_zero_with_student_gradient():
    pair = feature_pair(empty=True)
    result = DetectionLossCriterion().color_consistency(pair, level_weights=(1, 1, 1))
    loss = result.losses["loss_color_consistency"]
    assert loss.item() == 0
    loss.backward()
    assert all(feature.grad is not None for feature in pair.student)
    assert all(feature.grad is None for feature in pair.teacher)


@pytest.mark.parametrize("kwargs", [
    {"color_consistency_weight": -1}, {"color_consistency_weight": float("nan")},
    {"color_consistency_levels": ("p3", "p3")}, {"color_consistency_levels": ("p6",)},
    {"color_consistency_level_weights": (1, 1)},
    {"color_consistency_level_weights": (0, 0, 0)},
    {"color_consistency_level_weights": (1, float("inf"), 1)},
])
def test_loss_config_rejects_invalid_consistency_settings(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)


def test_real_adapter_aligns_three_levels_and_rejects_different_samples():
    model = DetectorWrapper(build_adapter(), DetectionLossCriterion()).train()
    teacher = DetectorWrapper(build_adapter(), DetectionLossCriterion()).eval()
    images = torch.rand(1, 3, 40, 56)
    targets = [{"boxes": torch.tensor([[3., 4., 20., 25.]]), "labels": torch.tensor([0])}]
    student_output = model.raw(images, targets, sample_ids=("source",), resize_short_edge=64)
    with torch.no_grad():
        teacher_output = teacher.raw(images, sample_ids=("source",), resize_short_edge=64)
    pair = model.adapter.prepare_color_consistency(student_output, teacher_output, levels=("p3", "p4", "p5"))
    assert pair.spatial_scales == (1 / 8, 1 / 16, 1 / 32)
    assert len(pair.student) == 3
    assert pair.targets[0]["boxes"][0, 2] > targets[0]["boxes"][0, 2]
    result = model.color_consistency_from_outputs(student_output, teacher_output,
                                                 levels=pair.levels, level_weights=(1, 1, 1))
    result.losses["loss_color_consistency"].backward()
    assert model.adapter.detector.backbone.proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    with pytest.raises(ValueError, match="sample"):
        model.adapter.prepare_color_consistency(student_output, replace(teacher_output, sample_ids=("other",)), levels=pair.levels)
    with pytest.raises(ValueError, match="size"):
        model.adapter.prepare_color_consistency(
            student_output, replace(teacher_output, context=replace(teacher_output.context, original_image_sizes=((20, 28),))),
            levels=pair.levels,
        )


@pytest.mark.parametrize("step,ratio,teacher_mode,weight,enabled", [
    (0, 1., "two_teacher", .1, False),
    (2, 1., "two_teacher", .1, True),
    (2, 0., "two_teacher", .1, True),
    (4, 0., "two_teacher", .1, False),
    (6, 1., "two_teacher", .1, False),
    (2, 1., "ir", .1, False),
    (2, 1., "two_teacher", 0., False),
])
def test_step_policy_only_enables_color_on_source_appearance_routes(step, ratio, teacher_mode, weight, enabled):
    config = TrainingConfig(teacher_mode=teacher_mode, loss=LossConfig(color_consistency_weight=weight),
                            curriculum=CurriculumConfig(phase1_end=2, phase2_end=4, phase3_end=6,
                                phase2_rgb_sampling_ratio=ratio, phase3_rgb_sampling_ratio=ratio))
    plan = resolve_step(config, CurriculumScheduler(config), step, ema_initialized=True)
    assert plan.color_consistency == enabled


@pytest.mark.parametrize("teacher_level,weight,forward_count", [("rgb", .1, 1), ("mid", .1, 2), ("rgb", 0., 1)])
def test_trainer_uses_pure_rgb_reference_and_can_disable_color_loss(teacher_level, weight, forward_count):
    from test_curriculum_wrapper_integration import build_config, build_trainer
    config = build_config()
    config.intermediate_domain = IntermediateDomainConfig(texture_strength=0)
    config.ema.start_steps = 0
    config.mid_routing.p2_rgb_flow.student_saga_level = "high"
    config.mid_routing.p2_rgb_flow.teacher_saga_level = teacher_level
    config.loss.color_consistency_weight = weight
    config.curriculum.phase2_rgb_sampling_ratio = 1
    trainer, _, _, _ = build_trainer(config)
    student = DetectorWrapper(build_adapter(), DetectionLossCriterion())
    from models.fcos_models import build_fcos_trio
    with mock.patch("models.fcos_models.build_fcos_model", return_value=student):
        trainer.student, trainer.rgb_teacher, trainer.ir_teacher = build_fcos_trio(config)
    trainer.optimizer = torch.optim.SGD(student.parameters(), lr=.001)
    from training.data_flow import RouteDataFlow
    images = torch.rand(1, 3, 32, 48)
    targets = [{"boxes": torch.tensor([[3., 4., 20., 25.]]), "labels": torch.tensor([0]), "stem": "source"}]
    trainer.data_flow = RouteDataFlow([(images, targets, ("source",))], None, device="cpu",
                                      augmentor=trainer.data_flow.augmentor)
    from models import fcos_models
    with mock.patch("models.fcos_models.forward_teacher", wraps=fcos_models.forward_teacher) as forward:
        logs = trainer.train_one_iteration()
    assert forward.call_count == forward_count
    torch.testing.assert_close(forward.call_args.args[1], images)
    if weight == 0:
        assert "loss_color_consistency" not in logs
        return
    assert logs["loss_color_consistency"] > 0
    for level in ("p3", "p4", "p5"):
        assert logs[f"color_loss_{level}"] >= 0
    assert logs["loss_color_consistency"] == pytest.approx(.1 * sum(logs[f"color_loss_{level}"] for level in ("p3", "p4", "p5")) / 3)


def test_cli_and_notebook_forward_color_configuration():
    from example_flir import build_training_config
    from notebook_support import make_training_args
    args = make_training_args(dict(data_root="unused", output_dir="out", color_consistency_weight=.2,
                                  color_consistency_levels=("p3", "p4", "p5"), color_consistency_level_weights=(1, 2, 1)))
    loss = build_training_config(args).loss
    assert loss.color_consistency_weight == .2
    assert loss.color_consistency_levels == ("p3", "p4", "p5")
    assert loss.color_consistency_level_weights == (1, 2, 1)


@pytest.mark.parametrize("domain,enabled", [("rgb", False), ("saga", True), ("ir_gan", False), ("ir_real", False)])
def test_gan_bridge_only_applies_color_to_the_appearance_bridge(domain, enabled):
    from config import BridgeScheduleConfig, GANBridgeConfig
    from scheduler import BridgeDomainScheduler
    ratios = tuple(int(name == domain) for name in ("rgb", "saga", "ir_gan", "ir_real"))
    config = TrainingConfig(workflow="gan_bridge", gan_bridge=GANBridgeConfig(checkpoint="fixture.pth"),
                            total_iters=100, bridge_schedule=BridgeScheduleConfig(phase_ratios=((1, 0, 0, 0),) + (ratios,) * 4))
    scheduler = BridgeDomainScheduler(config.bridge_schedule, config.total_iters)
    for step in range(81):
        plan = resolve_step(config, scheduler, step, ema_initialized=step >= 20)
    assert plan.color_consistency == enabled

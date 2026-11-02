"""RGB-to-IR transfer happens once, before phase-two forwards and teacher use."""

import copy
import tempfile
from pathlib import Path

import pytest
import torch

from config import CurriculumConfig
from models.domain_residual_neck import DomainResidualNeck
from test_domain_neck_training import make_trainer


def branch_state(neck, domain):
    return {
        name: value.detach().clone()
        for name, value in neck.state_dict().items()
        if f".{domain}_adapter." in name or f".{domain}_se." in name
    }


def assert_ir_matches(neck, rgb_state):
    actual = branch_state(neck, "ir")
    for name, expected in rgb_state.items():
        torch.testing.assert_close(actual[name.replace(".rgb_", ".ir_")], expected)


def fill_branch(neck, domain, value):
    with torch.no_grad():
        for name, parameter in neck.named_parameters():
            if f".{domain}_adapter." in name or f".{domain}_se." in name:
                parameter.fill_(value)


def build_transition_trainer(output_dir="outputs", ema_start=1):
    trainer = make_trainer(CurriculumConfig(
        phase1_end=1, phase2_end=4, phase3_end=5,
        phase2_rgb_sampling_ratio=1.0,
    ), output_dir)
    trainer.config.ema.start_steps = ema_start
    fill_branch(trainer.student.adapter.domain_neck, "rgb", 0.2)
    fill_branch(trainer.student.adapter.domain_neck, "ir", 0.7)
    return trainer


def test_neck_transfer_copies_all_levels_without_aliasing_or_changing_shared_weights():
    source = DomainResidualNeck(4, 4, se_reduction=2)
    target = DomainResidualNeck(4, 4, se_reduction=2)
    fill_branch(source, "rgb", 0.2)
    fill_branch(source, "ir", 0.7)
    before = copy.deepcopy(target.state_dict())
    parameter_ids = {name: id(p) for name, p in target.named_parameters()}
    expected = branch_state(source, "rgb")

    target.initialize_ir_from_rgb(source)

    assert_ir_matches(target, expected)
    for name, parameter in target.named_parameters():
        assert id(parameter) == parameter_ids[name]
        if ".ir_adapter." not in name and ".ir_se." not in name:
            torch.testing.assert_close(parameter, before[name])
    fill_branch(source, "rgb", 0.9)
    assert_ir_matches(target, expected)


@pytest.mark.parametrize("ema_start", [1, 3])
def test_boundary_initializes_student_and_teachers_before_forward_only_once(ema_start):
    trainer = build_transition_trainer(ema_start=ema_start)
    neck = trainer.student.adapter.domain_neck
    initial_ir = branch_state(neck, "ir")
    trainer.train_one_iteration()
    for name, value in initial_ir.items():
        torch.testing.assert_close(neck.state_dict()[name], value)
    expected = branch_state(neck, "rgb")
    captured = []

    def check_before_forward(module, args):
        assert_ir_matches(neck, expected)
        for teacher in (trainer.rgb_teacher, trainer.ir_teacher):
            assert_ir_matches(teacher.adapter.domain_neck, expected)
        captured.append(True)

    hook = trainer.student.adapter.register_forward_pre_hook(check_before_forward)
    trainer.train_one_iteration()
    hook.remove()
    assert captured == [True]
    assert trainer.ir_neck_initialized
    fill_branch(neck, "rgb", 0.4)
    trainer.train_one_iteration()
    assert_ir_matches(neck, expected)


@pytest.mark.parametrize("teacher_mode", ["rgb", "ir", "two_teacher"])
def test_early_ema_keeps_teacher_rgb_and_shared_weights_when_initializing_ir(teacher_mode):
    trainer = build_transition_trainer(ema_start=0)
    trainer.config.teacher_mode = teacher_mode
    if teacher_mode == "rgb":
        trainer.ir_teacher = None
    elif teacher_mode == "ir":
        trainer.rgb_teacher = None
    trainer.train_one_iteration()
    expected = branch_state(trainer.student.adapter.domain_neck, "rgb")
    active = [t for t in (trainer.rgb_teacher, trainer.ir_teacher) if t is not None]
    before = [copy.deepcopy(t.state_dict()) for t in active]
    captured = []

    def check_before_forward(module, args):
        for teacher, old_state in zip(active, before):
            assert_ir_matches(teacher.adapter.domain_neck, expected)
            for name, value in teacher.state_dict().items():
                if ".ir_adapter." not in name and ".ir_se." not in name:
                    torch.testing.assert_close(value, old_state[name])
            assert all(not p.requires_grad for p in teacher.parameters())
        captured.append(True)

    hook = trainer.student.adapter.register_forward_pre_hook(check_before_forward)
    trainer.train_one_iteration()
    hook.remove()
    assert captured == [True]


def test_resume_after_transfer_preserves_learned_ir_instead_of_copying_again():
    with tempfile.TemporaryDirectory() as directory:
        trainer = build_transition_trainer(directory)
        trainer.train_one_iteration()
        trainer.train_one_iteration()
        fill_branch(trainer.student.adapter.domain_neck, "ir", 0.6)
        expected = branch_state(trainer.student.adapter.domain_neck, "ir")
        trainer.save_checkpoint("warmstart.pth")
        resumed = build_transition_trainer(directory)
        resumed.load_checkpoint(str(Path(directory) / "warmstart.pth"))
        assert resumed.ir_neck_initialized
        resumed.train_one_iteration()
        for name, value in expected.items():
            torch.testing.assert_close(resumed.student.adapter.domain_neck.state_dict()[name], value)


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_at_boundary_still_performs_pending_transfer(legacy):
    with tempfile.TemporaryDirectory() as directory:
        trainer = build_transition_trainer(directory)
        trainer.train_one_iteration()
        expected = branch_state(trainer.student.adapter.domain_neck, "rgb")
        trainer.save_checkpoint("phase1.pth")
        path = Path(directory) / "phase1.pth"
        if legacy:
            checkpoint = torch.load(path, weights_only=False)
            checkpoint.pop("ir_neck_initialized", None)
            torch.save(checkpoint, path)
        resumed = build_transition_trainer(directory)
        resumed.load_checkpoint(str(path))
        resumed.train_one_iteration()
        assert_ir_matches(resumed.student.adapter.domain_neck, expected)


def test_legacy_checkpoint_after_boundary_is_rejected_before_loading_weights():
    with tempfile.TemporaryDirectory() as directory:
        trainer = build_transition_trainer(directory)
        trainer.train_one_iteration()
        trainer.train_one_iteration()
        trainer.save_checkpoint("old.pth")
        path = Path(directory) / "old.pth"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint.pop("ir_neck_initialized", None)
        torch.save(checkpoint, path)
        resumed = build_transition_trainer(directory)
        before = copy.deepcopy(resumed.student.state_dict())
        with pytest.raises(ValueError, match="warmstart"):
            resumed.load_checkpoint(str(path))
        for name, value in resumed.student.state_dict().items():
            torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("mode", ["neck_disabled", "rgb_baseline"])
def test_transfer_does_not_run_outside_enabled_neck_curriculum(mode):
    trainer = build_transition_trainer()
    trainer.global_step = 1
    if mode == "neck_disabled":
        trainer.config.model.domain_neck_enabled = False
    else:
        trainer.config.workflow = "rgb_baseline"
    expected = branch_state(trainer.student.adapter.domain_neck, "ir")
    trainer.train_one_iteration()
    for name, value in expected.items():
        torch.testing.assert_close(trainer.student.adapter.domain_neck.state_dict()[name], value)

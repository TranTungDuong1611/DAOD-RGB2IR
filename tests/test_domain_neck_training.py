from collections import OrderedDict
import copy
from pathlib import Path
import tempfile

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import CurriculumConfig, EMAConfig
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_adapter import AdapterOutput, DetectorAdapter, DistillationPair, Predictions, SupervisedBatch
from models.d3t_wrapper import D3TWrapper
from models.domain_residual_neck import DomainResidualNeck
from trainer import CurriculumDomainAdaptationTrainer
from test_curriculum_wrapper_integration import build_config, RGBDataset, IRDataset, collate_rgb, collate_ir


class SmallDomainAdapter(DetectorAdapter):
    """Real neck and loss graph with a minimal detector-neutral prediction head."""
    def __init__(self):
        super().__init__()
        self.domain_neck = DomainResidualNeck(4, 4, se_reduction=2)
        self.domains = []
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.1)

    def forward(self, images, targets=None, sample_ids=None, domain="rgb"):
        self.domains.append(domain)
        batch = torch.stack(tuple(images))
        feature = batch.mean((1, 2, 3)).reshape(-1, 1, 1, 1).expand(-1, 4, 2, 2)
        features = OrderedDict((str(i), feature) for i in range(5))
        output = self.domain_neck(features, domain=domain)
        values = torch.stack([f.mean((1, 2, 3)) for f in output.values()]).mean(0)
        predictions = tuple(Predictions(
            value.expand(2, 3),
            torch.stack((value, value, value + 10, value + 10)).expand(2, 4),
            value.expand(2),
        ) for value in values)
        return AdapterOutput(predictions, sample_ids=sample_ids)

    def prepare_supervised(self, output, targets):
        aligned = tuple({
            "class_targets": torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            "box_targets": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
            "quality_targets": torch.ones(2),
            "foreground": torch.ones(2, dtype=torch.bool),
        } for _ in output.predictions)
        return SupervisedBatch(output.predictions, aligned)

    def prepare_distillation(self, student, teacher):
        assert student.sample_ids == teacher.sample_ids
        return DistillationPair(student.predictions, teacher.predictions)

    def postprocess(self, output):
        return []


def make_trainer(curriculum, output_dir="outputs"):
    config = build_config(curriculum=curriculum, ema=EMAConfig(alpha=0.0, start_steps=0),
                          output_dir=output_dir)
    config.model.domain_neck_enabled = True
    config.model.domain_neck_bottleneck_channels = 4
    config.model.domain_neck_se_reduction = 2
    student = D3TWrapper(SmallDomainAdapter(), D3TLossCriterion())
    rgb, ir = copy.deepcopy(student), copy.deepcopy(student)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    return CurriculumDomainAdaptationTrainer(
        student, rgb, ir, optimizer, config,
        DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb),
        DataLoader(IRDataset(), batch_size=1, collate_fn=collate_ir),
    )


def assert_branch_updated(trainer, before, selected, inactive):
    after = dict(trainer.student.adapter.domain_neck.named_parameters())
    for kind in ("adapter", "se"):
        selected_names = [name for name in after if f".{selected}_{kind}." in name]
        inactive_names = [name for name in after if f".{inactive}_{kind}." in name]
        assert any(not torch.equal(before[name], after[name]) for name in selected_names)
        assert all(torch.equal(before[name], after[name]) for name in inactive_names)
        assert all(after[name].grad is None for name in inactive_names)
    assert trainer.student.adapter.domain_neck.blocks[0].conv2.weight.grad.abs().sum() > 0


def test_rgb_supervision_updates_only_rgb_attention_and_shared_convolutions():
    trainer = make_trainer(CurriculumConfig(phase1_end=2, phase2_end=3, phase3_end=4))
    before = {name: p.detach().clone() for name, p in trainer.student.adapter.domain_neck.named_parameters()}
    logs = trainer.train_one_iteration()
    assert "sup_loss_cls" in logs
    assert_branch_updated(trainer, before, "rgb", "ir")


@pytest.mark.parametrize("phase", [3, 4])
def test_real_ir_kd_updates_only_ir_branch_and_ema_includes_attention(phase):
    curriculum = CurriculumConfig(phase1_end=0, phase2_end=0,
        phase3_end=4 if phase == 3 else 0, phase3_rgb_sampling_ratio=0.0)
    trainer = make_trainer(curriculum)
    trainer._ensure_ema_initialized()
    with torch.no_grad():
        trainer.ir_teacher.adapter.domain_neck.blocks[0].ir_se.se[3].bias.add_(1.0)
    before = {name: p.detach().clone() for name, p in trainer.student.adapter.domain_neck.named_parameters()}
    logs = trainer.train_one_iteration()
    assert trainer.student.adapter.domains[-1] == "ir"
    assert trainer.ir_teacher.adapter.domains[-1] == "ir"
    assert not any(name.startswith("sup_") for name in logs)
    assert logs["loss_kd_quality_ir"] > 0
    assert_branch_updated(trainer, before, "ir", "rgb")
    for name, parameter in trainer.ir_teacher.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, dict(trainer.student.named_parameters())[name])


def test_saga_high_uses_rgb_attention_even_with_ir_named_teacher():
    trainer = make_trainer(CurriculumConfig(phase1_end=0, phase2_end=3,
        phase3_end=4, phase2_rgb_sampling_ratio=0.0))
    data = trainer._prepare_data_by_route("p2_ir_flow", trainer.config.mid_routing.p2_ir_flow)
    assert data["student_domain"] == data["teacher_domain"] == "rgb"
    trainer.train_one_iteration()
    assert trainer.student.adapter.domains[-1] == "rgb"
    assert trainer.ir_teacher.adapter.domains[-1] == "rgb"


@pytest.fixture
def checkpoint_dir():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_resume_restores_both_domain_branches_and_optimizer(checkpoint_dir):
    curriculum = CurriculumConfig(phase1_end=0, phase2_end=0, phase3_end=0)
    trainer = make_trainer(curriculum, str(checkpoint_dir))
    trainer.train_one_iteration()
    trainer.save_checkpoint("neck.pth")
    resumed = make_trainer(curriculum, str(checkpoint_dir))
    resumed.load_checkpoint(str(checkpoint_dir / "neck.pth"))
    assert resumed.global_step == trainer.global_step
    assert resumed.ema_initialized
    assert resumed.optimizer.state_dict()["state"].keys() == trainer.optimizer.state_dict()["state"].keys()
    for name, parameter in resumed.student.named_parameters():
        torch.testing.assert_close(parameter, dict(trainer.student.named_parameters())[name])
    resumed.train_one_iteration()
    assert resumed.student.adapter.domains[-1] == "ir"

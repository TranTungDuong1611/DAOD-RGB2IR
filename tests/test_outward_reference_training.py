"""Exercise auxiliary supervision through the real FCOS adapter and trainer."""

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from config import CurriculumConfig, EMAConfig
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_wrapper import D3TWrapper
from models.domain_residual_neck import DomainResidualNeck
from models.torchvision_fcos_adapter import TorchvisionFCOSAdapter
from trainer import CurriculumDomainAdaptationTrainer
from test_curriculum_wrapper_integration import build_config, RGBDataset, IRDataset, collate_rgb, collate_ir
from test_torchvision_fcos_adapter import build_adapter


def make_outref_trainer(curriculum, output_dir="outputs", threshold=0.):
    torch.manual_seed(13)
    config = build_config(curriculum=curriculum, ema=EMAConfig(alpha=.5, start_steps=0),
                          output_dir=output_dir)
    config.model.domain_neck_enabled = config.model.out_ref_enabled = True
    config.model.domain_neck_bottleneck_channels = 8
    config.model.out_ref_channels = 4
    config.loss.out_ref_teacher_score_thresh = threshold
    neck = DomainResidualNeck(64, 8, out_ref_enabled=True, out_ref_channels=4)
    student = D3TWrapper(TorchvisionFCOSAdapter(build_adapter().detector, domain_neck=neck),
                         D3TLossCriterion())
    with torch.no_grad():
        student.adapter.detector.head.regression_head.bbox_reg.bias.fill_(1.)
    optimizer = torch.optim.SGD(student.parameters(), lr=.001, momentum=.9)
    return CurriculumDomainAdaptationTrainer(
        student, copy.deepcopy(student), copy.deepcopy(student), optimizer, config,
        DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb),
        DataLoader(IRDataset(), batch_size=1, collate_fn=collate_ir),
    )


@pytest.mark.parametrize("route", ["rgb", "saga", "ir"])
def test_outref_loss_updates_gradient_predictor_on_each_route_with_one_forward(route):
    curriculum = CurriculumConfig(
        phase1_end=1 if route == "rgb" else 0,
        phase2_end=2 if route != "ir" else 0, phase3_end=3,
        phase2_rgb_sampling_ratio=0., phase3_rgb_sampling_ratio=0.,
    )
    trainer = make_outref_trainer(curriculum)
    counts = {"student": 0, "teacher": 0}
    def count(name):
        def hook(*args):
            counts[name] += 1
        return hook
    handles = [trainer.student.adapter.detector.backbone.register_forward_hook(count("student")),
               trainer.ir_teacher.adapter.detector.backbone.register_forward_hook(count("teacher"))]
    predictor = trainer.student.adapter.domain_neck.blocks[0].out_ref.gdt_convs_pred
    before = predictor.weight.detach().clone()
    try:
        log = trainer.train_one_iteration()
    finally:
        for handle in handles:
            handle.remove()
    assert log["loss_out_ref"] > 0
    assert counts == {"student": 1, "teacher": 0 if route == "rgb" else 1}
    assert predictor.weight.grad.abs().sum() > 0
    assert not torch.equal(before, predictor.weight)
    expected_domain = "rgb" if route == "rgb" else "ir"
    neck = trainer.student.adapter.domain_neck
    selected = getattr(neck.blocks[0], f"{expected_domain}_adapter")
    inactive = getattr(neck.blocks[0], f"{'ir' if route == 'rgb' else 'rgb'}_adapter")
    assert selected.weight.grad.abs().sum() > 0 and inactive.weight.grad is None
    assert ("sup_loss_cls" in log) == (route != "ir")
    assert all(p.grad is None for teacher in (trainer.rgb_teacher, trainer.ir_teacher)
               for p in teacher.parameters())


def test_real_ir_without_confident_teacher_boxes_has_zero_auxiliary_gradient():
    trainer = make_outref_trainer(CurriculumConfig(phase1_end=0, phase2_end=0,
        phase3_end=3, phase3_rgb_sampling_ratio=0.), threshold=1.)
    log = trainer.train_one_iteration()
    assert log["loss_out_ref"] == 0
    predictor = trainer.student.adapter.domain_neck.blocks[0].out_ref.gdt_convs_pred
    assert predictor.weight.grad.count_nonzero() == 0
    assert any(name.startswith("loss_kd_") for name in log)


def test_zero_auxiliary_weight_does_not_update_edge_predictor():
    trainer = make_outref_trainer(CurriculumConfig(phase1_end=1, phase2_end=2, phase3_end=3))
    trainer.config.loss.weight_out_ref = 0
    log = trainer.train_one_iteration()
    assert "loss_out_ref" not in log
    assert trainer.student.adapter.domain_neck.blocks[0].out_ref.gdt_convs_pred.weight.grad is None

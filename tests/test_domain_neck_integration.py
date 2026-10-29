import copy

import pytest
import torch

from config import FCOSModelConfig, TrainingConfig
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_wrapper import D3TWrapper
from models.domain_residual_neck import DomainResidualNeck
from models.fcos_factory import build_checkpoint_metadata, build_fcos_d3t_trio, validate_checkpoint_metadata
from models.torchvision_fcos_adapter import TorchvisionFCOSAdapter
from test_torchvision_fcos_adapter import build_adapter
from torchvision.models.detection import fcos_resnet50_fpn


def test_domain_neck_changes_fpn_features_before_head_without_changing_layout():
    neck = DomainResidualNeck(channels=64, bottleneck_channels=8)
    with torch.no_grad():
        for parameter in neck.parameters():
            parameter.zero_()
        for block in neck.blocks:
            block.skip.weight[:, :, 0, 0].copy_(2 * torch.eye(64))
    adapter = TorchvisionFCOSAdapter(build_adapter().detector, domain_neck=neck)
    model = D3TWrapper(adapter, D3TLossCriterion()).eval()
    observed = {}
    def capture_backbone(module, args, output):
        observed["fpn"] = tuple(output.values())
    def capture_head(module, args):
        observed["head"] = args[0]
    handles = [adapter.detector.backbone.register_forward_hook(capture_backbone),
               adapter.detector.head.register_forward_pre_hook(capture_head)]
    try:
        output = model.raw([torch.rand(3, 64, 64)], sample_ids=("a",), domain="thermal")
    finally:
        for handle in handles:
            handle.remove()
    assert output.context.domain == "ir"
    assert len(output.context.features) == 5
    for original, actual, head in zip(observed["fpn"], output.context.features, observed["head"]):
        torch.testing.assert_close(actual, (2 * original).relu())
        assert head is actual
    assert sum(output.context.num_anchors_per_level) == len(output.predictions[0].boxes)


def test_no_neck_keeps_same_predictions_for_either_domain():
    model = D3TWrapper(build_adapter(), D3TLossCriterion()).eval()
    image = torch.rand(3, 64, 64)
    rgb = model.raw([image], domain="rgb")
    ir = model.raw([image], domain="ir")
    torch.testing.assert_close(rgb.predictions[0].class_logits, ir.predictions[0].class_logits)
    torch.testing.assert_close(rgb.predictions[0].boxes, ir.predictions[0].boxes)
    with pytest.raises(ValueError, match="domain"):
        model.raw([image], domain="weak")


def test_factory_copies_enabled_neck_to_independent_frozen_teachers():
    config = TrainingConfig(model=FCOSModelConfig(
        weights=None, pretrained_backbone=False, domain_neck_enabled=True,
        domain_neck_bottleneck_channels=8, domain_neck_se_reduction=8,
    ), device="cpu")
    base = fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    student, rgb, ir = build_fcos_d3t_trio(config, base_detector=base)
    for teacher in (rgb, ir):
        assert teacher.adapter.domain_neck is not student.adapter.domain_neck
        assert all(not p.requires_grad for p in teacher.adapter.domain_neck.parameters())
        assert not teacher.training
        for a, b in zip(student.adapter.domain_neck.parameters(), teacher.adapter.domain_neck.parameters()):
            torch.testing.assert_close(a, b)
            assert a.data_ptr() != b.data_ptr()


def test_checkpoint_validates_neck_architecture_and_legacy_disabled_metadata():
    baseline = TrainingConfig(model=FCOSModelConfig(weights=None, pretrained_backbone=False))
    legacy = build_checkpoint_metadata(baseline)
    legacy.pop("domain_neck", None)
    for key in ("domain_neck_enabled", "domain_neck_bottleneck_channels", "domain_neck_se_reduction"):
        legacy["config"]["model"].pop(key, None)
    validate_checkpoint_metadata(legacy, baseline)
    enabled = copy.deepcopy(baseline)
    enabled.model.domain_neck_enabled = True
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(legacy, enabled)
    metadata = build_checkpoint_metadata(enabled)
    validate_checkpoint_metadata(metadata, enabled)
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(metadata, baseline)
    changed = copy.deepcopy(enabled)
    changed.model.domain_neck_bottleneck_channels = 32
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(metadata, changed)
    changed = copy.deepcopy(enabled)
    changed.model.domain_neck_se_reduction = 8
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(metadata, changed)


@pytest.mark.parametrize("kwargs", [
    {"domain_neck_bottleneck_channels": 0}, {"domain_neck_se_reduction": 0},
    {"domain_neck_enabled": "yes"},
])
def test_config_rejects_invalid_neck_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FCOSModelConfig(**kwargs)

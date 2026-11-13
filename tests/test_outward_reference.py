"""OutRef retains the BiRefNet feature/prediction/attention split."""

import copy

import pytest
import torch

from config import FCOSModelConfig, TrainingConfig
from models.domain_residual_neck import DomainBottleneck, DomainResidualNeck
from models.fcos_factory import build_checkpoint_metadata, validate_checkpoint_metadata
from models.torchvision_fcos_adapter import TorchvisionFCOSAdapter
from test_torchvision_fcos_adapter import build_adapter


def test_outref_starts_neutral_and_has_separate_gradient_and_attention_heads():
    block = DomainBottleneck(4, 4, 4, out_ref_enabled=True,
                             out_ref_channels=4, out_ref_batch_norm=False)
    plain = DomainBottleneck(4, 4, 4)
    plain.load_state_dict({k: v for k, v in block.state_dict().items()
                           if not k.startswith("out_ref.")})
    x = torch.randn(2, 4, 5, 7)
    output, edge = block.forward_with_aux(x, "ir")
    torch.testing.assert_close(output, plain(x, "ir"))
    assert edge.shape == (2, 1, 5, 7)
    edge.square().mean().backward()
    assert block.out_ref.gdt_convs[0].weight.grad.abs().sum() > 0
    assert block.out_ref.gdt_convs_pred.weight.grad.abs().sum() > 0
    assert block.out_ref.gdt_convs_attn.weight.grad is None
    assert all(p.grad is None for p in block.ir_se.parameters())


def test_attention_scales_bottleneck_but_preserves_projected_skip():
    block = DomainBottleneck(1, 1, 1, reduction=1, out_ref_enabled=True,
                             out_ref_channels=1, out_ref_batch_norm=False)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
        block.conv1.weight.fill_(1)
        block.conv2.bias.fill_(1)
        block.rgb_adapter.weight.fill_(2)
        block.conv3.weight.fill_(3)
        block.skip.weight.fill_(4)
        block.out_ref.gdt_convs_attn.bias.fill_(torch.log(torch.tensor(3.)).item())
    # z=15, SE(z)=7.5, spatial gain=1.5, skip=8.
    torch.testing.assert_close(block(torch.full((1, 1, 3, 3), 2.), "rgb"),
                               torch.full((1, 1, 3, 3), 19.25))


def test_neck_keeps_outref_on_first_two_levels_and_skips_predictor_in_eval():
    neck = DomainResidualNeck(4, 4, out_ref_enabled=True, out_ref_batch_norm=False)
    features = {str(i): torch.ones(2, 4, 8 // (2 ** min(i, 2)), 8 // (2 ** min(i, 2)))
                for i in range(5)}
    output, logits = neck.forward_with_aux(features, domain="rgb")
    assert list(output) == list(features)
    assert len(logits) == 2
    for i, tensor in enumerate(logits):
        assert tensor.shape == (2, 1, *features[str(i)].shape[-2:])
    assert all(block.out_ref is None for block in neck.blocks[2:])
    def unexpected_predictor(*args):
        raise AssertionError("gradient predictor must not run during evaluation")
    handle = neck.blocks[0].out_ref.gdt_convs_pred.register_forward_pre_hook(unexpected_predictor)
    try:
        neck.eval()
        _, logits = neck.forward_with_aux(features, domain="ir")
        assert logits == ()
    finally:
        handle.remove()


def test_adapter_exposes_training_logits_in_each_output_not_module_state():
    neck = DomainResidualNeck(64, 8, out_ref_enabled=True, out_ref_batch_norm=False)
    adapter = TorchvisionFCOSAdapter(build_adapter().detector, domain_neck=neck).train()
    first = adapter([torch.rand(3, 64, 64)], sample_ids=("a",))
    second = adapter([torch.rand(3, 64, 64)], sample_ids=("b",), domain="ir")
    assert len(first.context.out_ref_logits) == len(second.context.out_ref_logits) == 2
    assert first.context.out_ref_logits[0] is not second.context.out_ref_logits[0]
    adapter.eval()
    assert adapter([torch.rand(3, 64, 64)]).context.out_ref_logits == ()


def test_checkpoint_rejects_changed_outref_architecture_and_accepts_old_neck():
    config = TrainingConfig(model=FCOSModelConfig(domain_neck_enabled=True))
    legacy = build_checkpoint_metadata(config)
    config.model.out_ref_enabled = True
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(legacy, config)
    metadata = build_checkpoint_metadata(config)
    validate_checkpoint_metadata(metadata, config)
    changed = copy.deepcopy(config)
    changed.model.out_ref_channels = 8
    with pytest.raises(ValueError, match="neck"):
        validate_checkpoint_metadata(metadata, changed)


@pytest.mark.parametrize("kwargs", [
    {"out_ref_enabled": True},
    {"out_ref_channels": 0},
    {"out_ref_enabled": "yes"},
    {"out_ref_batch_norm": "yes"},
])
def test_outref_config_rejects_invalid_architecture(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FCOSModelConfig(**kwargs)

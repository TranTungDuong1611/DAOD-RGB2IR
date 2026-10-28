from collections import OrderedDict
import math

import pytest
import torch

from models.domain_residual_neck import DomainBottleneck, DomainResidualNeck, SEBlock


def test_se_scales_channels_without_mixing_spatial_positions():
    block = SEBlock(2, reduction=2)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        block.se[3].bias[1] = math.log(3.0)
    feature = torch.tensor([[[[2.0, 4.0]], [[4.0, 8.0]]]])
    torch.testing.assert_close(
        block(feature), torch.tensor([[[[1.0, 2.0]], [[3.0, 6.0]]]])
    )


@pytest.mark.parametrize("domain, expected", [("rgb", 15.5), ("ir", 12.5), ("thermal", 12.5)])
def test_bottleneck_adds_parallel_adapter_then_se_then_projected_skip(domain, expected):
    block = DomainBottleneck(1, 1, 1, reduction=1)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        block.conv1.weight.fill_(1.0)
        block.conv2.bias.fill_(1.0)
        block.rgb_adapter.weight.fill_(2.0)
        block.ir_adapter.weight.fill_(1.0)
        block.conv3.weight.fill_(3.0)
        block.skip.weight.fill_(4.0)
    output = block(torch.full((1, 1, 3, 3), 2.0), domain)
    torch.testing.assert_close(output, torch.full((1, 1, 3, 3), expected))


@pytest.mark.parametrize("domain, selected, inactive", [("rgb", "rgb", "ir"), ("ir", "ir", "rgb")])
def test_only_selected_domain_branch_receives_gradients(domain, selected, inactive):
    block = DomainBottleneck(4, 4, 4, reduction=2)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(0.05)
    block(torch.ones(2, 4, 4, 4), domain).square().mean().backward()
    for name, parameter in block.named_parameters():
        if name.startswith((f"{inactive}_adapter.", f"{inactive}_se.")):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_neck_preserves_five_level_shapes_and_order():
    features = OrderedDict(
        (name, torch.randn(2, 8, height, width))
        for name, height, width in [("0", 9, 11), ("1", 5, 6), ("2", 3, 3), ("p6", 2, 2), ("p7", 1, 1)]
    )
    neck = DomainResidualNeck(channels=8, bottleneck_channels=4, se_reduction=2)
    output = neck(features, domain="ir")
    assert list(output) == list(features)
    for name in features:
        assert output[name].shape == features[name].shape
        assert (output[name] >= 0).all()
    output["0"].sum().backward()
    assert neck.blocks[0].conv1.weight.grad is not None
    assert neck.blocks[1].conv1.weight.grad is None


def test_neck_rejects_invalid_domain_and_level_count():
    neck = DomainResidualNeck(channels=8, bottleneck_channels=4)
    with pytest.raises(ValueError, match="domain"):
        neck({}, domain="saga")
    with pytest.raises(ValueError, match="levels"):
        neck({}, domain="rgb")


@pytest.mark.parametrize("kwargs", [{"channels": 0}, {"bottleneck_channels": 0}, {"num_levels": 0}, {"se_reduction": 0}])
def test_neck_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        DomainResidualNeck(**kwargs)

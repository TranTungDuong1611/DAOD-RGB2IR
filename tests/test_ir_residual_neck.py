from collections import OrderedDict

import pytest
import torch

from models.ir_residual_neck import IRResidualNeck


def _features(levels: int = 5):
    return OrderedDict(
        (
            str(index),
            torch.randn(
                2,
                256,
                8 // (2 ** min(index, 2)),
                8 // (2 ** min(index, 2)),
            ),
        )
        for index in range(levels)
    )


def test_non_ir_domain_is_an_exact_bypass() -> None:
    neck = IRResidualNeck()
    features = _features()

    output = neck(features, is_ir=False)

    assert output is features


def test_ir_domain_preserves_fpn_order_shapes_and_initial_values() -> None:
    neck = IRResidualNeck()
    features = _features()

    output = neck(features, is_ir=True)

    assert list(output) == list(features)
    for name in features:
        assert output[name].shape == features[name].shape
        torch.testing.assert_close(output[name], features[name])


def test_ir_residual_scales_receive_gradients_from_every_level() -> None:
    torch.manual_seed(0)
    neck = IRResidualNeck()
    output = neck(_features(), is_ir=True)

    sum(feature.square().mean() for feature in output.values()).backward()

    for adapter in neck.adapters:
        assert adapter.scale.grad is not None
        assert torch.count_nonzero(adapter.scale.grad).item() == 1


def test_ir_domain_rejects_an_unexpected_number_of_fpn_levels() -> None:
    neck = IRResidualNeck(num_levels=5)

    with pytest.raises(ValueError, match="expected 5 FPN levels"):
        neck(_features(levels=3), is_ir=True)

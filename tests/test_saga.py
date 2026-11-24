"""SAGA preserves image correspondence and changes only annotated objects."""

import pytest
import torch

from saga import SoftSAGA


def test_batch_boxes_must_not_silently_drop_images():
    images = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError, match="one box tensor per image"):
        SoftSAGA().apply_to_batch(images, [torch.empty(0, 4)], alpha=.5)


def test_soft_blend_preserves_background_input_and_gradients():
    image = torch.zeros(3, 3, 3)
    image[0] = 1.
    image.requires_grad_()
    boxes = torch.tensor([[-1., -1., 2., 2.]])
    result = SoftSAGA().apply(image, boxes, alpha=.5)
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([.6495, .1495, .1495]))
    torch.testing.assert_close(result[:, 2, 2], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([1., 0., 0.]))
    result.sum().backward()
    assert torch.isfinite(image.grad).all()
    torch.testing.assert_close(SoftSAGA().apply(image, torch.empty(0, 4), alpha=.5), image)

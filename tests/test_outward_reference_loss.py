"""Gradient supervision uses actual student pixels and trusted object regions."""

from types import SimpleNamespace

import pytest
import torch

from config import LossConfig
from loss.outward_reference import build_gradient_targets, outward_reference_loss, reference_regions


def context(images, sizes, boxes=None, logits=None):
    return SimpleNamespace(
        transformed_images=SimpleNamespace(tensors=images),
        transformed_image_sizes=tuple(sizes),
        original_image_sizes=tuple((h // 2, w // 2) for h, w in sizes),
        transformed_targets=None if boxes is None else tuple({"boxes": b} for b in boxes),
        image_mean=(0., 0., 0.), image_std=(1., 1., 1.),
        out_ref_logits=(torch.zeros(len(sizes), 1, 3, 4, requires_grad=True),)
        if logits is None else logits,
    )


def test_laplacian_target_comes_from_pixels_and_ignores_batch_padding():
    images = torch.full((2, 3, 8, 10), 99., requires_grad=True)
    with torch.no_grad():
        images[0, :, :6, :8] = .5
        images[1, :, :6, :8] = 0
        images[1, :, :6, 4:8] = 1
    boxes = [torch.tensor([[0., 0., 10., 8.]], requires_grad=True)] * 2
    target, mask = build_gradient_targets(images, [(6, 8)] * 2, boxes)
    assert target.shape == mask.shape == (2, 1, 8, 10)
    assert target[0].count_nonzero() == 0
    assert target[1, 0, :, 2:6].sum() > 0
    assert target.min() >= 0 and target.max() == 1
    assert mask.sum() == 2 * 6 * 8
    assert not mask[:, :, 6:].any() and not mask[:, :, :, 8:].any()
    assert not target.requires_grad and not mask.requires_grad


def test_roi_masks_do_not_draw_rectangle_edges_and_clip_invalid_boxes():
    image = torch.full((1, 3, 5, 7), .5)
    target, mask = build_gradient_targets(image, [(5, 7)], [torch.tensor([
        [-2., 1., 3., 4.], [6., 4., 99., 99.], [3., 2., 2., 4.],
        [float("nan"), 0., 4., 4.],
    ])])
    assert target.count_nonzero() == 0
    assert mask.sum() == 10
    assert mask[0, 0, 1:4, :3].all() and mask[0, 0, 4, 6]


def test_loss_uses_denormalized_student_view_and_detaches_targets():
    raw = torch.zeros(1, 3, 8, 8)
    raw[:, :, :, 4:] = 1
    mean = torch.tensor([.1, .3, .5]).view(1, 3, 1, 1)
    std = torch.tensor([.2, .4, .8]).view(1, 3, 1, 1)
    normalized = ((raw - mean) / std).requires_grad_()
    boxes = [torch.tensor([[0., 0., 8., 8.]], requires_grad=True)]
    logits = (torch.randn(1, 1, 8, 8, requires_grad=True),)
    ctx = context(normalized, [(8, 8)], boxes, logits)
    ctx.image_mean, ctx.image_std = tuple(mean.flatten().tolist()), tuple(std.flatten().tolist())
    loss = outward_reference_loss(ctx)
    target, _ = build_gradient_targets(raw, [(8, 8)], boxes)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], target)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits[0].grad.abs().sum() > 0
    assert normalized.grad is None and boxes[0].grad is None


def test_loss_averages_levels_and_valid_images_without_outside_roi_gradients():
    logits = (torch.zeros(2, 1, 8, 8, requires_grad=True),
              torch.full((2, 1, 8, 8), 2., requires_grad=True))
    ctx = context(torch.zeros(2, 3, 8, 8), [(6, 6), (8, 8)],
                  [torch.tensor([[1., 1., 3., 3.]]), torch.empty(0, 4)], logits)
    loss = outward_reference_loss(ctx)
    torch.testing.assert_close(loss, (torch.nn.functional.softplus(torch.tensor(0.)) +
                                     torch.nn.functional.softplus(torch.tensor(2.))) / 2)
    loss.backward()
    for logit in logits:
        assert logit.grad[1].count_nonzero() == 0
        assert logit.grad[0].count_nonzero() == 4


def test_teacher_boxes_filtered_and_resized_but_gt_takes_precedence():
    ctx = context(torch.zeros(1, 3, 8, 12), [(8, 12)])
    detections = ([{"boxes": torch.tensor([[1., 1., 3., 3.], [0., 0., 1., 1.]], requires_grad=True),
                    "scores": torch.tensor([.9, .4])}],)
    regions = reference_regions(ctx, detections, score_threshold=.7)
    torch.testing.assert_close(regions[0], torch.tensor([[2., 2., 6., 6.]]))
    assert not regions[0].requires_grad
    ctx.transformed_targets = ({"boxes": torch.tensor([[1., 1., 2., 2.]])},)
    torch.testing.assert_close(reference_regions(ctx, detections)[0], ctx.transformed_targets[0]["boxes"])


def test_no_confident_boxes_skips_loss_with_connected_zero():
    ctx = context(torch.rand(1, 3, 8, 8), [(8, 8)])
    loss = outward_reference_loss(ctx)
    assert loss.item() == 0 and torch.isfinite(loss)
    loss.backward()
    assert ctx.out_ref_logits[0].grad.count_nonzero() == 0


@pytest.mark.parametrize("kwargs", [
    {"weight_out_ref": -1}, {"weight_out_ref": float("nan")},
    {"out_ref_teacher_score_thresh": -1}, {"out_ref_teacher_score_thresh": 1.1},
    {"out_ref_teacher_score_thresh": float("nan")},
])
def test_invalid_auxiliary_loss_config(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)

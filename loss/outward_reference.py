"""BiRefNet-inspired gradient supervision adapted to detection box regions.

BiRefNet uses a 5x5 image Laplacian and a segmentation-derived mask. Here
normalized absolute edges are supervised only inside GT/confident teacher boxes.
Boxes select pixels; they do not generate artificial rectangle-edge targets.
"""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def build_gradient_targets(images, image_sizes, boxes):
    """Return detached [B,1,H,W] targets and ROI masks, excluding padding."""
    images = images.detach().float()
    targets = images.new_zeros((len(images), 1, *images.shape[-2:]))
    masks = torch.zeros_like(targets, dtype=torch.bool)
    # Kornia-style normalized Laplacian kernel, implemented without a dependency.
    kernel = images.new_ones((1, 1, 5, 5))
    kernel[0, 0, 2, 2] = -24
    kernel /= 48
    for i, ((height, width), regions) in enumerate(zip(image_sizes, boxes)):
        gray = images[i:i + 1, :, :height, :width].mean(1, keepdim=True)
        padding = "reflect" if min(height, width) > 2 else "replicate"
        edges = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode=padding), kernel).abs()
        scale = edges.amax()
        edges = torch.where(scale > 1e-6, edges / scale.clamp_min(1e-6), 0.)
        targets[i:i + 1, :, :height, :width] = edges
        for x1, y1, x2, y2 in regions.detach().cpu().tolist():
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                continue
            left, top = max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1)))
            right, bottom = max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))
            masks[i, 0, top:bottom, left:right] = True
    return targets, masks


@torch.no_grad()
def reference_regions(context, teacher_detections=(), *, score_threshold=.7):
    """Use transformed GT, otherwise map confident original-coordinate boxes."""
    if context.transformed_targets is not None:
        return tuple(target["boxes"].detach() for target in context.transformed_targets)
    regions = []
    for i, ((height, width), (original_h, original_w)) in enumerate(zip(
        context.transformed_image_sizes, context.original_image_sizes
    )):
        selected = []
        for detections in teacher_detections:
            detection = detections[i]
            keep = torch.isfinite(detection["scores"]) & (detection["scores"] >= score_threshold)
            boxes = detection["boxes"][keep].detach()
            selected.append(boxes * boxes.new_tensor([
                width / original_w, height / original_h, width / original_w, height / original_h
            ]))
        regions.append(torch.cat(selected) if selected else
                       context.transformed_images.tensors.new_empty((0, 4)))
    return tuple(regions)


def outward_reference_loss(context, teacher_detections=(), *, score_threshold=.7):
    """Average masked BCE over valid images and P3/P4, reusing one forward.

Like BiRefNet, resize edge logits to the image target resolution. Targets and
regions are detached; only the student gradient branch receives this loss.
Empty images contribute neither loss nor denominator. All-empty batches yield
a differentiable zero, including every predictor in the graph.
"""
    logits = context.out_ref_logits
    if not logits:
        raise ValueError("OutRef loss requires training-time edge logits")
    normalized = context.transformed_images.tensors.detach()
    mean = normalized.new_tensor(context.image_mean).view(1, -1, 1, 1)
    std = normalized.new_tensor(context.image_std).view(1, -1, 1, 1)
    targets, masks = build_gradient_targets(
        normalized * std + mean, context.transformed_image_sizes,
        reference_regions(context, teacher_detections, score_threshold=score_threshold),
    )
    counts = masks.flatten(1).sum(1)
    valid = counts > 0
    level_losses = []
    for level_logits in logits:
        resized = F.interpolate(level_logits.float(), size=targets.shape[-2:],
                                mode="bilinear", align_corners=True)
        pixels = F.binary_cross_entropy_with_logits(resized, targets, reduction="none")
        per_image = (pixels * masks).flatten(1).sum(1) / counts.clamp_min(1)
        level_losses.append((per_image * valid).sum() / valid.sum().clamp_min(1))
    return torch.stack(level_losses).mean()

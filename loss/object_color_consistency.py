"""Object-level feature consistency across RGB and color-shifted views."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align


def object_color_consistency_loss(
    student_p3: Tensor,
    teacher_p3: Tensor,
    transformed_targets: Sequence[dict],
    *,
    output_size: int = 7,
    spatial_scale: float = 1.0 / 8.0,
    eps: float = 1e-6,
) -> Optional[Tensor]:
    """Return mean object-level normalized L1, or ``None`` when no GT is valid.

    ``transformed_targets`` must use the same post-resize coordinates as both
    feature maps.  P3 is stride 8 for the FCOS ResNet-FPN adapter.
    """
    if student_p3.ndim != 4 or teacher_p3.ndim != 4:
        raise ValueError("P3 feature maps must have shape [B, C, H, W]")
    if student_p3.shape != teacher_p3.shape:
        raise ValueError("student and teacher P3 feature maps must have equal shapes")
    if len(transformed_targets) != student_p3.shape[0]:
        raise ValueError("transformed targets must match the feature batch size")
    if output_size <= 0 or spatial_scale <= 0 or eps <= 0:
        raise ValueError("output_size, spatial_scale, and eps must be positive")

    rois = []
    for image_index, target in enumerate(transformed_targets):
        boxes = target["boxes"].to(device=student_p3.device, dtype=student_p3.dtype)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("each target boxes tensor must have shape [N, 4]")
        valid = torch.isfinite(boxes).all(dim=1)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        if boxes.numel():
            batch_column = boxes.new_full((boxes.shape[0], 1), image_index)
            rois.append(torch.cat((batch_column, boxes), dim=1))

    if not rois:
        return None

    pooled_student = roi_align(
        student_p3,
        torch.cat(rois, dim=0),
        output_size=(output_size, output_size),
        spatial_scale=spatial_scale,
        sampling_ratio=-1,
        aligned=True,
    )
    with torch.no_grad():
        pooled_teacher = roi_align(
            teacher_p3.detach(),
            torch.cat(rois, dim=0),
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=-1,
            aligned=True,
        )

    student_embeddings = pooled_student.mean(dim=(-2, -1))
    teacher_embeddings = pooled_teacher.mean(dim=(-2, -1))
    student_embeddings = F.normalize(student_embeddings, p=2, dim=1, eps=eps)
    teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1, eps=eps)
    return F.l1_loss(student_embeddings, teacher_embeddings.detach(), reduction="mean")

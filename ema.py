"""
EMA (Exponential Moving Average) teacher update.

teacher_params ← alpha * teacher_params + (1 - alpha) * student_params

Warmup schedule: during early iterations alpha is ramped up so the teacher
does not diverge immediately from a random student.
"""

from typing import Optional

import torch
import torch.nn as nn


def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    alpha: float = 0.999,
    global_step: Optional[int] = None,
) -> None:
    """
    Update teacher parameters in-place with EMA from student.

    Args:
        teacher      : teacher model (requires_grad=False)
        student      : student model (being trained)
        alpha        : EMA decay. Higher = slower teacher change.
        global_step  : current iteration. If provided, applies warmup:
                       effective_alpha = min(alpha, (1 + step) / (10 + step))
                       This prevents the teacher from diverging too fast at
                       the very start when the student is still random.
    """
    if global_step is not None:
        # Warmup: alpha ramps from ~0.09 → target_alpha over first ~10k steps
        warmup_alpha = (1.0 + global_step) / (10.0 + global_step)
        alpha = min(alpha, warmup_alpha)

    with torch.no_grad():
        # Update learnable parameters
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())

        for name, t_p in t_params.items():
            if name in s_params:
                t_p.data.mul_(alpha).add_(s_params[name].data, alpha=1.0 - alpha)

        # Update buffers (e.g. BatchNorm running_mean / running_var)
        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())

        for name, t_b in t_bufs.items():
            if name in s_bufs and t_b.is_floating_point():
                t_b.data.mul_(alpha).add_(s_bufs[name].data, alpha=1.0 - alpha)


def copy_student_to_teacher(teacher: nn.Module, student: nn.Module) -> None:
    """Hard copy student weights into teacher (alpha=0). Used for initialization."""
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.copy_(s_p.data)
        for t_b, s_b in zip(teacher.buffers(), student.buffers()):
            if t_b.is_floating_point():
                t_b.data.copy_(s_b.data)

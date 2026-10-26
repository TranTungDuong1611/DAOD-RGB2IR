"""Execute teacher operations requested by the trainer, without schedule policy."""

import torch

from ema import copy_student_to_teacher, ema_update


def prepare_models(student, teachers, *, device):
    """Preserve device transfer order and freeze the supplied teachers."""
    student.to(device)
    for teacher in teachers.values():
        if teacher is None:
            continue
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False


def initialize_teachers(student, teachers, *, names):
    """Hard-copy only the names selected by the caller at its chosen start step."""
    for name in names:
        teacher = teachers[name]
        if teacher is None:
            raise ValueError(f"EMA start requires the '{name}' teacher")
        copy_student_to_teacher(teacher, student)


def get_teacher(teachers, name):
    teacher = teachers.get(name)
    if teacher is None:
        raise ValueError(f"teacher '{name}' is not configured")
    teacher.eval()
    return teacher


def forward_teacher(teacher, images, *, sample_ids, domain, resize_short_edge):
    """Reuse the student-selected geometry and keep teacher forward outside autograd."""
    with torch.no_grad():
        kwargs = {"sample_ids": sample_ids, "domain": domain}
        if resize_short_edge is not None:
            kwargs["resize_short_edge"] = resize_short_edge
        return teacher.raw(images, **kwargs)


def update_teacher(teacher, student, *, alpha, global_step):
    """Run the existing EMA primitive with caller-supplied parameters."""
    ema_update(teacher, student, alpha=alpha, global_step=global_step)

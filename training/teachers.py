"""Execute teacher operations requested by the trainer, without schedule policy."""

import torch

from ema import copy_student_to_teacher, ema_update


def prepare_models(student, teachers, *, device):
    """Preserve device transfer order and freeze the supplied teachers."""
    # Establish model placement first, then make every supplied teacher an
    # inference-only reference. Teacher enablement is decided by the caller.
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
    # Execute one initialization request. The trainer owns the start-step check
    # and persisted initialization flag, including the behavior after resume.
    for name in names:
        teacher = teachers[name]
        if teacher is None:
            raise ValueError(f"EMA start requires the '{name}' teacher")
        copy_student_to_teacher(teacher, student)


def get_teacher(teachers, name):
    # Resolve a requested reference and restore its inference mode before use.
    # Route/teacher-mode filtering happens upstream, rather than in this lookup.
    teacher = teachers.get(name)
    if teacher is None:
        raise ValueError(f"teacher '{name}' is not configured")
    teacher.eval()
    return teacher


def forward_teacher(teacher, images, *, sample_ids, resize_short_edge, domain="rgb"):
    """Reuse the student-selected geometry and keep teacher forward outside autograd."""
    # Produce KD targets from the prepared teacher view with the same sample
    # identity and resize selected for the student forward.
    with torch.no_grad():
        kwargs = {"sample_ids": sample_ids}
        if getattr(teacher, "supports_domain_routing", False):
            kwargs["domain"] = domain
        if resize_short_edge is not None:
            kwargs["resize_short_edge"] = resize_short_edge
        return teacher.raw(images, **kwargs)


def update_teacher(teacher, student, *, alpha, global_step):
    """Run the existing EMA primitive with caller-supplied parameters."""
    # The trainer issues this request after an eligible student update; the
    # existing EMA primitive retains responsibility for parameter/buffer math.
    ema_update(teacher, student, alpha=alpha, global_step=global_step)

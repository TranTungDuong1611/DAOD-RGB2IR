import torch

from loss.object_color_consistency import object_color_consistency_loss


def _target(boxes):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)}


def test_object_color_consistency_normalizes_each_object_embedding():
    student = torch.ones((1, 2, 8, 8))
    teacher = torch.full((1, 2, 8, 8), 5.0)

    loss = object_color_consistency_loss(
        student, teacher, [_target([[0, 0, 32, 32]])]
    )

    assert loss is not None
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_object_color_consistency_backpropagates_only_through_student_features():
    student = torch.zeros((1, 2, 8, 8), requires_grad=True)
    teacher = torch.zeros((1, 2, 8, 8), requires_grad=True)
    with torch.no_grad():
        student[:, 0] = 1.0
        teacher[:, 1] = 1.0

    loss = object_color_consistency_loss(
        student, teacher, [_target([[0, 0, 32, 32]])]
    )
    assert loss is not None
    loss.backward()

    assert student.grad is not None
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_object_color_consistency_skips_batches_without_valid_objects():
    feature = torch.randn((1, 2, 8, 8), requires_grad=True)

    loss = object_color_consistency_loss(
        feature, feature.detach(), [_target([])]
    )

    assert loss is None

"""Audit masks must expose the batch-global rows actually used by KD."""

import torch

from loss.d3t_criterion import D3TLossCriterion
from models.d3t_adapter import DistillationPair, DistillationSettings, Predictions
from training.loss_flow import LossFlow
from models.d3t_wrapper import D3TWrapper
from test_curriculum_wrapper_integration import TinyAdapter


def test_selection_masks_preserve_batch_global_topk_and_image_boundaries():
    teacher = tuple(Predictions(
        torch.logit(torch.tensor(probs)).reshape(-1, 1),
        torch.tensor([[0., 0., 2., 2.]] * len(probs)), torch.zeros(len(probs))
    ) for probs in ([.1, .9], [.8]))
    settings = DistillationSettings(top_ratio=.34, min_hm=0., hm_alpha=1., hm_beta=0.)
    result = D3TLossCriterion().distillation(DistillationPair(teacher, teacher), settings)
    assert [mask.tolist() for mask in result.selected_masks] == [[False, True], [True]]
    assert sum(mask.sum().item() for mask in result.selected_masks) == 2
    assert all(not mask.requires_grad for mask in result.selected_masks)


def test_loss_flow_exposes_selection_without_adding_masks_to_loss_or_logs():
    wrapper = D3TWrapper(TinyAdapter(), D3TLossCriterion())
    output = wrapper.raw(torch.ones(1, 3, 4, 4), sample_ids=("a",))
    flow = LossFlow(uhl=1., quality=1., deltas=1., logits=1.)
    result = flow.distill(wrapper, output, output, DistillationSettings(1., 1.),
                          teacher_name="ir", phase_weight=1.)
    assert result is not None
    assert [mask.tolist() for mask in result.selected_masks] == [[False, False]]
    assert flow.logs["kd_empty_mask_ir"] == 1.
    assert flow.total_loss.requires_grad
    flow.total_loss.backward()
    assert wrapper.adapter.weight.grad.item() == 0.

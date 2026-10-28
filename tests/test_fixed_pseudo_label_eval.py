"""Fixed teacher evaluation must aggregate images without touching training state."""

import copy
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from data.datasets import FLIRIRDataset
from data.dataloaders import ir_collate
from data.ir_annotations import IRAnnotationReader
from loss.d3t_criterion import D3TLossCriterion
from models.d3t_adapter import DistillationPair, DistillationSettings, Predictions
from models.d3t_wrapper import D3TWrapper
from test_torchvision_fcos_adapter import build_adapter


@pytest.fixture
def validation_root():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "JPEGImages").mkdir()
        (root / "Annotations").mkdir()
        stems = [f"image-{i}" for i in range(5)]
        (root / "align_validation.txt").write_text("\n".join(stems))
        for stem, label in zip(stems, ["person", "person", "person", "car", "bicycle"]):
            Image.new("L", (48, 32), color=80).save(root / "JPEGImages" / f"{stem}.jpeg")
            (root / "Annotations" / f"{stem}.xml").write_text(
                "<Annotation><size><width>48</width><height>32</height></size>"
                f"<object><name>{label}</name><bndbox>"
                "<xmin>4</xmin><ymin>3</ymin><xmax>14</xmax><ymax>15</ymax>"
                "</bndbox></object></Annotation>"
            )
        yield root


def test_shared_teacher_selection_matches_batch_global_training_mask():
    from loss.d3t_criterion import select_distillation_masks

    predictions = tuple(Predictions(
        torch.logit(torch.tensor(values)).reshape(-1, 1),
        torch.tensor([[0., 0., 2., 2.]] * len(values)), torch.zeros(len(values))
    ) for values in ([.1], [.9, .8]))
    settings = DistillationSettings(.34, 0., hm_alpha=1., hm_beta=0.)
    masks = select_distillation_masks(predictions, settings)
    assert [m.tolist() for m in masks] == [[False], [True, True]]
    trained = D3TLossCriterion().distillation(DistillationPair(predictions, predictions), settings)
    assert all(torch.equal(a, b) for a, b in zip(masks, trained.selected_masks))


def test_aggregate_uses_row_and_object_denominators_instead_of_batch_means():
    from training.pseudo_label_eval import aggregate_pseudo_label_metrics

    metrics = aggregate_pseudo_label_metrics([
        {"selected_rows": 3, "valid_rows": 1, "gt_count": 10, "covered_gt": 1,
         "finite_foreground_rows": 2, "correct_class_rows": 1,
         "foreground_box_iou_mean": 1., "quality_mae_foreground": .5},
        {"selected_rows": 1, "valid_rows": 1, "gt_count": 1, "covered_gt": 1,
         "finite_foreground_rows": 1, "correct_class_rows": 1,
         "foreground_box_iou_mean": .25, "quality_mae_foreground": 0.},
    ])
    assert metrics["valid_fraction"] == .5
    assert metrics["gt_coverage"] == pytest.approx(2 / 11)
    assert metrics["foreground_class_accuracy"] == pytest.approx(2 / 3)
    assert metrics["foreground_box_iou_mean"] == .75
    assert metrics["quality_mae_foreground"] == pytest.approx(1 / 3)


def test_fixed_subset_is_seeded_and_includes_rare_classes(validation_root):
    from training.pseudo_label_eval import select_fixed_ir_subset

    dataset = FLIRIRDataset(validation_root, split="validation")
    reader = IRAnnotationReader(validation_root)
    before = random.getstate()
    indices = select_fixed_ir_subset(dataset.ir_paths, reader, sample_size=3, seed=12)
    assert random.getstate() == before
    assert len(indices) == len(set(indices)) == 3
    assert 3 in indices and 4 in indices  # Car and bicycle must not disappear.
    assert indices == select_fixed_ir_subset(dataset.ir_paths, reader, sample_size=3, seed=12)
    assert len(select_fixed_ir_subset(dataset.ir_paths, reader, sample_size=256, seed=12)) == 5


def make_evaluator(root):
    from training.pseudo_label_eval import FixedIRPseudoLabelEvaluator

    dataset = FLIRIRDataset(root, split="validation")
    fixed_indices = [0, 3, 4]
    def loader(data):
        return DataLoader(data, batch_size=2, shuffle=False, collate_fn=ir_collate,
                          generator=torch.Generator().manual_seed(42))
    return FixedIRPseudoLabelEvaluator(
        fixed_loader=loader(Subset(dataset, fixed_indices)), full_loader=loader(dataset),
        reader=IRAnnotationReader(root), device="cpu",
        fixed_sample_ids=tuple(dataset.ir_paths[i].stem for i in fixed_indices),
    )


def test_fixed_and_full_passes_keep_last_batch_and_restore_model_and_rng(validation_root):
    evaluator = make_evaluator(validation_root)
    model = D3TWrapper(build_adapter(), D3TLossCriterion())
    model.train()
    model.adapter.detector.backbone.eval()  # Preserve mixed child modes too.
    modes = {n: m.training for n, m in model.named_modules()}
    weights = copy.deepcopy(model.state_dict())
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    settings = DistillationSettings(.1, 0.)
    for full, images, batches in [(False, 3, 2), (True, 5, 3)]:
        result = evaluator.evaluate(model, settings, iou_threshold=.5,
                                    resize_short_edge=64, full_dataset=full)
        assert result["images"] == images and result["batches"] == batches
        assert result["metrics"]["available_images"] == images
        assert len(result["sample_ids"]) == images
        assert {n: m.training for n, m in model.named_modules()} == modes
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert random.getstate() == python_rng
        after = np.random.get_state()
        assert numpy_rng[0] == after[0] and np.array_equal(numpy_rng[1], after[1])
        assert numpy_rng[2:] == after[2:]
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())


def test_evaluation_restores_state_when_teacher_forward_raises(validation_root):
    evaluator = make_evaluator(validation_root)
    model = D3TWrapper(build_adapter(), D3TLossCriterion()).train()
    before = torch.get_rng_state().clone()
    def fail(*args, **kwargs):
        torch.rand(1)
        raise RuntimeError("test forward failure")
    model.raw = fail
    with pytest.raises(RuntimeError, match="test forward failure"):
        evaluator.evaluate(model, DistillationSettings(.1, 0.), iou_threshold=.5,
                           resize_short_edge=64)
    assert model.training
    assert torch.equal(torch.get_rng_state(), before)


def test_resume_rejects_changed_fixed_subset_manifest(validation_root):
    evaluator = make_evaluator(validation_root)
    state = evaluator.state_dict()
    evaluator.load_state_dict(state)
    state["fixed_sample_ids"] = ["different-image"]
    with pytest.raises(ValueError, match="subset"):
        evaluator.load_state_dict(state)

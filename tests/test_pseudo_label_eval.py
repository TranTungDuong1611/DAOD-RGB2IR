"""Hand-calculated dense FCOS audit metrics, independent of detector AP."""

from types import SimpleNamespace

import pytest
import torch

from data.ir_annotations import IRAnnotation
from models.d3t_adapter import AdapterOutput, Predictions
from models.torchvision_fcos_adapter import FCOSAdapterContext
from test_torchvision_fcos_adapter import build_adapter


def fixture_output():
    # All boxes match one GT, even though the third location is outside it.
    prediction = Predictions(
        torch.tensor([[4., -4., -4.], [-4., 4., -4.], [4., -4., -4.]]),
        torch.tensor([[2., 2., 6., 6.]] * 3), torch.zeros(3)
    )
    anchors = torch.tensor([[2., 2., 4., 4.], [3., 3., 5., 5.], [20., 10., 22., 12.]])
    context = FCOSAdapterContext(
        transformed_images=SimpleNamespace(image_sizes=[(20, 40)]),
        transformed_targets=None, original_image_sizes=((10, 20),), features=(),
        anchors_per_image=(anchors,), anchors_per_level=((anchors,),),
        num_anchors_per_level=(3,), raw_head_outputs={}, sample_ids=("a",)
    )
    return AdapterOutput((prediction,), context=context, sample_ids=("a",))


def test_dense_audit_matches_boxes_regardless_of_location_and_preserves_resize():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    output = fixture_output()
    target = {"boxes": torch.tensor([[1., 1., 3., 3.]]), "labels": torch.tensor([0])}
    adapter = build_adapter()
    metrics = evaluate_pseudo_labels(
        adapter, output, (torch.tensor([True, True, True]),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    assert metrics["selected_rows"] == 3
    assert metrics["selected_foreground_rows"] == 3
    assert metrics["valid_rows"] == 2
    assert metrics["valid_fraction"] == pytest.approx(2 / 3)
    assert metrics["foreground_class_accuracy"] == pytest.approx(2 / 3)
    assert metrics["foreground_box_iou_mean"] == 1.
    assert metrics["quality_mae_foreground"] == .5
    assert metrics["gt_coverage"] == 1.
    assert metrics["covered_gt"] == 1  # Dense duplicates do not inflate coverage.
    assert output.context.transformed_targets is None
    assert target["boxes"].tolist() == [[1., 1., 3., 3.]]


def test_same_class_match_can_be_a_different_gt_from_the_closest_box():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    output = fixture_output()
    target = {"boxes": torch.tensor([[1., 1., 3., 3.], [.75, .75, 3.25, 3.25]]),
              "labels": torch.tensor([1, 0])}
    metrics = evaluate_pseudo_labels(
        build_adapter(), output, (torch.tensor([False, False, True]),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    # Row 3 predicts class 0. Its box has IoU 1 with the wrong-class GT and
    # IoU .64 with the correct-class GT; either FCOS location rule would miss it.
    assert metrics["valid_rows"] == 1
    assert metrics["valid_fraction"] == 1.
    assert metrics["covered_gt"] == 1
    assert metrics["gt_coverage"] == .5


def test_matching_checks_iou_boundary_and_keeps_unmatched_rows_in_denominator():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    output = fixture_output()
    prediction, = output.predictions
    prediction.class_logits[:] = prediction.class_logits[0].clone()
    prediction.boxes[:] = torch.tensor([[2., 2., 4., 6.], [2., 2., 3.8, 6.], [20., 20., 24., 24.]])
    target = {"boxes": torch.tensor([[1., 1., 3., 3.]]), "labels": torch.tensor([0])}
    metrics = evaluate_pseudo_labels(
        build_adapter(), output, (torch.ones(3, dtype=torch.bool),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    assert metrics["selected_rows"] == 3
    assert metrics["selected_foreground_rows"] == 1
    assert metrics["valid_rows"] == 1
    assert metrics["valid_fraction"] == pytest.approx(1 / 3)
    assert metrics["foreground_box_iou_mean"] == .5


def test_geometric_matches_do_not_hide_wrong_class_predictions():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    target = {"boxes": torch.tensor([[1., 1., 3., 3.]]), "labels": torch.tensor([0])}
    metrics = evaluate_pseudo_labels(
        build_adapter(), fixture_output(), (torch.tensor([False, True, False]),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    assert metrics["selected_foreground_rows"] == 1
    assert metrics["foreground_class_accuracy"] == 0.
    assert metrics["valid_fraction"] == 0.
    assert metrics["gt_coverage"] == 0.


def test_missing_labels_are_excluded_but_annotated_empty_is_background():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    output = fixture_output()
    empty = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    for record, denominator, fraction in [
        (IRAnnotation("missing"), 0, None),
        (IRAnnotation("invalid"), 0, None),
        (IRAnnotation("available", empty), 3, 0.),
    ]:
        metrics = evaluate_pseudo_labels(
            build_adapter(), output, (torch.ones(3, dtype=torch.bool),),
            (record,), iou_threshold=.5
        )
        assert metrics["selected_rows"] == denominator
        assert metrics["valid_fraction"] == fraction
        assert metrics["gt_coverage"] is None
        assert metrics["quality_mae_foreground"] is None
        assert metrics["available_images"] == int(record.status == "available")


def test_no_selected_rows_has_coverage_zero_without_fake_quality_score():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    target = {"boxes": torch.tensor([[1., 1., 3., 3.]]), "labels": torch.tensor([0])}
    metrics = evaluate_pseudo_labels(
        build_adapter(), fixture_output(), (torch.zeros(3, dtype=torch.bool),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    assert metrics["valid_fraction"] is None
    assert metrics["gt_coverage"] == 0.
    assert metrics["quality_mae_foreground"] is None


def test_audit_rejects_misaligned_masks_instead_of_truncating_batch():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    with pytest.raises(ValueError, match="align"):
        evaluate_pseudo_labels(build_adapter(), fixture_output(), (), (), iou_threshold=.5)


def test_correct_dense_duplicates_cover_one_object_and_nonfinite_rows_stay_invalid():
    from training.pseudo_label_eval import evaluate_pseudo_labels

    output = fixture_output()
    prediction, = output.predictions
    prediction.class_logits[1] = prediction.class_logits[0]
    prediction.boxes[2, 0] = float("nan")
    target = {"boxes": torch.tensor([[1., 1., 3., 3.]]), "labels": torch.tensor([0])}
    metrics = evaluate_pseudo_labels(
        build_adapter(), output, (torch.ones(3, dtype=torch.bool),),
        (IRAnnotation("available", target),), iou_threshold=.5
    )
    assert metrics["valid_rows"] == 2
    assert metrics["valid_fraction"] == pytest.approx(2 / 3)
    assert metrics["covered_gt"] == 1
    assert metrics["gt_coverage"] == 1.
    assert metrics["nonfinite_selected_rows"] == 1

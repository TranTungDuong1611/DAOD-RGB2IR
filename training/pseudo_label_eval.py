"""Observe selected dense KD rows against IR GT; never produce training inputs."""

import torch

from models.torchvision_fcos_adapter import _aligned_box_iou


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def evaluate_pseudo_labels(adapter, output, selected_masks, annotations, *, iou_threshold):
    """Report FCOS-assigned row validity and unique GT coverage before NMS.

    Multiple selected locations may correctly cover one object. These metrics
    describe KD supervision, not detection precision/recall or teacher AP.
    """
    if selected_masks is None or not (
        len(output.predictions) == len(selected_masks) == len(annotations)
    ):
        raise ValueError("diagnostic predictions, masks, and annotations must align")
    if not 0. <= iou_threshold <= 1.:
        raise ValueError("diagnostic IoU threshold must be in [0, 1]")
    assigned = adapter.assign_diagnostic_targets(output, [r.target for r in annotations])
    counts = dict(available_images=0, missing_images=0, invalid_images=0,
                  selected_rows_all=0, selected_rows=0, selected_foreground_rows=0,
                  finite_foreground_rows=0, nonfinite_selected_rows=0,
                  valid_rows=0, correct_class_rows=0, gt_count=0, covered_gt=0)
    iou_sum = quality_error_sum = 0.
    for prediction, selected, record, target in zip(
        output.predictions, selected_masks, annotations, assigned
    ):
        if selected.dtype != torch.bool or selected.shape != prediction.quality_logits.shape:
            raise ValueError("diagnostic mask must align with prediction rows")
        counts["selected_rows_all"] += int(selected.sum().item())
        counts[f"{record.status}_images"] += 1
        # Unknown annotations contribute only to availability statistics. A
        # valid empty annotation still counts selected rows as background.
        if target is None:
            continue
        counts["selected_rows"] += int(selected.sum().item())
        counts["gt_count"] += len(target["boxes"])
        matched = target["matched_idxs"]
        foreground = selected & (matched >= 0)
        counts["selected_foreground_rows"] += int(foreground.sum().item())
        finite = (torch.isfinite(prediction.boxes).all(dim=1)
                  & torch.isfinite(prediction.class_logits).all(dim=1)
                  & torch.isfinite(prediction.quality_logits))
        counts["nonfinite_selected_rows"] += int((selected & ~finite).sum().item())
        foreground &= finite
        count = int(foreground.sum().item())
        counts["finite_foreground_rows"] += count
        if not count:
            continue

        # Quality is trained only on foreground and predicts box IoU. Compare
        # it with the same assigned GT definition, without background zeroes.
        gt_indices = matched[foreground]
        iou = _aligned_box_iou(prediction.boxes[foreground], target["boxes"][gt_indices])
        correct_class = (prediction.class_logits[foreground].argmax(dim=1)
                         == target["labels"][gt_indices])
        valid = correct_class & (iou >= iou_threshold)
        counts["correct_class_rows"] += int(correct_class.sum().item())
        counts["valid_rows"] += int(valid.sum().item())
        counts["covered_gt"] += int(gt_indices[valid].unique().numel())
        iou_sum += float(iou.sum().item())
        quality_error_sum += float((prediction.quality_logits[foreground].sigmoid() - iou).abs().sum().item())

    # Undefined denominators stay null in JSON/logs rather than suggesting a
    # perfect or zero score for an unlabeled batch or an empty foreground set.
    foreground_count = counts["finite_foreground_rows"]
    return {
        **counts,
        "valid_fraction": _ratio(counts["valid_rows"], counts["selected_rows"]),
        "foreground_class_accuracy": _ratio(counts["correct_class_rows"], foreground_count),
        "foreground_box_iou_mean": _ratio(iou_sum, foreground_count),
        "quality_mae_foreground": _ratio(quality_error_sum, foreground_count),
        "gt_coverage": _ratio(counts["covered_gt"], counts["gt_count"]),
    }

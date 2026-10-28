"""Evaluate dense teacher supervision on fixed and full IR validation sets."""

from contextlib import contextmanager
import random

import numpy as np
from PIL import Image
import torch

from loss.d3t_criterion import select_distillation_masks
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


def aggregate_pseudo_label_metrics(batches):
    """Pool counts and weighted sums so small/empty batches cannot bias ratios."""
    batches = list(batches)
    count_names = (
        "available_images", "missing_images", "invalid_images", "selected_rows_all",
        "selected_rows", "selected_foreground_rows", "finite_foreground_rows",
        "nonfinite_selected_rows", "valid_rows", "correct_class_rows", "gt_count", "covered_gt",
    )
    counts = {name: sum(batch.get(name, 0) for batch in batches) for name in count_names}
    foreground = counts["finite_foreground_rows"]
    # The per-batch means already exclude non-finite/unassigned rows. Recover
    # their sums using that exact denominator, not the number of images/batches.
    means = {}
    for name in ("foreground_box_iou_mean", "quality_mae_foreground"):
        total = sum((batch.get(name) or 0.) * batch["finite_foreground_rows"] for batch in batches)
        means[name] = _ratio(total, foreground)
    return {
        **counts, **means,
        "valid_fraction": _ratio(counts["valid_rows"], counts["selected_rows"]),
        "foreground_class_accuracy": _ratio(counts["correct_class_rows"], foreground),
        "gt_coverage": _ratio(counts["covered_gt"], counts["gt_count"]),
    }


def select_fixed_ir_subset(image_paths, reader, *, sample_size, seed):
    """Choose reproducible labeled images, retaining class coverage when possible."""
    if sample_size <= 0:
        raise ValueError("fixed IR sample size must be positive")
    labels_by_index = {}
    # Read headers/annotations only during assembly. Never iterate a training
    # loader or consume its augmentation/sampling RNG to choose audit images.
    for index, path in enumerate(image_paths):
        with Image.open(path) as image:
            width, height = image.size
        record, = reader.read((path.stem,), ((height, width),), hflipped=False)
        if record.status == "available":
            labels_by_index[index] = set(record.target["labels"].tolist())
    if not labels_by_index:
        raise ValueError("no valid IR validation annotations for pseudo-label evaluation")
    candidates = list(labels_by_index)
    random.Random(seed).shuffle(candidates)
    limit = min(sample_size, len(candidates))
    classes = sorted(set().union(*labels_by_index.values()))
    selected = []
    for label in classes:
        if len(selected) == limit:
            break
        if any(label in labels_by_index[i] for i in selected):
            continue
        selected.append(next(i for i in candidates if label in labels_by_index[i]))
    selected.extend(i for i in candidates if i not in selected)
    return tuple(sorted(selected[:limit]))


@contextmanager
def _preserve_evaluation_state(model):
    """Leave training RNG and every pre-existing model mode intact, even on errors."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        yield
    finally:
        for module, was_training in modes:
            module.training = was_training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


class FixedIRPseudoLabelEvaluator:
    """Run isolated teacher-only passes using caller-supplied loaders and settings."""

    def __init__(self, *, fixed_loader, full_loader, reader, device, fixed_sample_ids):
        self.fixed_loader = fixed_loader
        self.full_loader = full_loader
        self.reader = reader
        self.device = device
        self.fixed_sample_ids = tuple(fixed_sample_ids)

    def state_dict(self):
        return {"fixed_sample_ids": list(self.fixed_sample_ids),
                "batch_size": self.fixed_loader.batch_size}

    def load_state_dict(self, state):
        # Comparing the manifest makes resumed metrics refer to the same images
        # and batch-global top-k groups; a changed dataset is not silently mixed.
        if state != self.state_dict():
            raise ValueError("pseudo-label evaluation subset or batch size changed on resume")

    def evaluate(self, teacher, settings, *, iou_threshold, resize_short_edge, full_dataset=False):
        loader = self.full_loader if full_dataset else self.fixed_loader
        batch_metrics, sample_ids = [], []
        # No student forward, target-dependent selection, backward, or EMA is
        # performed here. Teacher weights stay fixed throughout the entire pass.
        with _preserve_evaluation_state(teacher), torch.inference_mode():
            for images, ids in loader:
                images = (images.to(self.device) if isinstance(images, torch.Tensor)
                          else tuple(image.to(self.device) for image in images))
                output = teacher.raw(images, sample_ids=ids, resize_short_edge=resize_short_edge)
                annotations = self.reader.read(ids, output.context.original_image_sizes, hflipped=False)
                masks = select_distillation_masks(output.predictions, settings)
                batch_metrics.append(evaluate_pseudo_labels(
                    teacher.adapter, output, masks, annotations, iou_threshold=iou_threshold,
                ))
                sample_ids.extend(ids)
        if not batch_metrics:
            raise ValueError("pseudo-label evaluation loader is empty")
        return {
            "scope": "full_ir_validation" if full_dataset else "fixed_ir_validation",
            "images": len(sample_ids), "batches": len(batch_metrics),
            "batch_size": loader.batch_size, "sample_ids": sample_ids,
            "metrics": aggregate_pseudo_label_metrics(batch_metrics),
        }

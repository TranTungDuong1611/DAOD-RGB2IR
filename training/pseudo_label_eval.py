"""Evaluate dense teacher supervision on fixed and full RGB/IR validation sets."""

from contextlib import contextmanager
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_iou

from data.augmentations import default_ir_transform, default_rgb_transform
from data.datasets import _resolve_split_file
from loss.d3t_criterion import select_distillation_masks
from utils.helper import ir_stem_to_rgb_filename, read_split_file


class ValidationImages(Dataset):
    """Load unaugmented audit images while retaining the shared XML sample ID."""

    def __init__(self, root, domain):
        root = Path(root)
        self.domain = domain
        self.transform = default_rgb_transform() if domain == "rgb" else default_ir_transform()
        self.samples = []
        # RGB filenames differ from annotation stems. Keep the original split
        # identity so both domains read the same aligned XML without renaming it.
        for stem in read_split_file(_resolve_split_file(root, "validation")):
            filename = ir_stem_to_rgb_filename(stem) if domain == "rgb" else f"{stem}.jpeg"
            path = root / "JPEGImages" / filename
            if path.is_file():
                self.samples.append((path, stem))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, stem = self.samples[index]
        # Labels are read only by the observer, so a missing/corrupt XML remains
        # an availability result rather than interrupting the full image pass.
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB") if self.domain == "rgb" else image)
        return tensor, stem


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def evaluate_pseudo_labels(adapter, output, selected_masks, annotations, *, iou_threshold):
    """Match selected decoded boxes to same-class GT by IoU, before NMS.

    A valid row can match any same-class GT, regardless of its FCOS location,
    level, or center region. Dense duplicates count as rows but cover a GT once.
    """
    if selected_masks is None or not (
        len(output.predictions) == len(selected_masks) == len(annotations)
    ):
        raise ValueError("diagnostic predictions, masks, and annotations must align")
    if not 0. <= iou_threshold <= 1.:
        raise ValueError("diagnostic IoU threshold must be in [0, 1]")
    targets = adapter.prepare_diagnostic_targets(output, [r.target for r in annotations])
    counts = dict(available_images=0, missing_images=0, invalid_images=0,
                  selected_rows_all=0, selected_rows=0, selected_foreground_rows=0,
                  finite_foreground_rows=0, nonfinite_selected_rows=0,
                  valid_rows=0, correct_class_rows=0, gt_count=0, covered_gt=0)
    iou_sum = quality_error_sum = 0.
    for prediction, selected, record, target in zip(
        output.predictions, selected_masks, annotations, targets
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
        finite = (torch.isfinite(prediction.boxes).all(dim=1)
                  & torch.isfinite(prediction.class_logits).all(dim=1)
                  & torch.isfinite(prediction.quality_logits))
        counts["nonfinite_selected_rows"] += int((selected & ~finite).sum().item())
        usable = selected & finite
        if not usable.any() or not len(target["boxes"]):
            continue

        # Geometry alone defines the foreground denominator; do not filter by
        # class first, which would make foreground class accuracy always 100%.
        ious = box_iou(prediction.boxes[usable], target["boxes"])
        best_iou = ious.max(dim=1).values
        foreground = best_iou >= iou_threshold
        count = int(foreground.sum().item())
        counts["selected_foreground_rows"] += count
        counts["finite_foreground_rows"] += count
        if not count:
            continue

        # Search same-class GT separately: a nearer wrong-class object must not
        # hide another GT that satisfies both the class and IoU requirements.
        labels = prediction.class_logits[usable].argmax(dim=1)
        same_class = labels[:, None] == target["labels"][None, :]
        class_iou, gt_indices = ious.masked_fill(~same_class, -1.).max(dim=1)
        valid = class_iou >= iou_threshold
        counts["correct_class_rows"] += int(valid.sum().item())
        counts["valid_rows"] += int(valid.sum().item())
        counts["covered_gt"] += int(gt_indices[valid].unique().numel())
        # IoU/quality describe geometrically matched rows, including wrong-class
        # predictions. All selected rows still remain in the validity denominator.
        iou_sum += float(best_iou[foreground].sum().item())
        quality_error_sum += float((prediction.quality_logits[usable][foreground].sigmoid()
                                    - best_iou[foreground]).abs().sum().item())

    # Undefined denominators stay null in JSON/logs rather than suggesting a
    # perfect or zero score for an unlabeled batch or an empty matched set.
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
    # The per-batch means already exclude non-finite/unmatched rows. Recover
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


def select_fixed_subset(samples, reader, *, sample_size, seed):
    """Choose reproducible labeled images, retaining class coverage when possible."""
    if sample_size <= 0:
        raise ValueError("fixed sample size must be positive")
    labels_by_index = {}
    # Read headers/annotations only during assembly. Never iterate a training
    # loader or consume its augmentation/sampling RNG to choose audit images.
    for index, (path, stem) in enumerate(samples):
        with Image.open(path) as image:
            width, height = image.size
        record, = reader.read((stem,), ((height, width),), hflipped=False)
        if record.status == "available":
            labels_by_index[index] = set(record.target["labels"].tolist())
    if not labels_by_index:
        raise ValueError("no valid validation annotations for pseudo-label evaluation")
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
def _preserve_evaluation_state(model, generator):
    """Leave training RNG and every pre-existing model mode intact, even on errors."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    loader_rng = generator.get_state() if generator is not None else None
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
        if loader_rng is not None:
            generator.set_state(loader_rng)


class FixedPseudoLabelEvaluator:
    """Run isolated teacher-only passes using caller-supplied loaders and settings."""

    def __init__(self, *, fixed_loaders, full_loaders, reader, device, fixed_sample_ids):
        self.fixed_loaders = fixed_loaders
        self.full_loaders = full_loaders
        self.reader = reader
        self.device = device
        self.fixed_sample_ids = {domain: tuple(ids) for domain, ids in fixed_sample_ids.items()}

    def state_dict(self):
        return {domain: {"fixed_sample_ids": list(ids),
                         "batch_size": self.fixed_loaders[domain].batch_size}
                for domain, ids in self.fixed_sample_ids.items()}

    def load_state_dict(self, state):
        # Comparing the manifest makes resumed metrics refer to the same images
        # and batch-global top-k groups; a changed dataset is not silently mixed.
        # Older observers used one IR manifest for both teachers. Retain its IR
        # grouping, while RGB starts with its newly assembled validation subset.
        if "fixed_sample_ids" in state:
            state = {"ir": state} if "ir" in self.fixed_loaders else {}
        current = self.state_dict()
        if any(current.get(domain) != saved for domain, saved in state.items()):
            raise ValueError("pseudo-label evaluation subset or batch size changed on resume")

    def evaluate(self, teacher, settings, *, source_domain, iou_threshold, resize_short_edge, full_dataset=False):
        # The trainer resolves teacher -> domain policy; this executor only
        # selects the supplied loader and performs an isolated evaluation pass.
        loader = (self.full_loaders if full_dataset else self.fixed_loaders)[source_domain]
        batch_metrics, sample_ids = [], []
        # No student forward, target-dependent selection, backward, or EMA is
        # performed here. Teacher weights stay fixed throughout the entire pass.
        with _preserve_evaluation_state(teacher, loader.generator), torch.inference_mode():
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
        if not full_dataset and tuple(sample_ids) != self.fixed_sample_ids[source_domain]:
            raise ValueError("fixed pseudo-label evaluation image order changed")
        return {
            "matching": "same_class_box_iou",
            "source_domain": source_domain,
            "scope": f"{'full' if full_dataset else 'fixed'}_{source_domain}_validation",
            "images": len(sample_ids), "batches": len(batch_metrics),
            "batch_size": loader.batch_size, "sample_ids": sample_ids,
            "metrics": aggregate_pseudo_label_metrics(batch_metrics),
        }

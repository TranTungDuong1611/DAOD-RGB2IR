"""Detection metrics and domain evaluation for the training lifecycle."""

from collections import defaultdict
import copy
import logging
import time
from typing import Dict, List, Optional

import torch
from torchvision.ops import box_iou

from contextlib import contextmanager
import random

import numpy as np
from PIL import Image
from models.fcos_adapter import _aligned_box_iou
from loss.detection_criterion import select_distillation_masks
from training.observability import log_pseudo_label_evaluation
from training.step_policy import configured_teacher_names, distillation_settings

logger = logging.getLogger(__name__)


def _compute_ap_voc11(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    return sum(
        precisions[recalls >= threshold].max().item()
        if (recalls >= threshold).any()
        else 0.0
        for threshold in torch.linspace(0.0, 1.0, 11)
    ) / 11.0


def _compute_ap_auc(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = torch.maximum(mpre[index], mpre[index + 1])
    indices = torch.where(mrec[1:] != mrec[:-1])[0]
    return ((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]).sum().item()


def _compute_class_ap(
    pred_list: List[Dict],
    gt_list: List[Dict],
    iou_thresh: float = 0.5,
    interp: str = "voc11",
) -> float:
    num_gt = sum(len(item["boxes"]) for item in gt_list)
    if num_gt == 0:
        return float("nan")
    scores: List[float] = []
    true_positives: List[int] = []
    for prediction, target in zip(pred_list, gt_list):
        pred_boxes, pred_scores, gt_boxes = (
            prediction["boxes"], prediction["scores"], target["boxes"]
        )
        if len(pred_boxes) == 0:
            continue
        order = pred_scores.argsort(descending=True)
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        for index in order:
            scores.append(float(pred_scores[index]))
            if len(gt_boxes) == 0:
                true_positives.append(0)
                continue
            ious = box_iou(pred_boxes[index].unsqueeze(0), gt_boxes)[0]
            best_iou, best_index = ious.max(dim=0)
            is_true_positive = best_iou >= iou_thresh and not matched[best_index]
            true_positives.append(int(is_true_positive))
            if is_true_positive:
                matched[best_index] = True
    if not scores:
        return 0.0
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    tp = torch.tensor([true_positives[index] for index in order], dtype=torch.float32)
    cumulative_tp = tp.cumsum(0)
    cumulative_fp = (1.0 - tp).cumsum(0)
    recalls = cumulative_tp / num_gt
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-9)
    return (_compute_ap_voc11 if interp == "voc11" else _compute_ap_auc)(
        recalls, precisions
    )


class DetectionEvaluator:
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
        interp: str = "voc11",
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or [f"cls_{i}" for i in range(num_classes)]
        self.iou_thresholds = iou_thresholds or [0.5]
        self.interp = interp
        self.reset()

    def reset(self) -> None:
        self._preds: List[Dict] = []
        self._gts: List[Dict] = []

    def update(self, predictions, targets) -> None:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have equal batch length")
        for prediction, target in zip(predictions, targets):
            self._preds.append({
                key: value.detach().cpu()
                for key, value in prediction.items()
                if isinstance(value, torch.Tensor)
            })
            self._gts.append({
                key: value.detach().cpu()
                for key, value in target.items()
                if isinstance(value, torch.Tensor)
            })

    def compute(self) -> Dict[str, float]:
        if not self._preds:
            return {"mAP@0.5": 0.0}
        by_class_predictions = defaultdict(lambda: [{} for _ in self._preds])
        by_class_targets = defaultdict(lambda: [{} for _ in self._gts])
        for image_index, (prediction, target) in enumerate(zip(self._preds, self._gts)):
            for class_index in range(self.num_classes):
                pred_mask = prediction["labels"] == class_index
                target_mask = target["labels"] == class_index
                by_class_predictions[class_index][image_index] = {
                    "boxes": prediction["boxes"][pred_mask],
                    "scores": prediction["scores"][pred_mask],
                }
                by_class_targets[class_index][image_index] = {
                    "boxes": target["boxes"][target_mask]
                }
        results: Dict[str, float] = {}
        maps = []
        for threshold in self.iou_thresholds:
            class_aps = []
            for class_index in range(self.num_classes):
                ap = _compute_class_ap(
                    by_class_predictions[class_index],
                    by_class_targets[class_index],
                    threshold,
                    self.interp,
                )
                if ap == ap:
                    class_aps.append(ap)
                    if threshold == 0.5:
                        results[f"AP@0.5/{self.class_names[class_index]}"] = round(ap, 4)
            mean_ap = sum(class_aps) / len(class_aps) if class_aps else 0.0
            maps.append(mean_ap)
            if threshold == 0.5:
                results["mAP@0.5"] = round(mean_ap, 4)
        if len(self.iou_thresholds) > 1:
            results["mAP@0.5:0.95"] = round(sum(maps) / len(maps), 4)
        return results


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


class PseudoLabelEvaluator:
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
        if not full_dataset and tuple(sample_ids) != self.fixed_sample_ids:
            raise ValueError("fixed pseudo-label evaluation image order changed")
        return {
            "scope": "full_ir_validation" if full_dataset else "fixed_ir_validation",
            "images": len(sample_ids), "batches": len(batch_metrics),
            "batch_size": loader.batch_size, "sample_ids": sample_ids,
            "metrics": aggregate_pseudo_label_metrics(batch_metrics),
        }


class TrainingEvaluator:
    """Own validation/audit scheduling and progress; the trainer owns best metrics."""

    def __init__(self, config, *, evaluator=None, ir_val_loader=None,
                 rgb_val_loader=None, teachers=None, pseudo_label_evaluator=None):
        self.config = config
        self.evaluator = evaluator
        self.ir_val_loader = ir_val_loader
        self.rgb_val_loader = rgb_val_loader
        self.device = torch.device(config.device)
        self.eval_every_n = config.eval_period
        self.teachers = teachers
        self.pseudo_label_evaluator = pseudo_label_evaluator
        self._last_phase = None
        self.history = []
        self.last_pseudo_label_eval_step = 0
        self.last_full_pseudo_label_eval_step = 0

    def state_dict(self):
        return {
            "last_phase": self._last_phase,
            "history": copy.deepcopy(self.history),
            "last_pseudo_label_eval_step": self.last_pseudo_label_eval_step,
            "last_full_pseudo_label_eval_step": self.last_full_pseudo_label_eval_step,
            "pseudo_label_evaluator": (
                None if self.pseudo_label_evaluator is None
                else self.pseudo_label_evaluator.state_dict()
            ),
        }

    def load_state_dict(self, state):
        self._last_phase = state.get("last_phase")
        self.history = copy.deepcopy(state.get("history", []))
        self.last_pseudo_label_eval_step = state.get("last_pseudo_label_eval_step", 0)
        self.last_full_pseudo_label_eval_step = state.get("last_full_pseudo_label_eval_step", 0)
        audit_state = state.get("pseudo_label_evaluator")
        if self.pseudo_label_evaluator is not None and audit_state is not None:
            self.pseudo_label_evaluator.load_state_dict(audit_state)

    def step(self, model, global_step, current_phase):
        phase_changed = self._last_phase is not None and current_phase != self._last_phase
        periodic = global_step > 0 and global_step % self.eval_every_n == 0
        if phase_changed or periodic:
            return self.evaluate(model, global_step, current_phase)
        self._last_phase = current_phase
        return None

    def evaluate(self, model, global_step, current_phase, trigger_reason="manual"):
        if self.evaluator is None:
            return None
        results = {"global_step": global_step, "phase": current_phase.name, "trigger": trigger_reason}
        if self.ir_val_loader is not None:
            results.update(self._run_eval_on_loader(model, self.ir_val_loader, "IR"))
        if self.rgb_val_loader is not None and current_phase.name in {
            "PHASE1_RGB_WARMUP", "PHASE2_TRANSITION"
        }:
            rgb_results = self._run_eval_on_loader(model, self.rgb_val_loader, "RGB")
            results.update({f"rgb_{key}": value for key, value in rgb_results.items()})
        self.history.append(results)
        self._last_phase = current_phase
        return results

    def audit(self, global_step, current_phase, *, ema_initialized, final=False):
        audit = self.config.pseudo_label_eval
        if self.pseudo_label_evaluator is None or not audit.enabled or not ema_initialized:
            return []
        if final:
            if global_step < self.config.total_iters or self.last_full_pseudo_label_eval_step == global_step:
                return []
        elif global_step // audit.interval <= self.last_pseudo_label_eval_step // audit.interval:
            return []
        records = []
        for name in configured_teacher_names(self.config):
            settings = distillation_settings(self.config, name, current_phase)
            result = self.pseudo_label_evaluator.evaluate(
                self.teachers[name], settings, iou_threshold=audit.iou_threshold,
                resize_short_edge=self.config.model.min_sizes[-1], full_dataset=final,
            )
            records.append({
                **result, "global_step": global_step, "phase": current_phase.name,
                "source_domain": "ir", "teacher": name,
                "trigger": "final" if final else "periodic",
                "top_ratio": settings.top_ratio, "min_hm": settings.min_hm,
                "hm_alpha": settings.hm_alpha, "hm_beta": settings.hm_beta,
                "resize_short_edge": self.config.model.min_sizes[-1],
                "iou_threshold": audit.iou_threshold,
            })
        for record in records:
            log_pseudo_label_evaluation(record, output_dir=self.config.output_dir, logger=logger)
        if final:
            self.last_full_pseudo_label_eval_step = global_step
        else:
            self.last_pseudo_label_eval_step = global_step
        return records

    def _run_eval_on_loader(self, model, loader, domain_name) -> Dict:
        self.evaluator.reset()
        started = time.time()
        with _preserve_evaluation_state(model, getattr(loader, "generator", None)), torch.inference_mode():
            for batch in loader:
                images, targets, sample_ids = batch
                images = images.to(self.device) if isinstance(images, torch.Tensor) else images
                targets = [
                    {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                     for key, value in target.items()}
                    for target in targets
                ]
                predictions = model(images, sample_ids=sample_ids)
                self.evaluator.update(predictions, targets)
        metrics = self.evaluator.compute()
        logger.info(
            "[%s Val] mAP@0.5: %.4f (%.1fs)",
            domain_name,
            metrics.get("mAP@0.5", 0.0),
            time.time() - started,
        )
        return metrics

    def print_history(self) -> None:
        for result in self.history:
            logger.info(
                "step=%s phase=%s mAP@0.5=%.4f",
                result.get("global_step"),
                result.get("phase"),
                result.get("mAP@0.5", 0.0),
            )

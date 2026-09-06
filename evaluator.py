"""Small dependency-free detection evaluator and phase callback."""

from collections import defaultdict
import logging
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

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


class PhaseEvaluator:
    """Evaluate domains and invoke the best callback exactly on improvement."""

    def __init__(
        self,
        evaluator: DetectionEvaluator,
        ir_val_loader: Optional[DataLoader],
        device: torch.device,
        rgb_val_loader: Optional[DataLoader] = None,
        eval_every_n: Optional[int] = 500,
        vis_dir: Optional[str] = None,
    ) -> None:
        self.evaluator = evaluator
        self.ir_val_loader = ir_val_loader
        self.rgb_val_loader = rgb_val_loader
        self.device = device
        self.eval_every_n = eval_every_n
        self.vis_dir = vis_dir
        self.best_ir_map = -1.0
        self.best_rgb_map = -1.0
        self._last_phase = None
        self.history = []
        self.on_new_best_fn = None

    def register_best_fn(self, fn) -> None:
        self.on_new_best_fn = fn

    def step(self, model, global_step: int, current_phase) -> Optional[Dict]:
        phase_changed = self._last_phase is not None and current_phase != self._last_phase
        periodic = self.eval_every_n is not None and global_step > 0 and global_step % self.eval_every_n == 0
        self._last_phase = current_phase
        if phase_changed or periodic:
            return self.evaluate(model, global_step, current_phase)
        return None

    @staticmethod
    def _phase_name(phase) -> str:
        return phase.name if hasattr(phase, "name") else str(phase)

    def evaluate(self, model, global_step: int, current_phase, trigger_reason: str = "manual") -> Dict:
        was_training = model.training
        model.eval()
        phase_name = self._phase_name(current_phase)
        results = {"global_step": global_step, "phase": phase_name, "trigger": trigger_reason}
        try:
            if self.ir_val_loader is not None:
                ir_results = self._run_eval_on_loader(model, self.ir_val_loader, "IR")
            else:
                ir_results = {"mAP@0.5": 0.0}
            results.update(ir_results)
            if ir_results.get("mAP@0.5", 0.0) > self.best_ir_map:
                self.best_ir_map = ir_results["mAP@0.5"]
                results["is_best_ir"] = True
                if self.on_new_best_fn is not None:
                    self.on_new_best_fn(results)

            if self.rgb_val_loader is not None and phase_name in {
                "PHASE1_RGB_WARMUP", "PHASE2_TRANSITION"
            }:
                rgb_results = self._run_eval_on_loader(model, self.rgb_val_loader, "RGB")
                results.update({f"rgb_{key}": value for key, value in rgb_results.items()})
                if rgb_results.get("mAP@0.5", 0.0) > self.best_rgb_map:
                    self.best_rgb_map = rgb_results["mAP@0.5"]
                    results["is_best_rgb"] = True
            self.history.append(results)
            return results
        finally:
            model.train(was_training)

    @staticmethod
    def _split_batch(batch):
        if len(batch) == 3:
            return batch[0], batch[1], tuple(batch[2])
        if len(batch) == 2:
            return batch[0], batch[1], tuple(
                target.get("stem", str(index))
                for index, target in enumerate(batch[1])
            )
        raise ValueError("evaluation batch must contain images and targets")

    def _run_eval_on_loader(self, model, loader, domain_name) -> Dict:
        self.evaluator.reset()
        started = time.time()
        with torch.no_grad():
            for batch in loader:
                images, targets, sample_ids = self._split_batch(batch)
                images = images.to(self.device) if isinstance(images, torch.Tensor) else images
                targets = [
                    {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                     for key, value in target.items()}
                    for target in targets
                ]
                try:
                    predictions = model(images, sample_ids=sample_ids)
                except TypeError:
                    predictions = model(images)
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

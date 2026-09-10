"""Evaluate a D3T FCOS checkpoint with the official COCO detection metric."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import FCOSModelConfig, TrainingConfig
from data import (
    FLIRIRValDataset,
    FLIRRGBDataset,
    ir_val_collate,
    rgb_collate,
)
from models.fcos_factory import CHECKPOINT_SCHEMA_VERSION, build_fcos_d3t_model


logger = logging.getLogger(__name__)


def _xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_inputs(predictions, targets, class_names: Sequence[str]):
    """Convert zero-based FLIR tensors into one-based COCO dictionaries."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same images")
    categories = [
        {"id": index + 1, "name": name}
        for index, name in enumerate(class_names)
    ]
    images = [{"id": index + 1} for index in range(len(targets))]
    annotations = []
    detections = []
    annotation_id = 1

    for image_index, (prediction, target) in enumerate(
        zip(predictions, targets), start=1
    ):
        for box, label in zip(target["boxes"], target["labels"]):
            category_id = int(label) + 1
            if not 1 <= category_id <= len(categories):
                raise ValueError("target label is outside the configured class range")
            xywh = _xyxy_to_xywh(box.tolist())
            annotations.append({
                "id": annotation_id,
                "image_id": image_index,
                "category_id": category_id,
                "bbox": xywh,
                "area": xywh[2] * xywh[3],
                "iscrowd": 0,
            })
            annotation_id += 1

        for box, score, label in zip(
            prediction["boxes"], prediction["scores"], prediction["labels"]
        ):
            category_id = int(label) + 1
            if not 1 <= category_id <= len(categories):
                raise ValueError("prediction label is outside the configured class range")
            detections.append({
                "image_id": image_index,
                "category_id": category_id,
                "bbox": _xyxy_to_xywh(box.tolist()),
                "score": float(score),
            })

    dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return dataset, detections


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else float("nan")


def summarize_coco_eval(coco_eval, class_names: Sequence[str]) -> dict[str, float]:
    """Return standard COCO summary metrics plus AP for each category."""

    stat_names = (
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    )
    metrics = {
        name: float(coco_eval.stats[index])
        for index, name in enumerate(stat_names)
    }
    precision = coco_eval.eval["precision"]  # [IoU, recall, class, area, maxDet]
    area_index = list(coco_eval.params.areaRngLbl).index("all")
    max_det_index = len(coco_eval.params.maxDets) - 1
    iou_thresholds = np.asarray(coco_eval.params.iouThrs)

    for class_index, class_name in enumerate(class_names):
        class_precision = precision[:, :, class_index, area_index, max_det_index]
        metrics[f"AP/{class_name}"] = _mean_valid(class_precision)
        for threshold, label in ((0.5, "AP50"), (0.75, "AP75")):
            matches = np.flatnonzero(np.isclose(iou_thresholds, threshold))
            metrics[f"{label}/{class_name}"] = (
                _mean_valid(class_precision[matches[0]])
                if matches.size
                else float("nan")
            )
    return metrics


def _load_checkpoint_model(checkpoint_path: str, device: torch.device, score_thresh: float):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no D3T metadata")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema is not supported")
    saved_model = metadata.get("config", {}).get("model", {})
    class_names = tuple(metadata.get("class_names", ()))
    if not class_names or int(metadata.get("num_classes", 0)) != len(class_names):
        raise ValueError("checkpoint class metadata is invalid")

    model_config = FCOSModelConfig(
        num_classes=len(class_names),
        class_names=class_names,
        weights=None,
        classification_init_mode=metadata["classification_init_mode"],
        pretrained_backbone=False,
        trainable_backbone_layers=int(saved_model.get("trainable_backbone_layers", 3)),
        min_sizes=tuple(saved_model.get("min_sizes", (640, 672, 704, 736, 768, 800))),
        max_size=int(saved_model.get("max_size", 1333)),
        center_sampling_radius=float(saved_model.get("center_sampling_radius", 0.0)),
        score_thresh=score_thresh,
        nms_thresh=float(saved_model.get("nms_thresh", 0.6)),
        topk_candidates=int(saved_model.get("topk_candidates", 1000)),
        detections_per_img=100,
        ir_residual_neck_enabled=bool(
            saved_model.get("ir_residual_neck_enabled", False)
        ),
        ir_residual_bottleneck_channels=int(
            saved_model.get("ir_residual_bottleneck_channels", 64)
        ),
        ir_residual_norm_groups=int(
            saved_model.get("ir_residual_norm_groups", 16)
        ),
        vfl_alpha=float(saved_model.get("vfl_alpha", 0.75)),
        vfl_gamma=float(saved_model.get("vfl_gamma", 2.0)),
        vfl_weight_type=str(saved_model.get("vfl_weight_type", "iou")),
    )
    config = TrainingConfig(
        model=model_config,
        teacher_mode=str(metadata.get("teacher_mode", "two_teacher")),
        device=str(device),
    )
    model = build_fcos_d3t_model(config)
    model.load_state_dict(checkpoint["student"])
    model.to(device).eval()
    return model, class_names


def _build_loader(data_root: str, domain: str, batch_size: int, workers: int):
    if domain == "ir":
        dataset = FLIRIRValDataset(data_root, split="validation")
        collate_fn = ir_val_collate
    else:
        dataset = FLIRRGBDataset(data_root, split="validation")
        collate_fn = rgb_collate
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def _collect(model, loader, device: torch.device, domain: str):
    predictions = []
    targets = []
    with torch.inference_mode():
        for batch_index, (images, batch_targets, sample_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            output = model(
                images,
                sample_ids=sample_ids,
                domain=domain,
            )
            predictions.extend(
                {key: value.detach().cpu() for key, value in item.items()}
                for item in output
            )
            targets.extend(
                {
                    key: value.detach().cpu()
                    for key, value in item.items()
                    if isinstance(value, torch.Tensor)
                }
                for item in batch_targets
            )
            if batch_index % 50 == 0:
                logger.info("Evaluated %d/%d batches", batch_index, len(loader))
    return predictions, targets


def evaluate(args) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "COCO evaluation requires pycocotools; install it with "
            "'pip install pycocotools' inside the training environment"
        ) from error

    device = torch.device(args.device)
    model, class_names = _load_checkpoint_model(
        args.checkpoint, device, args.score_thresh
    )
    loader = _build_loader(
        args.data_root, args.domain, args.batch_size, args.workers
    )
    logger.info(
        "COCO evaluation: checkpoint=%s domain=%s images=%d device=%s",
        args.checkpoint, args.domain.upper(), len(loader.dataset), device,
    )
    predictions, targets = _collect(
        model,
        loader,
        device,
        domain=args.domain,
    )
    dataset, detections = build_coco_inputs(predictions, targets, class_names)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": dataset["images"],
            "categories": dataset["categories"],
            "annotations": [],
        }
        coco_dt.createIndex()
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = list(range(1, len(class_names) + 1))
    coco_eval.params.imgIds = list(range(1, len(targets) + 1))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    metrics = summarize_coco_eval(coco_eval, class_names)

    print("\nPer-class COCO metrics")
    for class_name in class_names:
        print(
            f"{class_name:>10}: "
            f"AP={metrics[f'AP/{class_name}']:.4f}  "
            f"AP50={metrics[f'AP50/{class_name}']:.4f}  "
            f"AP75={metrics[f'AP75/{class_name}']:.4f}"
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("Metrics saved -> %s", output_path)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a D3T FCOS checkpoint with official COCO bbox metrics"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", "--data_root", dest="data_root", required=True)
    parser.add_argument("--domain", choices=("ir", "rgb"), default="ir")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.001,
        help="Low threshold preserves candidates needed by COCO AP ranking",
    )
    parser.add_argument("--output", default=None, help="Optional metrics JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    evaluate(parse_args())

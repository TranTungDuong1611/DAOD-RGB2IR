"""
Loss functions for each domain step.

Expected detector API (Faster-RCNN / FCOS / DINO style):
  Training:  model(images, targets) → Dict[str, Tensor]  (named loss components)
  Inference: model(images)          → List[Dict]          (boxes, labels, scores)

All teacher forward passes run under torch.no_grad() — callers must not wrap
this module in no_grad() since the student path needs gradients.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import HarmonyConfig, LossConfig

try:
    from torchvision.ops import box_iou as _tv_box_iou
    _HAS_TORCHVISION_IOU = True
except ImportError:
    _HAS_TORCHVISION_IOU = False

# Global or per-class confidence threshold
ThreshType = Union[float, Dict[int, float]]

_DEFAULT_THRESH = 0.7   # fallback for classes absent from per-class dict

# RPN-specific loss keys in torchvision Faster R-CNN loss dict
_RPN_LOSS_KEYS = frozenset({"loss_objectness", "loss_rpn_box_reg"})


# ---------------------------------------------------------------------------
# Pseudo-label filtering (original — unchanged for backward compat)
# ---------------------------------------------------------------------------

def filter_pseudo_labels(
    predictions: List[Dict[str, torch.Tensor]],
    conf_thresh: ThreshType = 0.7,
) -> List[Dict[str, torch.Tensor]]:
    """
    Filter teacher predictions by confidence score.

    Args:
        predictions : output of model(images) in inference mode
                      each dict has "boxes" [N,4], "labels" [N], "scores" [N]
        conf_thresh : float  — global threshold applied to all classes
                      Dict[int, float] — per-class threshold; classes absent
                      from the dict fall back to _DEFAULT_THRESH (0.7)

    Returns:
        filtered list of dicts (same length as predictions, empty dicts possible)
    """
    pseudo = []
    for pred in predictions:
        scores = pred.get("scores", torch.zeros(0))
        if scores.numel() == 0:
            pseudo.append({
                "boxes":  torch.zeros(0, 4, device=scores.device),
                "labels": torch.zeros(0, dtype=torch.long, device=scores.device),
                "scores": scores,
            })
            continue

        if isinstance(conf_thresh, dict):
            labels = pred["labels"]
            keep = torch.tensor(
                [scores[i].item() >= conf_thresh.get(int(labels[i].item()), _DEFAULT_THRESH)
                 for i in range(len(scores))],
                dtype=torch.bool,
                device=scores.device,
            )
        else:
            keep = scores >= conf_thresh

        pseudo.append({
            "boxes":  pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
        })
    return pseudo


def filter_pseudo_labels_with_harmony(
    predictions: List[Dict[str, torch.Tensor]],
    conf_thresh: ThreshType = 0.7,
    harmony_cfg: Optional[HarmonyConfig] = None,
    gt_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, float]]:
    """
    Confidence-filter teacher predictions and optionally attach harmony weights.

    When harmony_cfg.use_harmony_weight=False (default), this is equivalent to
    filter_pseudo_labels() but also returns logging metrics.

    gt_targets : optional GT for the same batch. When provided, localization
                 quality u_i = max IoU(pseudo_box_i, GT_boxes) [supervised path].
                 When None, u_i = max_{j≠i} IoU(box_i, box_j) [paper Eq.9].

    Returns
    -------
    pseudo_targets : List[Dict] — each dict has boxes/labels/scores and,
                     when harmony is enabled, "harmony_weights" [N] in [0,1].
    log            : Dict[str, float] — counts and harmony statistics.
    """
    if harmony_cfg is None:
        harmony_cfg = HarmonyConfig()

    pseudo: List[Dict[str, torch.Tensor]] = []
    log: Dict[str, float] = {}

    total_before = 0
    total_after_conf = 0
    total_after_harmony = 0
    all_h: List[torch.Tensor] = []
    all_p: List[torch.Tensor] = []
    all_u: List[torch.Tensor] = []

    for idx, pred in enumerate(predictions):
        scores = pred.get("scores", torch.zeros(0))
        boxes  = pred.get("boxes",  torch.zeros(0, 4, device=scores.device))
        labels = pred.get("labels", torch.zeros(0, dtype=torch.long, device=scores.device))

        total_before += scores.numel()

        if scores.numel() == 0:
            pseudo.append({
                "boxes":  boxes,
                "labels": labels,
                "scores": scores,
            })
            continue

        # ── Step 1: confidence threshold ──────────────────────────────────
        if isinstance(conf_thresh, dict):
            keep = torch.tensor(
                [scores[i].item() >= conf_thresh.get(int(labels[i].item()), _DEFAULT_THRESH)
                 for i in range(len(scores))],
                dtype=torch.bool, device=scores.device,
            )
        else:
            keep = scores >= conf_thresh

        boxes_f  = boxes[keep]
        labels_f = labels[keep]
        scores_f = scores[keep]
        n_conf   = int(keep.sum().item())
        total_after_conf += n_conf

        if not harmony_cfg.use_harmony_weight:
            pseudo.append({"boxes": boxes_f, "labels": labels_f, "scores": scores_f})
            continue

        if n_conf == 0:
            pseudo.append({"boxes": boxes_f, "labels": labels_f, "scores": scores_f,
                           "harmony_weights": scores_f.new_zeros(0)})
            continue

        # ── Step 2: localization quality u_i ─────────────────────────────
        gt_boxes = gt_targets[idx]["boxes"] if gt_targets is not None else None
        u = compute_localization_quality_u(boxes_f, scores_f, gt_boxes=gt_boxes)
        h = compute_harmony_score(scores_f, u, beta=harmony_cfg.beta)

        all_h.append(h);  all_p.append(scores_f.float());  all_u.append(u)

        # ── Step 3: optional harmony-score threshold ───────────────────────
        if harmony_cfg.min_threshold is not None and harmony_cfg.min_threshold > 0.0:
            h_keep   = h >= harmony_cfg.min_threshold
            boxes_f  = boxes_f[h_keep]
            labels_f = labels_f[h_keep]
            scores_f = scores_f[h_keep]
            h        = h[h_keep]

        # ── Step 4: optional max-boxes-per-image cap ───────────────────────
        if harmony_cfg.max_boxes_per_image is not None and h.numel() > harmony_cfg.max_boxes_per_image:
            _, top_idx = h.topk(harmony_cfg.max_boxes_per_image)
            boxes_f  = boxes_f[top_idx]
            labels_f = labels_f[top_idx]
            scores_f = scores_f[top_idx]
            h        = h[top_idx]

        total_after_harmony += h.numel()
        pseudo.append({
            "boxes":           boxes_f,
            "labels":          labels_f,
            "scores":          scores_f,
            "harmony_weights": h,          # detached, in [0,1]
        })

    # ── Logging ───────────────────────────────────────────────────────────
    log["n_pseudo_before_thresh"]     = float(total_before)
    log["n_pseudo_after_conf_thresh"] = float(total_after_conf)
    if harmony_cfg.use_harmony_weight:
        log["n_pseudo_after_harmony_thresh"] = float(total_after_harmony)
        if all_h:
            h_cat = torch.cat(all_h)
            p_cat = torch.cat(all_p)
            u_cat = torch.cat(all_u)
            log["harmony_mean"] = h_cat.mean().item()
            log["harmony_std"]  = h_cat.std().item() if h_cat.numel() > 1 else 0.0
            log["harmony_min"]  = h_cat.min().item()
            log["harmony_max"]  = h_cat.max().item()
            log["pseudo_p_mean"] = p_cat.mean().item()
            log["pseudo_u_mean"] = u_cat.mean().item()

    return pseudo, log


def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum all scalar tensors in a detector loss dict."""
    return sum(v for v in loss_dict.values())


def _sum_pseudo_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    harmony_cfg: Optional[HarmonyConfig] = None,
) -> torch.Tensor:
    """
    Sum a pseudo-label loss dict, optionally scaling RPN losses by a separate
    factor or excluding them entirely.

    When harmony_cfg is None OR use_harmony_weight=False, behaves identically
    to _sum_loss_dict — guarantees exact baseline behavior.
    When use_harmony_weight=True and use_rpn_pseudo=False, RPN losses are dropped.
    When use_harmony_weight=True and rpn_pseudo_factor != 1.0, RPN losses are scaled.
    """
    if harmony_cfg is None or not harmony_cfg.use_harmony_weight:
        return _sum_loss_dict(loss_dict)

    total = sum(
        v * (harmony_cfg.rpn_pseudo_factor if k in _RPN_LOSS_KEYS else 1.0)
        for k, v in loss_dict.items()
        if harmony_cfg.use_rpn_pseudo or k not in _RPN_LOSS_KEYS
    )
    return total


# ---------------------------------------------------------------------------
# Harmony weight computation
# ---------------------------------------------------------------------------

def _box_iou_safe(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU between boxes_a [M,4] and boxes_b [N,4] → [M,N]."""
    if _HAS_TORCHVISION_IOU:
        return _tv_box_iou(boxes_a, boxes_b)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp(min=0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp(min=0).prod(dim=-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def compute_localization_quality_u(
    boxes: torch.Tensor,                          # [N, 4] post-NMS teacher boxes
    scores: torch.Tensor,                         # [N]   teacher confidence
    gt_boxes: Optional[torch.Tensor] = None,      # [M, 4] GT boxes, if available
) -> torch.Tensor:
    """
    Localization quality proxy u_i (Harmonious Teacher, Eq. 5 & 9).

    Supervised path  (gt_boxes provided):
        u_i = max_j IoU(pseudo_box_i, gt_box_j)
        Exact IoU against annotation — reflects true localization quality.

    Unsupervised path (gt_boxes=None, paper Eq. 9):
        u_i = max_{j≠i} IoU(pseudo_box_i, pseudo_box_j)
        A predicted box that overlaps with other predicted boxes is more
        likely to be on a real object than an isolated spurious detection.
        Fallback: single-box image → u_i = p_i (confidence).

    Returns u: Tensor[N] in [0, 1], float32, detached.
    """
    N = boxes.shape[0]
    if N == 0:
        return boxes.new_zeros(0)

    if gt_boxes is not None and gt_boxes.numel() > 0:
        # Supervised: exact IoU against GT
        ious = _box_iou_safe(boxes, gt_boxes.to(boxes.device))  # [N, M]
        u = ious.max(dim=1).values                               # [N]
    elif N == 1:
        # Single pseudo-box, no peers to compare against
        u = scores.float().clamp(0.0, 1.0)
    else:
        # Unsupervised heuristic (paper Eq. 9): max IoU with any other box
        ious = _box_iou_safe(boxes, boxes)   # [N, N]
        ious.fill_diagonal_(0.0)             # exclude self
        u = ious.max(dim=1).values           # [N]

    return u.float().clamp(0.0, 1.0).detach()


def compute_harmony_score(
    scores: torch.Tensor,  # p_i [N]
    u: torch.Tensor,       # u_i [N]
    beta: float = 0.5,
) -> torch.Tensor:
    """
    h_i = p_i^beta * u_i^(1-beta)   (Harmonious Teacher, Eq. 5)

    beta=1.0 → h=p (confidence only)
    beta=0.0 → h=u (localization quality only)
    beta=0.5 → geometric mean sqrt(p * u)

    Detached — no gradient flows through teacher outputs.
    """
    h = scores.float().pow(beta) * u.pow(1.0 - beta)
    return h.clamp(0.0, 1.0).detach()


# ---------------------------------------------------------------------------
# RGB step loss
# ---------------------------------------------------------------------------

def compute_rgb_loss(
    student: nn.Module,
    images: torch.Tensor,
    gt_targets: List[Dict[str, torch.Tensor]],
    rgb_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    RGB supervised step:
      - mandatory:  supervised GT loss (student sees strong aug)
      - optional:   pseudo-label loss from rgb_teacher (rgb_pseudo_weight > 0)

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on images (strong aug).
                    If None, both use images.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else images

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    # --- Supervised GT loss (always active) ---
    gt_loss_dict = student(images, gt_targets)
    gt_loss = _sum_loss_dict(gt_loss_dict) * config.rgb_gt_weight
    components.append(gt_loss)
    log["rgb_gt_loss"] = gt_loss.item()

    # --- Optional pseudo-label loss from rgb_teacher ---
    if rgb_teacher is not None and config.rgb_pseudo_weight > 0.0:
        with torch.no_grad():
            pseudo_preds = rgb_teacher(t_images)

        # GT available → supervised u_i = IoU(pseudo_box, GT_box)
        pseudo_targets, harmony_log = filter_pseudo_labels_with_harmony(
            pseudo_preds, conf_thresh, config.harmony, gt_targets=gt_targets
        )
        log.update({f"rgb_{k}": v for k, v in harmony_log.items()})

        pseudo_loss_dict = student(images, pseudo_targets)
        pseudo_loss = _sum_pseudo_loss_dict(
            pseudo_loss_dict, config.harmony
        ) * config.rgb_pseudo_weight
        components.append(pseudo_loss)
        log["rgb_pseudo_loss"] = pseudo_loss.item()

    total_loss = sum(components)
    log["rgb_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# MID step loss
# ---------------------------------------------------------------------------

def compute_mid_loss(
    student: nn.Module,
    mid_images: torch.Tensor,                              # student sees this (strong aug)
    rgb_teacher: nn.Module,
    ir_teacher: nn.Module,
    gt_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_source: str = "both",                          # "rgb" | "ir" | "both"
    rgb_weight_override: Optional[float] = None,
    ir_weight_override: Optional[float] = None,
    teacher_images: Optional[torch.Tensor] = None,         # teacher sees this (weak aug)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MID (intermediate domain) step:
      - student receives SAGA-transformed images
      - learns from rgb_teacher pseudo-labels (mid_rgb_weight)
      - learns from ir_teacher  pseudo-labels (mid_ir_weight)
      - optional: GT loss if gt_targets provided  (mid_gt_weight)

    Both teacher forward passes run under no_grad.
    Student forward passes require grad.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    rgb_w = rgb_weight_override if rgb_weight_override is not None else config.mid_rgb_weight
    ir_w  = ir_weight_override  if ir_weight_override  is not None else config.mid_ir_weight

    # Teacher infers on weak images; student trains on strong images
    t_images = teacher_images if teacher_images is not None else mid_images

    # --- rgb_teacher pseudo-labels (weak) → student loss (strong) ---
    if teacher_source in ("rgb", "both") and rgb_w > 0.0:
        with torch.no_grad():
            rgb_preds = rgb_teacher(t_images)
        # GT available when gt_targets passed → supervised u_i path
        rgb_pseudo, harmony_log = filter_pseudo_labels_with_harmony(
            rgb_preds, conf_thresh, config.harmony, gt_targets=gt_targets
        )
        log.update({f"mid_rgb_{k}": v for k, v in harmony_log.items()})

        loss_dict = student(mid_images, rgb_pseudo)
        loss = _sum_pseudo_loss_dict(loss_dict, config.harmony) * rgb_w
        components.append(loss)
        log["mid_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo-labels (weak) → student loss (strong) ---
    if teacher_source in ("ir", "both") and ir_w > 0.0:
        with torch.no_grad():
            ir_preds = ir_teacher(t_images)
        # GT available when gt_targets passed → supervised u_i path
        ir_pseudo, harmony_log = filter_pseudo_labels_with_harmony(
            ir_preds, conf_thresh, config.harmony, gt_targets=gt_targets
        )
        log.update({f"mid_ir_{k}": v for k, v in harmony_log.items()})

        loss_dict = student(mid_images, ir_pseudo)
        loss = _sum_pseudo_loss_dict(loss_dict, config.harmony) * ir_w
        components.append(loss)
        log["mid_ir_teacher_loss"] = loss.item()

    # --- Optional GT loss (uses original RGB GT mapped to MID space) ---
    if gt_targets is not None and config.mid_gt_weight > 0.0:
        loss_dict = student(mid_images, gt_targets)
        loss = _sum_loss_dict(loss_dict) * config.mid_gt_weight
        components.append(loss)
        log["mid_gt_loss"] = loss.item()

    if not components:
        # Safety: all weights are zero — return a zero-grad loss
        total_loss = mid_images.sum() * 0.0
    else:
        total_loss = sum(components)

    log["mid_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# IR step loss
# ---------------------------------------------------------------------------

def compute_ir_loss(
    student: nn.Module,
    ir_images: torch.Tensor,
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    IR (target domain) unsupervised step — ir_teacher pseudo-labels only.

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on ir_images (strong aug).
                    If None, both use ir_images.

    Returns:
        total_loss : scalar tensor with grad
        log_dict   : float-valued metrics for logging
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else ir_images
    log: Dict[str, float] = {}

    with torch.no_grad():
        ir_preds = ir_teacher(t_images)
    ir_pseudo, harmony_log = filter_pseudo_labels_with_harmony(
        ir_preds, conf_thresh, config.harmony
    )
    log.update({f"ir_{k}": v for k, v in harmony_log.items()})

    loss_dict  = student(ir_images, ir_pseudo)
    total_loss = _sum_pseudo_loss_dict(loss_dict, config.harmony) * config.ir_ir_teacher_weight

    log["ir_ir_teacher_loss"] = total_loss.item()
    log["ir_total_loss"]      = total_loss.item()
    return total_loss, log

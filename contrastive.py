"""
contrastive.py — Object-level supervised contrastive loss (CMT-style).

Used in:
  Phase 2 : student (strong RGB/MID) vs rgb_teacher (weak RGB/MID)  — GT labels
  Phase 3 : student (strong MID)     vs rgb_teacher (weak MID)      — GT labels
             student (strong IR)     vs ir_teacher  (weak IR)       — pseudo-labels

Coordinate convention:
  boxes_orig  — boxes in the image space AFTER geometric aug, BEFORE
                GeneralizedRCNNTransform (same space as aug_targets / pseudo-labels).
  trans_w/h   — image dimensions AFTER GeneralizedRCNNTransform (FPN tensor space).
  spatial_scale for roi_align = feat_w / trans_w.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. NeurIPS 2020), contrast_mode='one'.

    Only student features serve as anchors; teacher features extend the contrast
    set.  Full contrast set = [student_feats | teacher_feats] (2N).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_feats: torch.Tensor,  # [N, C]  L2-normalised, has grad
        teacher_feats: torch.Tensor,  # [N, C]  L2-normalised, no grad
        labels: torch.Tensor,          # [N]     int class labels
    ) -> torch.Tensor:
        N = student_feats.shape[0]
        if N == 0:
            return student_feats.sum() * 0.0

        device = student_feats.device
        all_feats = torch.cat([student_feats, teacher_feats], dim=0)  # [2N, C]

        # [N, 2N] cosine-similarity logits
        sim = torch.mm(student_feats, all_feats.t()) / self.temperature

        # Positive mask: same class across the full [N × 2N] contrast grid
        labels_2n = torch.cat([labels, labels])                          # [2N]
        pos_mask = labels.unsqueeze(1) == labels_2n.unsqueeze(0)         # [N, 2N]

        # Exclude self-comparison (student_i vs student_i at column i)
        self_mask = torch.zeros(N, 2 * N, dtype=torch.bool, device=device)
        self_mask[:, :N] = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask &= ~self_mask

        # Numerically-stable log-softmax (self masked out of denominator)
        sim = sim.masked_fill(self_mask, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        n_pos = pos_mask.float().sum(dim=1)   # [N]
        valid = n_pos > 0
        if not valid.any():
            return student_feats.sum() * 0.0

        loss = -(pos_mask.float() * log_prob)[valid].sum(dim=1) / n_pos[valid]
        return loss.mean()


# ---------------------------------------------------------------------------
# RoI-level feature extraction
# ---------------------------------------------------------------------------

def build_roi_boxes(
    targets: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten per-image target dicts into a single (roi_boxes, labels) pair.

    Returns:
        roi_boxes : [N, 5]  float  (batch_idx, x1, y1, x2, y2)
        labels    : [N]     long
    """
    rois, lbls = [], []
    for i, t in enumerate(targets):
        boxes = t.get("boxes")
        if boxes is None or boxes.numel() == 0:
            continue
        idx = boxes.new_full((boxes.shape[0], 1), float(i))
        rois.append(torch.cat([idx, boxes], dim=1))   # [n_i, 5]
        lbls.append(t["labels"])

    if not rois:
        dev = targets[0]["boxes"].device if targets else torch.device("cpu")
        return (
            torch.zeros((0, 5), device=dev),
            torch.zeros((0,), dtype=torch.long, device=dev),
        )
    return torch.cat(rois, dim=0).float(), torch.cat(lbls, dim=0)


def extract_object_features(
    fpn_features: Dict[str, torch.Tensor],
    roi_boxes: torch.Tensor,    # [N, 5]  boxes in original (pre-transform) image space
    orig_h: int,
    orig_w: int,
    trans_h: int,
    trans_w: int,
    levels: Tuple[str, ...] = ("0", "1", "2"),
    output_size: int = 7,
) -> torch.Tensor:
    """
    RoI Align on FPN feature maps → [N, C] mean-pooled object features.

    Boxes live in the original-image space (after geometric aug, before
    GeneralizedRCNNTransform).  FPN features live in the transformed space.
    Both scaling steps are applied internally.
    """
    if roi_boxes.shape[0] == 0:
        C = next(iter(fpn_features.values())).shape[1]
        return torch.zeros((0, C), device=roi_boxes.device)

    # Scale boxes: original space → transformed (FPN) space
    scale_x = trans_w / orig_w
    scale_y = trans_h / orig_h
    scaled = roi_boxes.float().clone()
    scaled[:, 1] *= scale_x
    scaled[:, 2] *= scale_y
    scaled[:, 3] *= scale_x
    scaled[:, 4] *= scale_y

    per_level = []
    for k in levels:
        if k not in fpn_features:
            continue
        feat = fpn_features[k]                         # [B, C, H_f, W_f]
        spatial_scale = float(feat.shape[3]) / trans_w
        pooled = roi_align(
            feat,
            scaled,
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )                                              # [N, C, output_size, output_size]
        per_level.append(pooled.mean(dim=[-2, -1]))    # [N, C]

    if not per_level:
        C = next(iter(fpn_features.values())).shape[1]
        return torch.zeros((roi_boxes.shape[0], C), device=roi_boxes.device)

    return torch.stack(per_level, dim=0).mean(dim=0)   # [N, C]


# ---------------------------------------------------------------------------
# Pseudo-label utilities
# ---------------------------------------------------------------------------

def filter_pseudo_labels(
    preds: List[Dict[str, torch.Tensor]],
    conf_thresh: float,
) -> List[Dict[str, torch.Tensor]]:
    """
    Filter teacher predictions by confidence threshold.
    Returns list of same length as preds (empty dicts where no box passes).
    """
    filtered = []
    for p in preds:
        keep = p["scores"] >= conf_thresh
        filtered.append({
            "boxes":  p["boxes"][keep],
            "labels": p["labels"][keep],
        })
    return filtered


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def compute_contrastive_loss(
    student: nn.Module,
    teacher: nn.Module,
    student_images: torch.Tensor,                   # [B, C, H, W]  strong-aug
    teacher_images: torch.Tensor,                   # [B, C, H, W]  weak-aug
    targets: List[Dict[str, torch.Tensor]],         # GT or pre-filtered pseudo-labels
    supcon: SupConLoss,
    weight: float,
    levels: Tuple[str, ...] = ("0", "1", "2"),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full contrastive pipeline:
      1. Build roi boxes from targets
      2. Extract student and teacher FPN features (student has grad; teacher no_grad)
      3. RoI Align → [N, C] per-object vectors
      4. L2-normalise → SupConLoss

    Args:
        targets : list of {boxes, labels} — GT or high-conf pseudo-labels.
                  Must be in the same spatial space as student/teacher images
                  (i.e. produced under the same geometric aug).
        weight  : scalar multiplier for the loss.

    Returns:
        (weighted_loss, log_dict)  where log_dict has keys con_n_boxes, con_loss.
    """
    log: Dict[str, float] = {}

    roi_boxes, labels = build_roi_boxes(targets)
    N = roi_boxes.shape[0]
    log["con_n_boxes"] = float(N)

    if N == 0 or weight == 0.0:
        log["con_loss"] = 0.0
        return student_images.sum() * 0.0, log

    orig_h, orig_w = student_images.shape[2], student_images.shape[3]

    # Student FPN features — gradient flows through here
    s_feats, s_trans_w, s_trans_h = student.get_fpn_features(
        student_images, level_keys=levels
    )

    # Teacher FPN features — no gradient
    with torch.no_grad():
        t_feats, t_trans_w, t_trans_h = teacher.get_fpn_features(
            teacher_images, level_keys=levels
        )

    s_obj = extract_object_features(
        s_feats, roi_boxes, orig_h, orig_w, s_trans_h, s_trans_w, levels=levels
    )  # [N, C]

    with torch.no_grad():
        t_obj = extract_object_features(
            t_feats, roi_boxes, orig_h, orig_w, t_trans_h, t_trans_w, levels=levels
        )  # [N, C]

    s_obj = F.normalize(s_obj, dim=1)
    t_obj = F.normalize(t_obj, dim=1)

    con_loss = supcon(s_obj, t_obj, labels)
    log["con_loss"] = (weight * con_loss).item()
    return weight * con_loss, log

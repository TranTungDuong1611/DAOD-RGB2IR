"""
Sample visualization for detection evaluation.

Draws GT boxes (green) and predicted boxes (red) on IR images,
saves a grid PNG per evaluation trigger.

Dependencies: matplotlib (standard in ML environments)
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

# Class colors for per-class box coloring (predictions)
_PRED_COLORS = ["#FF4444", "#FF8800", "#CC00FF", "#0088FF", "#00CC88"]
_GT_COLOR    = "#00FF44"   # bright green for GT


# ---------------------------------------------------------------------------
# Core draw function
# ---------------------------------------------------------------------------

def draw_boxes_on_ax(
    ax,
    image: torch.Tensor,          # [3, H, W]  float [0,1]
    gt_boxes: Optional[torch.Tensor],      # [M, 4] xyxy  (can be None)
    gt_labels: Optional[torch.Tensor],     # [M]          (can be None)
    pred_boxes: Optional[torch.Tensor],    # [N, 4] xyxy  (can be None)
    pred_labels: Optional[torch.Tensor],   # [N]          (can be None)
    pred_scores: Optional[torch.Tensor],   # [N]          (can be None)
    class_names: Optional[List[str]] = None,
    score_thresh: float = 0.3,
    title: str = "",
) -> None:
    """Draw one image with GT and prediction boxes onto a matplotlib Axes."""
    img_np = image.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    ax.imshow(img_np, cmap="gray" if img_np.mean(axis=2).std() < 0.02 else None)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7, pad=2)

    # GT boxes — solid green
    if gt_boxes is not None and len(gt_boxes) > 0:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=_GT_COLOR, facecolor="none",
            )
            ax.add_patch(rect)
            label_str = ""
            if gt_labels is not None and class_names is not None:
                c = gt_labels[i].item()
                label_str = class_names[c] if c < len(class_names) else str(c)
            if label_str:
                ax.text(x1, y1 - 2, label_str, color=_GT_COLOR,
                        fontsize=5, va="bottom", clip_on=True)

    # Pred boxes — per-class color, dashed
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            score = pred_scores[i].item() if pred_scores is not None else 1.0
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            c = pred_labels[i].item() if pred_labels is not None else 0
            color = _PRED_COLORS[c % len(_PRED_COLORS)]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor=color, facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            label_str = ""
            if class_names is not None:
                label_str = class_names[c] if c < len(class_names) else str(c)
            score_str = f"{score:.2f}"
            text = f"{label_str} {score_str}" if label_str else score_str
            ax.text(x2, y1 - 2, text, color=color,
                    fontsize=5, va="bottom", ha="right", clip_on=True)


# ---------------------------------------------------------------------------
# Grid visualization
# ---------------------------------------------------------------------------

def visualize_eval_samples(
    model: "torch.nn.Module",
    val_loader: "torch.utils.data.DataLoader",
    device: torch.device,
    save_path: str,
    num_samples: int = 8,
    cols: int = 4,
    score_thresh: float = 0.3,
    class_names: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """
    Run model on the first `num_samples` images from val_loader,
    draw GT + predictions, save as a grid PNG.

    Args:
        model        : student model (will be set to eval then back to train)
        val_loader   : yields (images [B,3,H,W], targets List[Dict])
        device       : inference device
        save_path    : output PNG path (parent dirs created automatically)
        num_samples  : how many images to visualize
        cols         : grid columns
        score_thresh : minimum score to draw a predicted box
        class_names  : list of class name strings (e.g. ["person","car","bicycle"])
        title        : suptitle for the figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    samples_images: List[torch.Tensor] = []
    samples_targets: List[Dict]        = []
    samples_preds: List[Dict]          = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch
            images = images.to(device)
            preds  = model(images)

            for i in range(len(images)):
                if len(samples_images) >= num_samples:
                    break
                samples_images.append(images[i].cpu())
                samples_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                        for k, v in targets[i].items()})
                samples_preds.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                      for k, v in preds[i].items()})

            if len(samples_images) >= num_samples:
                break

    model.train()

    n    = len(samples_images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0))
    axes = axes.flatten() if n > 1 else [axes]

    for idx in range(n):
        tgt  = samples_targets[idx]
        pred = samples_preds[idx]
        draw_boxes_on_ax(
            ax=axes[idx],
            image=samples_images[idx],
            gt_boxes=tgt.get("boxes"),
            gt_labels=tgt.get("labels"),
            pred_boxes=pred.get("boxes"),
            pred_labels=pred.get("labels"),
            pred_scores=pred.get("scores"),
            class_names=class_names,
            score_thresh=score_thresh,
            title=f"sample {idx}",
        )

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    # Legend
    legend_handles = [
        patches.Patch(edgecolor=_GT_COLOR,         facecolor="none", label="GT"),
        patches.Patch(edgecolor=_PRED_COLORS[0],   facecolor="none", linestyle="--",
                      label=f"pred (thresh={score_thresh})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))

    sup = title or "Evaluation samples"
    fig.suptitle(sup, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info(f"[Visualize] saved {n} samples → {save_path}")

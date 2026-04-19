"""
Typed batch containers for each domain.

Domain hierarchy:   RGB  →  MID (SAGA)  →  IR
Data availability:  labeled   semi-labeled   unlabeled
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class RGBBatch:
    """
    Source domain batch — fully labeled.

    images  : [B, 3, H, W]  float32, values in model's expected range
    targets : list of B dicts, each with
                "boxes"  : [N, 4]  float32  xyxy
                "labels" : [N]     int64
    """
    images: torch.Tensor
    targets: List[Dict[str, torch.Tensor]]


@dataclass
class MidBatch:
    """
    Intermediate domain batch — created by applying SAGA to an RGB batch.

    images         : [B, 3, H, W]  SAGA-transformed (objects→gray, BG→RGB)
    targets        : list of B dicts — original RGB GT (for optional mid_gt_loss)
    source_images  : [B, 3, H, W]  original RGB images before SAGA (optional)
    """
    images: torch.Tensor
    targets: List[Dict[str, torch.Tensor]]
    source_images: Optional[torch.Tensor] = None


@dataclass
class IRBatch:
    """
    Target domain batch — unlabeled.

    images : [B, C, H, W]  float32  (C=1 for thermal or C=3 for pseudo-RGB)
    """
    images: torch.Tensor

"""
Typed batch containers for each domain step.

Domain hierarchy:   RGB  →  MID (GAN)  →  IR
Data availability:  labeled   labeled       unlabeled
"""

from dataclasses import dataclass
from typing import Dict, List

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
class IRBatch:
    """
    Target domain batch — unlabeled.

    images : [B, C, H, W]  float32  (C=1 for thermal or C=3 for pseudo-RGB)
    """
    images: torch.Tensor

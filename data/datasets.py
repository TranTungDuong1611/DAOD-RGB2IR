"""
FLIR ADAS Aligned Dataset Loader — VOC XML format.

Actual dataset structure:
  align/
  ├── JPEGImages/
  │   ├── FLIR_XXXXX_PreviewData.jpeg   ← IR (thermal) image
  │   ├── FLIR_XXXXX_RGB.jpg            ← RGB image (spatially aligned with IR)
  │   └── ...
  ├── Annotations/
  │   ├── FLIR_XXXXX_PreviewData.xml    ← VOC XML annotation for IR image
  │   └── ...
  └── ImageSets/Main/
      ├── align_train.txt               ← stems: "FLIR_XXXXX_PreviewData"
      └── align_validation.txt

Key conventions:
  - stem  = "FLIR_XXXXX_PreviewData"
  - IR  image  : JPEGImages/{stem}.jpeg
  - RGB image  : JPEGImages/{stem.replace('_PreviewData','')}_RGB.jpg
  - Annotation : Annotations/{stem}.xml

Classes (FCOS 0-indexed, background excluded):
  person  → 0
  car     → 1
  bicycle → 2

Ignored labels: "FLIR" (source tag in XML), "dog" (too rare)

Domain adaptation roles:
  FLIRRGBDataset   → source domain  (labeled RGB, annotations from aligned IR XML)
  FLIRIRDataset    → target domain  (unlabeled IR, training only)
  FLIRIRValDataset → evaluation     (labeled IR, mAP computation)
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.helper import ir_stem_to_rgb_filename, read_split_file
from data.augmentations import default_rgb_transform, default_ir_transform
from data.preprocessing import parse_voc_xml, objects_to_tensors

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FLIRRGBDataset(Dataset):
    """
    Labeled RGB source domain.

    Loads: JPEGImages/FLIR_XXXXX_RGB.jpg
    Labels: parsed from Annotations/FLIR_XXXXX_PreviewData.xml
    (annotations are aligned → valid for paired RGB images)

    Args:
        root       : path to the `align/` directory
        split      : "train" or "validation"
        transform  : image transform (default: ToTensor)
        min_area   : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or default_rgb_transform()
        self.min_area  = min_area

        split_file = self.root / "ImageSets" / "Main" / f"align_{split}.txt"
        self.stems = read_split_file(split_file)

        # Filter to stems where the RGB image actually exists
        self.stems = [
            s for s in self.stems
            if (self.root / "JPEGImages" / ir_stem_to_rgb_filename(s)).exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, str]:
        stem     = self.stems[idx]
        rgb_path = self.root / "JPEGImages" / ir_stem_to_rgb_filename(stem)
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(rgb_path).convert("RGB")
        img_t  = self.transform(img)

        objects       = parse_voc_xml(ann_path)
        boxes, labels = objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target, stem


class FLIRIRDataset(Dataset):
    """
    Unlabeled IR target domain — for training only.

    Loads: JPEGImages/FLIR_XXXXX_PreviewData.jpeg  (no labels)

    Args:
        root      : path to the `align/` directory
        split     : "train" or "validation"
        transform : image transform (default: Grayscale→3ch, ToTensor)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or default_ir_transform()

        split_file = self.root / "ImageSets" / "Main" / f"align_{split}.txt"
        all_stems  = read_split_file(split_file)

        self.ir_paths = [
            self.root / "JPEGImages" / f"{s}.jpeg"
            for s in all_stems
            if (self.root / "JPEGImages" / f"{s}.jpeg").exists()
        ]

    def __len__(self) -> int:
        return len(self.ir_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img   = Image.open(self.ir_paths[idx])
        img_t = self.transform(img)
        return img_t, self.ir_paths[idx].stem


class FLIRIRValDataset(Dataset):
    """
    Labeled IR target domain — for evaluation only.

    Loads: JPEGImages/FLIR_XXXXX_PreviewData.jpeg
    Labels: Annotations/FLIR_XXXXX_PreviewData.xml

    Args:
        root      : path to the `align/` directory
        split     : "train" or "validation"
        transform : image transform
        min_area  : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split: str = "validation",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or default_ir_transform()
        self.min_area  = min_area

        split_file = self.root / "ImageSets" / "Main" / f"align_{split}.txt"
        all_stems  = read_split_file(split_file)

        # Keep only stems where both image and annotation exist
        self.stems = [
            s for s in all_stems
            if (self.root / "JPEGImages"  / f"{s}.jpeg").exists()
            and (self.root / "Annotations" / f"{s}.xml").exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, str]:
        stem     = self.stems[idx]
        ir_path  = self.root / "JPEGImages"  / f"{stem}.jpeg"
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(ir_path)
        img_t  = self.transform(img)

        objects       = parse_voc_xml(ann_path)
        boxes, labels = objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target, stem
    




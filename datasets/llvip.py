"""
LLVIP Dataset Loader — VOC XML format.

Actual dataset structure:
  LLVIP/
  ├── visible/
  │   ├── train/{stem}.jpg     ← RGB (visible) image
  │   └── test/{stem}.jpg
  ├── infrared/
  │   ├── train/{stem}.jpg     ← IR (thermal) image, SAME stem as its RGB pair
  │   └── test/{stem}.jpg
  └── Annotations/{stem}.xml   ← VOC XML annotation, shared by the RGB/IR pair

Key conventions:
  - stem         = "010079"  (no extension, identical across visible/ and infrared/)
  - RGB image    : visible/{split}/{stem}.jpg
  - IR  image    : infrared/{split}/{stem}.jpg
  - Annotation   : Annotations/{stem}.xml  (same box for both, images are pixel-aligned)
  - split        : "train" or "test" (LLVIP's own folder split — no separate
                   ImageSets/*.txt file like FLIR)

Classes (FCOS 0-indexed, background excluded):
  person → 0   (LLVIP only annotates pedestrians)

Domain adaptation roles:
  LLVIPRGBDataset   → source domain  (labeled RGB)
  LLVIPIRDataset    → target domain  (unlabeled IR, training only)
  LLVIPIRValDataset → evaluation     (labeled IR, mAP computation)
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

LLVIP_CLASSES    = ["person"]
LLVIP_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(LLVIP_CLASSES)}
NUM_CLASSES      = len(LLVIP_CLASSES)   # 1

# COCO 91-class output indices for each LLVIP class (used to slice COCO cls_logits).
# COCO label space: 0=__background__, 1=person, ...
LLVIP_TO_COCO_IDX: List[int] = [1]


# ---------------------------------------------------------------------------
# VOC XML parser
# ---------------------------------------------------------------------------

def _parse_voc_xml(xml_path: Path) -> List[Dict]:
    """
    Parse a VOC-format XML file and return a list of object dicts.

    Each dict:  {"label": str,  "box": [xmin, ymin, xmax, ymax]}
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name not in LLVIP_CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            objects.append({"label": name, "box": [xmin, ymin, xmax, ymax]})
    return objects


def _objects_to_tensors(
    objects: List[Dict],
    min_area: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert parsed objects to (boxes [N,4], labels [N]) tensors."""
    boxes, labels = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(LLVIP_CLASS_TO_IDX[obj["label"]])

    if boxes:
        return (
            torch.as_tensor(boxes,  dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.int64),
        )
    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,),   dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _list_stems(folder: Path) -> List[str]:
    """List stems (filename without extension) of images directly under `folder`."""
    return sorted(p.stem for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _split_disjoint(
    stems: List[str],
    source_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split a stem list into two DISJOINT halves: (source, target).

    Per the D3T paper (sec. 4.1), the RGB source and thermal target must not
    reuse the same aligned images, otherwise the model can "cheat" via the
    pixel-aligned correspondence between RGB and IR. LLVIP's train split is
    aligned (each stem has a matching RGB/IR pair), so rgb_train and ir_train
    should be built from disjoint stem lists.

      seed >= 0 : deterministic shuffle, then cut at `source_frac`.
      seed <  0 : keep original order, take "first N" (paper's exact wording).
    """
    s = list(stems)
    if seed >= 0:
        random.Random(seed).shuffle(s)
    k = int(round(len(s) * source_frac))
    return s[:k], s[k:]


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def _default_rgb_transform() -> T.Compose:
    """ToTensor only — FCOS GeneralizedRCNNTransform handles ImageNet normalisation."""
    return T.Compose([T.ToTensor()])


def _default_ir_transform() -> T.Compose:
    """IR (thermal JPEG): convert to 3-channel float [0,1]. FCOS expects 3-ch input."""
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LLVIPRGBDataset(Dataset):
    """
    Labeled RGB (visible) source domain.

    Loads: visible/{split}/{stem}.jpg
    Labels: parsed from Annotations/{stem}.xml (shared with the aligned IR image)

    Args:
        root       : path to the `LLVIP/` directory
        split      : "train" or "test"
        transform  : image transform (default: ToTensor)
        min_area   : skip boxes smaller than this (pixels²)
        stems      : explicit stem list to use instead of listing
                     visible/{split}/ (e.g. one half of a disjoint split).
                     None (default) preserves the old behaviour.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
        stems: Optional[List[str]] = None,
    ) -> None:
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or _default_rgb_transform()
        self.min_area  = min_area

        if stems is not None:
            self.stems = list(stems)
        else:
            self.stems = _list_stems(self.root / "visible" / split)

        # Filter to stems where the RGB image actually exists
        self.stems = [
            s for s in self.stems
            if (self.root / "visible" / split / f"{s}.jpg").exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        rgb_path = self.root / "visible" / self.split / f"{stem}.jpg"
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(rgb_path).convert("RGB")
        img_t  = self.transform(img)

        objects       = _parse_voc_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target


class LLVIPIRDataset(Dataset):
    """
    Unlabeled IR (infrared) target domain — for training only.

    Loads: infrared/{split}/{stem}.jpg  (no labels)

    Args:
        root      : path to the `LLVIP/` directory
        split     : "train" or "test"
        transform : image transform (default: Grayscale→3ch, ToTensor)
        stems     : explicit stem list to use instead of listing
                    infrared/{split}/ (e.g. the disjoint target half).
                    None (default) preserves the old behaviour.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        stems: Optional[List[str]] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_ir_transform()

        ir_dir = self.root / "infrared" / split
        all_stems = list(stems) if stems is not None else _list_stems(ir_dir)

        self.ir_paths = [
            ir_dir / f"{s}.jpg"
            for s in all_stems
            if (ir_dir / f"{s}.jpg").exists()
        ]

    def __len__(self) -> int:
        return len(self.ir_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        img   = Image.open(self.ir_paths[idx])
        img_t = self.transform(img)
        return (img_t,)


class LLVIPIRValDataset(Dataset):
    """
    Labeled IR target domain — for evaluation only.

    Loads: infrared/{split}/{stem}.jpg
    Labels: Annotations/{stem}.xml

    Args:
        root      : path to the `LLVIP/` directory
        split     : "train" or "test" (use "test" for evaluation)
        transform : image transform
        min_area  : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or _default_ir_transform()
        self.min_area  = min_area

        ir_dir     = self.root / "infrared" / split
        all_stems  = _list_stems(ir_dir)

        # Keep only stems where both image and annotation exist
        self.stems = [
            s for s in all_stems
            if (ir_dir / f"{s}.jpg").exists()
            and (self.root / "Annotations" / f"{s}.xml").exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        ir_path  = self.root / "infrared" / self.split / f"{stem}.jpg"
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(ir_path)
        img_t  = self.transform(img)

        objects       = _parse_voc_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def rgb_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def ir_collate(batch: List) -> Tuple[torch.Tensor]:
    return (torch.stack([b[0] for b in batch]),)


def ir_val_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

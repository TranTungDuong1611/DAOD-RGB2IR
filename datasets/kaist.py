"""
KAIST Multispectral Pedestrian Dataset Loader — KAIST XML format (x,y,w,h).

Actual dataset structure (kaist-cvpr15/):
  kaist-cvpr15/
  ├── images/
  │   └── {set}/{video}/
  │       ├── visible/{stem}.jpg   ← RGB image, e.g. set00/V000/visible/I00019.jpg
  │       └── lwir/{stem}.jpg      ← thermal (LWIR) image, SAME stem, pixel-aligned
  ├── annotations-xml-new-sanitized/
  │   └── {set}/{video}/{stem}.xml ← corrected annotations (prefer over annotations-xml-new)
  └── imageSets/
      ├── train-all-{01,02,04,20}.txt   ← set00-set05, frame stride 1/2/4/20
      └── test-all-{01,20}.txt          ← set06-set11
          (also train/test-{day,night}-*.txt subsets)

Key conventions:
  - key (called "stem" below for consistency with datasets/flir.py and
    datasets/llvip.py) = the FULL relative triplet "set/video/frame", e.g.
    "set00/V000/I00019" — this is what each imageSets/*.txt line already is,
    and it's globally unique across the whole dataset (unlike FLIR/LLVIP
    where a bare frame id is enough).
  - RGB image  : images/{key}.jpg  with "visible" substituted for the last "/"
                 segment's parent, i.e. images/{set}/{video}/visible/{frame}.jpg
  - IR  image  : images/{set}/{video}/lwir/{frame}.jpg
  - Annotation : annotations-xml-new-sanitized/{key}.xml
  - Box format : <bndbox><x><y><w><h></bndbox>  (top-left + width/height,
                 NOT xmin/ymin/xmax/ymax like VOC) → converted to xyxy here.

Classes: KAIST annotates "person", "people" (ambiguous group of persons),
"cyclist", and "person?" (uncertain/heavily occluded). Only "person" is used
as a detection class here (single-class, matching the standard KAIST
"reasonable" pedestrian-detection protocol and datasets/llvip.py's
single-class setup) — the rest are ignored (dropped, not treated as a
negative/background region).

Official split convention (imageSets/train-all-XX.txt / test-all-XX.txt):
  train = set00..set05   (day+night sequences)
  test  = set06..set11
The "-01/02/04/20" suffix is the frame sub-sampling stride (20 = every 20th
frame). train-all-02 (~25k frames) and test-all-20 (~2.2k frames) are the
combination most commonly used in the literature; both are exposed here as
configurable split-file names rather than hard-coded.

Domain adaptation roles:
  KAISTRGBDataset   → source domain  (labeled RGB/visible)
  KAISTIRDataset    → target domain  (unlabeled LWIR, training only)
  KAISTIRValDataset → evaluation     (labeled LWIR, mAP computation)
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

KAIST_CLASSES    = ["person"]
KAIST_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(KAIST_CLASSES)}
NUM_CLASSES      = len(KAIST_CLASSES)   # 1

# COCO 91-class output indices for each KAIST class (used to slice COCO cls_logits).
# COCO label space: 0=__background__, 1=person, ...
KAIST_TO_COCO_IDX: List[int] = [1]

# Labels present in the XML that are NOT trained as a class: "people" (an
# ambiguous group of persons, no individual box), "cyclist" (different object
# category), "person?" (uncertain/heavily occluded, standard protocol ignores it).
_IGNORE_LABELS = {"people", "cyclist", "person?"}


# ---------------------------------------------------------------------------
# KAIST XML parser (x, y, w, h — NOT VOC xmin/xmax/ymin/ymax)
# ---------------------------------------------------------------------------

def _parse_kaist_xml(xml_path: Path) -> List[Dict]:
    """
    Parse a KAIST-format XML file and return a list of object dicts.

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
        if name in _IGNORE_LABELS or name not in KAIST_CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            x = float(bnd.findtext("x"))
            y = float(bnd.findtext("y"))
            w = float(bnd.findtext("w"))
            h = float(bnd.findtext("h"))
        except (TypeError, ValueError):
            continue
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = x1 + w, y1 + h
        if x2 > x1 and y2 > y1:
            objects.append({"label": name, "box": [x1, y1, x2, y2]})
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
        labels.append(KAIST_CLASS_TO_IDX[obj["label"]])

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

def _rgb_path(root: Path, key: str) -> Path:
    """'set00/V000/I00019' -> images/set00/V000/visible/I00019.jpg"""
    set_, video, frame = key.split("/")
    return root / "images" / set_ / video / "visible" / f"{frame}.jpg"


def _ir_path(root: Path, key: str) -> Path:
    """'set00/V000/I00019' -> images/set00/V000/lwir/I00019.jpg"""
    set_, video, frame = key.split("/")
    return root / "images" / set_ / video / "lwir" / f"{frame}.jpg"


def _ann_path(root: Path, key: str) -> Path:
    return root / "annotations-xml-new-sanitized" / f"{key}.xml"


def _read_split_file(split_file: Path) -> List[str]:
    """Read imageSets/*.txt — one 'set/video/frame' key per line (CRLF-safe)."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _split_disjoint(
    keys: List[str],
    source_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split a key list into two DISJOINT halves: (source, target).

    Per the D3T paper (sec. 4.1), the RGB source and thermal target must not
    reuse the same aligned frames, otherwise the model can "cheat" via the
    pixel-aligned correspondence between RGB and IR. KAIST's visible/lwir
    pairs are co-registered per frame, so rgb_train and ir_train should be
    built from disjoint key lists (same approach as datasets/flir.py and
    datasets/llvip.py).

      seed >= 0 : deterministic shuffle, then cut at `source_frac`.
      seed <  0 : keep original order, take "first N".
    """
    s = list(keys)
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
    """LWIR (thermal JPEG): convert to 3-channel float [0,1]. FCOS expects 3-ch input."""
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class KAISTRGBDataset(Dataset):
    """
    Labeled RGB (visible) source domain.

    Loads: images/{set}/{video}/visible/{frame}.jpg
    Labels: parsed from annotations-xml-new-sanitized/{set}/{video}/{frame}.xml
    (annotations are aligned → valid for the paired visible image)

    Args:
        root        : path to the `kaist-cvpr15/` directory
        split_file  : name (without .txt) of an imageSets/ file, e.g.
                      "train-all-02" (default) or "train-all-20"
        transform   : image transform (default: ToTensor)
        min_area    : skip boxes smaller than this (pixels²)
        keys        : explicit "set/video/frame" key list to use instead of
                      reading imageSets/{split_file}.txt (e.g. one half of a
                      disjoint split). None (default) preserves the old
                      behaviour.
    """

    def __init__(
        self,
        root: str,
        split_file: str = "train-all-02",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
        keys: Optional[List[str]] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_rgb_transform()
        self.min_area  = min_area

        if keys is not None:
            self.keys = list(keys)
        else:
            self.keys = _read_split_file(self.root / "imageSets" / f"{split_file}.txt")

        # Filter to keys where the RGB image actually exists
        self.keys = [k for k in self.keys if _rgb_path(self.root, k).exists()]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        key      = self.keys[idx]
        rgb_path = _rgb_path(self.root, key)
        ann_path = _ann_path(self.root, key)

        img    = Image.open(rgb_path).convert("RGB")
        img_t  = self.transform(img)

        objects       = _parse_kaist_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": key}
        return img_t, target


class KAISTIRDataset(Dataset):
    """
    Unlabeled LWIR (thermal) target domain — for training only.

    Loads: images/{set}/{video}/lwir/{frame}.jpg  (no labels)

    Args:
        root       : path to the `kaist-cvpr15/` directory
        split_file : name (without .txt) of an imageSets/ file
        transform  : image transform (default: Grayscale→3ch, ToTensor)
        keys       : explicit key list to use instead of reading
                     imageSets/{split_file}.txt (e.g. the disjoint target
                     half). None (default) preserves the old behaviour.
    """

    def __init__(
        self,
        root: str,
        split_file: str = "train-all-02",
        transform: Optional[Callable] = None,
        keys: Optional[List[str]] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_ir_transform()

        if keys is not None:
            all_keys = list(keys)
        else:
            all_keys = _read_split_file(self.root / "imageSets" / f"{split_file}.txt")

        self.ir_paths = [
            _ir_path(self.root, k) for k in all_keys
            if _ir_path(self.root, k).exists()
        ]

    def __len__(self) -> int:
        return len(self.ir_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        img   = Image.open(self.ir_paths[idx])
        img_t = self.transform(img)
        return (img_t,)


class KAISTIRValDataset(Dataset):
    """
    Labeled LWIR target domain — for evaluation only.

    Loads: images/{set}/{video}/lwir/{frame}.jpg
    Labels: annotations-xml-new-sanitized/{set}/{video}/{frame}.xml

    Args:
        root       : path to the `kaist-cvpr15/` directory
        split_file : name (without .txt) of an imageSets/ file, e.g.
                     "test-all-20" (default, standard KAIST test protocol)
        transform  : image transform
        min_area   : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split_file: str = "test-all-20",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_ir_transform()
        self.min_area  = min_area

        all_keys = _read_split_file(self.root / "imageSets" / f"{split_file}.txt")

        # Keep only keys where both the IR image and its annotation exist
        self.keys = [
            k for k in all_keys
            if _ir_path(self.root, k).exists() and _ann_path(self.root, k).exists()
        ]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        key      = self.keys[idx]
        ir_path  = _ir_path(self.root, key)
        ann_path = _ann_path(self.root, key)

        img    = Image.open(ir_path)
        img_t  = self.transform(img)

        objects       = _parse_kaist_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": key}
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

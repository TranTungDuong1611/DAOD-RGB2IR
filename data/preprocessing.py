import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import torch

FLIR_CLASSES    = ["person", "car", "bicycle"]
FLIR_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FLIR_CLASSES)}
NUM_CLASSES     = len(FLIR_CLASSES)   # 3

# Labels to silently skip (noise / artefacts in the XML)
_IGNORE_LABELS  = {"FLIR", "dog"}

def parse_voc_xml(xml_path: Path) -> List[Dict]:
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
        if name in _IGNORE_LABELS or name not in FLIR_CLASS_TO_IDX:
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


def objects_to_tensors(
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
        labels.append(FLIR_CLASS_TO_IDX[obj["label"]])

    if boxes:
        return (
            torch.as_tensor(boxes,  dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.int64),
        )
    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,),   dtype=torch.int64),
    )
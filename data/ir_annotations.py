"""Read audit-only IR labels without changing training datasets or sampling."""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from data.preprocessing import FLIR_CLASS_TO_IDX, objects_to_tensors


@dataclass(frozen=True)
class IRAnnotation:
    """Missing/invalid labels are unknown; an available empty target is known."""

    status: str
    target: dict | None = None


class IRAnnotationReader:
    def __init__(self, annotation_dir):
        self.annotation_dir = Path(annotation_dir)

    def read(self, sample_ids, image_sizes, *, hflipped: bool):
        """Read by training sample identity and apply its already sampled flip."""
        if len(sample_ids) != len(image_sizes):
            raise ValueError("IR annotation IDs and image sizes must align")
        return tuple(self._read_one(stem, size, hflipped) for stem, size in
                     zip(sample_ids, image_sizes))

    def _read_one(self, stem, image_size, hflipped):
        # Reject unavailable/corrupt annotations instead of counting them as
        # background. Do not filter images out of the existing training loader.
        if not isinstance(stem, str) or Path(stem).name != stem or stem in {"", ".", ".."}:
            raise ValueError("IR annotation ID must be a file stem")
        path = self.annotation_dir / f"{stem}.xml"
        try:
            root = ET.parse(path).getroot()
        except FileNotFoundError:
            return IRAnnotation("missing")
        except (ET.ParseError, OSError):
            return IRAnnotation("invalid")
        height, width = image_size
        try:
            if root.tag != "annotation":
                raise ValueError("expected VOC annotation")
            size = root.find("size")
            if size is not None and (
                int(size.findtext("height")) != height
                or int(size.findtext("width")) != width
            ):
                raise ValueError("annotation and IR image sizes differ")
            objects = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "").strip()
                if name not in FLIR_CLASS_TO_IDX:
                    continue
                box = obj.find("bndbox")
                coords = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
                x1, y1, x2, y2 = coords
                if not all(math.isfinite(v) for v in coords) or x2 <= x1 or y2 <= y1:
                    raise ValueError("invalid annotation box")
                objects.append({"label": name, "box": coords})
        except (TypeError, ValueError, AttributeError):
            return IRAnnotation("invalid")

        # Match IR evaluation's class mapping and retain small boxes. Geometry
        # follows the existing batch flip; no augmentation or RNG is run here.
        boxes, labels = objects_to_tensors(objects, min_area=0.)
        if hflipped:
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return IRAnnotation("available", {"boxes": boxes, "labels": labels, "stem": stem})

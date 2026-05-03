from typing import List
from pathlib import Path


def ir_stem_to_rgb_filename(stem: str) -> str:
    """FLIR_XXXXX_PreviewData  →  FLIR_XXXXX_RGB.jpg"""
    base = stem.replace("_PreviewData", "")
    return f"{base}_RGB.jpg"


def read_split_file(split_file: Path) -> List[str]:
    """Read ImageSets/Main/*.txt — one stem per line, ignoring blank lines."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]
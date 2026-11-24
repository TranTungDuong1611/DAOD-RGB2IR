from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from utils.helper import read_split_file

from .datasets import (
    FLIRRGBDataset,
    FLIRIRDataset,
    FLIRIRValDataset,
    _resolve_split_file,
)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def detection_collate(batch: List) -> Tuple[torch.Tensor, List[Dict], Tuple[str, ...]]:
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        tuple(b[2] for b in batch),
    )


def ir_collate(batch: List) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    return torch.stack([b[0] for b in batch]), tuple(b[1] for b in batch)


def _disjoint_train_stems(root):
    """Partition the manifest before modality-specific missing-file filtering."""

    stems = read_split_file(_resolve_split_file(Path(root), "train"))
    if len(stems) < 2:
        raise ValueError("disjoint training requires at least two image IDs in align_train.txt")
    if len(set(stems)) != len(stems):
        raise ValueError("align_train.txt contains duplicate image IDs; disjoint training requires unique IDs")
    midpoint = len(stems) // 2
    return stems[:midpoint], stems[midpoint:]


def build_loader(dataset, settings, *, collate_fn, device, shuffle, drop_last, generator=None):
    """Create train, validation and audit loaders with the same runtime settings."""
    return DataLoader(
        dataset, batch_size=settings.batch_size, num_workers=settings.num_workers,
        shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fn,
        pin_memory=torch.device(device).type == "cuda", generator=generator,
    )


def build_rgb_train_dataloader(config, *, stems=None):
    if stems is None:
        stems, _ = _disjoint_train_stems(config.data.root)
    dataset = FLIRRGBDataset(config.data.root, split="train", stems=stems, min_area=config.data.rgb_min_area)
    settings = config.loader.train
    return build_loader(dataset, settings, collate_fn=detection_collate, device=config.device,
                        shuffle=settings.shuffle, drop_last=settings.drop_last)


def build_ir_train_dataloader(config, *, stems=None):
    if stems is None:
        _, stems = _disjoint_train_stems(config.data.root)
    dataset = FLIRIRDataset(config.data.root, split="train", stems=stems)
    settings = config.loader.train
    return build_loader(dataset, settings, collate_fn=ir_collate, device=config.device,
                        shuffle=settings.shuffle, drop_last=settings.drop_last)


def build_rgb_val_dataloader(config):
    dataset = FLIRRGBDataset(config.data.root, split="validation", min_area=config.data.rgb_min_area)
    settings = config.loader.eval
    return build_loader(dataset, settings, collate_fn=detection_collate, device=config.device,
                        shuffle=settings.shuffle, drop_last=False,
                        generator=torch.Generator().manual_seed(settings.seed))


def build_ir_val_dataloader(config):
    dataset = FLIRIRValDataset(config.data.root, split="validation", min_area=config.data.ir_val_min_area)
    settings = config.loader.eval
    return build_loader(dataset, settings, collate_fn=detection_collate, device=config.device,
                        shuffle=settings.shuffle, drop_last=False,
                        generator=torch.Generator().manual_seed(settings.seed))


def build_dataloaders(config) -> Dict[str, DataLoader]:
    source_stems, target_stems = _disjoint_train_stems(config.data.root)
    return {
        "rgb_train": build_rgb_train_dataloader(config, stems=source_stems),
        "ir_train": build_ir_train_dataloader(config, stems=target_stems),
        "rgb_val": build_rgb_val_dataloader(config),
        "ir_val": build_ir_val_dataloader(config),
    }

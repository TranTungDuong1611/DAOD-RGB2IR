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

def rgb_collate(batch: List) -> Tuple[torch.Tensor, List[Dict], Tuple[str, ...]]:
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        tuple(b[2] for b in batch),
    )


def ir_collate(batch: List) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    return torch.stack([b[0] for b in batch]), tuple(b[1] for b in batch)


def ir_val_collate(batch: List) -> Tuple[torch.Tensor, List[Dict], Tuple[str, ...]]:
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],
        tuple(b[2] for b in batch),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _disjoint_train_stems(root):
    """Partition the manifest before modality-specific missing-file filtering."""

    stems = read_split_file(_resolve_split_file(Path(root), "train"))
    if len(stems) < 2:
        raise ValueError("disjoint training requires at least two image IDs in align_train.txt")
    if len(set(stems)) != len(stems):
        raise ValueError("align_train.txt contains duplicate image IDs; disjoint training requires unique IDs")
    midpoint = len(stems) // 2
    return stems[:midpoint], stems[midpoint:]


def build_rgb_train_dataloader(config):
    source_stems, _ = _disjoint_train_stems(config.data.root)
    dataset = FLIRRGBDataset(
        root=config.data.root,
        split="train",
        stems=source_stems,
    )

    return DataLoader(
        dataset,
        batch_size=config.loader.train.batch_size,
        shuffle=config.loader.train.shuffle,
        num_workers=config.loader.train.num_workers,
        collate_fn=rgb_collate,
        pin_memory=True,
        drop_last=config.loader.train.drop_last,
    )


def build_ir_train_dataloader(config):
    _, target_stems = _disjoint_train_stems(config.data.root)
    dataset = FLIRIRDataset(
        root=config.data.root,
        split="train",
        stems=target_stems,
    )

    return DataLoader(
        dataset,
        batch_size=config.loader.train.batch_size,
        shuffle=config.loader.train.shuffle,
        num_workers=config.loader.train.num_workers,
        collate_fn=ir_collate,
        pin_memory=True,
        drop_last=config.loader.train.drop_last,
    )


def build_rgb_val_dataloader(config):
    dataset = FLIRRGBDataset(
        root=config.data.root,
        split="validation",
    )

    return DataLoader(
        dataset,
        batch_size=config.loader.eval.batch_size,
        shuffle=config.loader.eval.shuffle,
        num_workers=config.loader.eval.num_workers,
        collate_fn=rgb_collate,
        pin_memory=True,
    )


def build_ir_val_dataloader(config):
    dataset = FLIRIRValDataset(
        root=config.data.root,
        split="validation",
    )

    return DataLoader(
        dataset,
        batch_size=config.loader.eval.batch_size,
        shuffle=config.loader.eval.shuffle,
        num_workers=config.loader.eval.num_workers,
        collate_fn=ir_val_collate,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_dataloaders(config) -> Dict[str, DataLoader]:
    return {
        "rgb_train": build_rgb_train_dataloader(config),
        "rgb_val":   build_rgb_val_dataloader(config),
        "ir_train":  build_ir_train_dataloader(config),
        "ir_val":    build_ir_val_dataloader(config),
    }

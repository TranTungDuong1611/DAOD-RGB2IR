from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader

from .datasets import (
    FLIRRGBDataset,
    FLIRIRDataset,
    FLIRIRValDataset,
)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def rgb_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def ir_collate(batch: List) -> torch.Tensor:
    return torch.stack([b[0] for b in batch])


def ir_val_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_rgb_train_dataloader(config):
    dataset = FLIRRGBDataset(
        root=config.data.root,
        split="train",
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
    dataset = FLIRIRDataset(
        root=config.data.root,
        split="train",
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
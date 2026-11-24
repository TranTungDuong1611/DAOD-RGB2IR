"""Dataset, loader and appearance augmentation settings."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    root: str = "/home/intern/D3T/data/align"
    rgb_min_area: float = 16.0
    ir_val_min_area: float = 0.0


@dataclass
class TrainLoaderConfig:
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class EvalLoaderConfig:
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = False
    seed: int = 0


@dataclass
class DataLoaderConfig:
    train: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    eval: EvalLoaderConfig   = field(default_factory=EvalLoaderConfig)


@dataclass
class AugConfig:
    """Augmentation config for student (applied on top of DataLoader transforms)."""
    # Geometric
    hflip_prob:              float = 0.5   # horizontal flip
    # Photometric
    blur_prob:               float = 0.5
    blur_sigma_max:          float = 1.0
    brightness_prob:         float = 0.3
    brightness_mag:          float = 0.2   # ±20% brightness
    contrast_prob:           float = 0.3
    contrast_mag:            float = 0.2   # ±20% contrast


@dataclass
class SoftSAGAConfig:
    """SoftSAGA alpha per MID level (1.0=pure RGB, 0.0=pure gray)."""
    alpha_near_rgb:     float = 0.75   # objects mostly RGB
    alpha_intermediate: float = 0.50   # half RGB, half gray
    alpha_near_ir:      float = 0.25   # objects mostly gray


__all__ = ['DataConfig', 'TrainLoaderConfig', 'EvalLoaderConfig', 'DataLoaderConfig', 'AugConfig', 'SoftSAGAConfig']

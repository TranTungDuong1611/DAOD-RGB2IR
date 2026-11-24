"""Kaggle-only orchestration around the canonical CLI and evaluator."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch


def make_training_args(settings):
    from example_flir import parse_args
    argv = ["--data_root", str(settings["data_root"]), "--output_dir", str(settings["output_dir"])]
    options = {
        "total_iters": "--total_iters", "batch_size": "--batch_size",
        "eval_batch_size": "--eval_batch_size", "workers": "--workers",
        "weights": "--weights", "device": "--device", "teacher_mode": "--teacher-mode",
        "workflow": "--workflow", "eval_every": "--eval_every", "log_interval": "--log_interval",
        "intermediate_domain": "--intermediate-domain", "texture_strength": "--texture-strength",
        "btf_radius": "--btf-radius", "btf_iterations": "--btf-iterations",
        "domain_sampling": "--domain-sampling",
        "color_consistency_weight": "--color-consistency-weight",
        "color_consistency_levels": "--color-consistency-levels",
        "color_consistency_level_weights": "--color-consistency-level-weights",
        "gan_weights": "--gan-weights",
        "bridge_ratios": "--bridge-ratios",
        "bridge_phase_fractions": "--bridge-phase-fractions",
        "bridge_source_sup_weight": "--bridge-source-sup-weight",
        "bridge_saga_student_alphas": "--bridge-saga-student-alphas",
        "bridge_saga_teacher_alphas": "--bridge-saga-teacher-alphas",
        "bridge_saga_teachers": "--bridge-saga-teachers",
        "resume": "--resume", "lr": "--lr", "min_sizes": "--min-sizes", "max_size": "--max_size",
        "phase1_end": "--phase1-end", "phase2_end": "--phase2-end", "phase3_end": "--phase3-end",
        "phase2_rgb_ratio": "--phase2-rgb-ratio", "phase3_rgb_ratio": "--phase3-rgb-ratio",
        "phase4_rgb_ratio": "--phase4-rgb-ratio",
    }
    for key, flag in options.items():
        value = settings.get(key)
        if value is not None:
            argv.append(flag)
            if key == "bridge_ratios" and not isinstance(value, str):
                argv.append(",".join(":".join(str(weight) for weight in row) for row in value))
            elif isinstance(value, (tuple, list)):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
    if settings.get("eval_pseudo_labels", False):
        argv.append("--eval-pseudo-labels")
    return parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset(data_root, batch_size):
    from data.dataloaders import _disjoint_train_stems
    from data.datasets import FLIRRGBDataset, FLIRIRDataset, FLIRIRValDataset
    root = Path(data_root)
    for directory in ("JPEGImages", "Annotations"):
        if not (root / directory).is_dir():
            raise FileNotFoundError(f"Missing {directory} at {root}; attach the FLIR dataset and check DATA_ROOT")
    source_stems, target_stems = _disjoint_train_stems(root)
    rgb = FLIRRGBDataset(str(root), split="train", stems=source_stems)
    ir = FLIRIRDataset(str(root), split="train", stems=target_stems)
    validation = FLIRIRValDataset(str(root), split="validation")
    if len(rgb) < batch_size or len(ir) < batch_size or not len(validation):
        raise ValueError(f"Dataset too small/empty for drop_last training: RGB={len(rgb)}, IR={len(ir)}, IR val={len(validation)}, batch={batch_size}")
    missing = [str(root / "Annotations" / f"{stem}.xml") for stem in rgb.stems if not (root / "Annotations" / f"{stem}.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RGB source annotation(s), first: {missing[:5]}")
    return {"rgb_train": len(rgb), "ir_train": len(ir), "ir_validation": len(validation)}


def preflight(args, manifest=None):
    """Check the actual dependencies/data/initial weights before model assembly."""
    import cv2
    import torchvision
    from torchvision.ops import nms
    from pycocotools.coco import COCO  # noqa: F401
    from dataclasses import asdict
    from example_flir import build_training_config
    from randommono_btf import require_btf
    config = build_training_config(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Kaggle select Settings > Accelerator > GPU, then restart and Run All.")
    box = torch.tensor([[0., 0., 8., 8.]], device=device)
    nms(box, torch.ones(1, device=device), .5)
    counts = validate_dataset(args.data_root, args.batch_size)
    if config.workflow == "gan_bridge" and not Path(config.gan_bridge.checkpoint).is_file():
        raise FileNotFoundError(f"Missing GAN bridge checkpoint: {config.gan_bridge.checkpoint}")
    if (
        config.workflow in {"curriculum", "gan_bridge"}
        and config.intermediate_domain.mode == "randommono_btf"
        and config.intermediate_domain.texture_strength
    ):
        function = require_btf()
        result = function(np.full((12, 12), .5, np.float32), fr=config.intermediate_domain.btf_radius, numIter=1)
        if result.shape != (12, 12) or not np.isfinite(result).all():
            raise RuntimeError("OpenCV BTF capability check failed")
    # Resolve weights up front. Keep model factory initialization unchanged.
    try:
        if config.model.weights:
            from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
            state = FCOS_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
            del state
        elif config.model.pretrained_backbone:
            from torchvision.models import ResNet50_Weights
            state = ResNet50_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
            del state
    except Exception as error:
        raise RuntimeError("Pretrained weights could not be loaded. Enable Kaggle Internet for the first download or populate the torchvision cache. No random-initialization fallback was used.") from error
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effective = asdict(config)
    (output / "effective_config.json").write_text(
        json.dumps(effective, indent=2, default=lambda value: value.value if hasattr(value, "value") else str(value)),
        encoding="utf-8",
    )
    if manifest is not None:
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__, "device": str(device), **counts})
    print(f"Training: {config.total_iters} iterations; outputs: {output}; bridge: {config.intermediate_domain}")
    return config


def save_bridge_preview(args, *, count=2):
    """Preview only source-train images; restore torch CPU RNG afterwards."""
    from data.dataloaders import _disjoint_train_stems
    from data.datasets import FLIRRGBDataset
    from example_flir import build_training_config
    from randommono_btf import RandomMonoBTF
    from saga import SoftSAGA
    config = build_training_config(args)
    source_stems, _ = _disjoint_train_stems(args.data_root)
    dataset = FLIRRGBDataset(args.data_root, split="train", stems=source_stems)
    items = [dataset[i] for i in range(min(count, len(dataset)))]
    alphas = (1., .75, .5, .25, 0.)
    state = torch.random.get_rng_state()
    start = time.perf_counter()
    try:
        torch.random.default_generator.manual_seed(123)
        if config.intermediate_domain.mode == "saga":
            saga = SoftSAGA()
            views = tuple(
                tuple(saga.apply(item[0], item[1]["boxes"], alpha) for item in items)
                for alpha in alphas
            )
        else:
            views = RandomMonoBTF(config.intermediate_domain).make_views(
                tuple(item[0] for item in items), alphas,
            )
    finally:
        torch.random.set_rng_state(state)
    elapsed = time.perf_counter() - start
    width, height = 256, 205
    sheet = Image.new("RGB", (width * len(alphas), (height + 26) * len(items)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(items):
        for col, alpha in enumerate(alphas):
            array = (views[col][row].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            tile = Image.fromarray(array).resize((width, height))
            sheet.paste(tile, (col * width, row * (height + 26) + 26))
            draw.text((col * width + 4, row * (height + 26) + 5), f"alpha={alpha:g} | {item[2]}", fill="black")
    output = Path(args.output_dir) / "bridge_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Bridge preprocessing: {elapsed / max(1, len(items)):.3f} s/image (CPU preview); {output}")
    return output


def select_checkpoint(output_dir, override=None):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"Requested evaluation checkpoint does not exist: {path}")
        return path
    for name in ("best_model.pth", "final_model.pth"):
        path = Path(output_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No best_model.pth or final_model.pth in {output_dir}; run training or set eval_checkpoint")


def release_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

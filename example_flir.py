"""
example_flir.py — Curriculum DA training on FLIR ADAS Aligned dataset.

Dataset structure (align/ directory):
  align/
  ├── JPEGImages/
  │   ├── FLIR_XXXXX_PreviewData.jpeg   ← IR (thermal) images
  │   ├── FLIR_XXXXX_RGB.jpg            ← RGB images (paired, spatially aligned)
  │   └── ...
  ├── Annotations/
  │   ├── FLIR_XXXXX_PreviewData.xml    ← VOC XML annotations (for IR)
  │   └── ...
  └── ImageSets/Main/
      ├── align_train.txt               ← 4129 training stems
      └── align_validation.txt          ← 1013 validation stems

Classes (FCOS 0-indexed):
  0=person  1=car  2=bicycle

Run (từ trong thư mục DAOD-RGB2IR/):
  python example_flir.py --data_root /path/to/align --device cuda

Run (từ thư mục cha DomainAdaptation/):
  python DAOD-RGB2IR/example_flir.py --data_root /path/to/align --device cuda
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Đảm bảo thư mục của script luôn nằm trong sys.path,
# cho dù chạy từ thư mục nào.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

from torch.utils.data import DataLoader

from config import (
    BlockSchedulerConfig,
    DisjointSplitConfig,
    DistillConfig,
    EMAConfig,
    GANConfig,
    IRAugConfig,
    LambdaConfig,
    RGBAugConfig,
    TeacherWeightConfig,
    TrainingConfig,
)
from gan_translator import GANTranslator
from datasets import (
    FLIR_CLASSES,
    FLIR_TO_COCO_IDX,
    NUM_CLASSES,
    FLIRIRDataset,
    FLIRIRValDataset,
    FLIRRGBDataset,
    ir_collate,
    ir_val_collate,
    rgb_collate,
)
from datasets.flir import _read_split_file, _split_disjoint
from ema import copy_student_to_teacher
from evaluator import DetectionEvaluator, PhaseEvaluator
from faster_rcnn_wrapper import build_faster_rcnn_trio
from fcos_wrapper import build_fcos_trio
from scheduler import BlockDomainScheduler, parse_ratio_schedule
from trainer import BlockDomainAdaptationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_training_config(device: str, target_h: int = 512, target_w: int = 640,
                         gan_checkpoint: str = "", args=None) -> TrainingConfig:
    a = args
    return TrainingConfig(
        ema=EMAConfig(alpha=a.ema_alpha if a else 0.9996, use_warmup=True),
        gan=GANConfig(
            checkpoint_path=gan_checkpoint,
            input_nc=3,
            output_nc=1,
            ngf=64,
            n_blocks=9,
        ),
        rgb_aug=RGBAugConfig(
            hflip_prob=0.5,
            multiscale_min=0.5,
            multiscale_max=1.5,
            multiscale_target_h=target_h,
            multiscale_target_w=target_w,
            blur_prob=0.5,
            blur_sigma_max=1.0,
            color_jitter_prob=0.5,
            cj_brightness=0.2,
            cj_contrast=0.2,
            cj_saturation=0.3,
            cj_hue=0.05,
            random_erasing_prob=0.3,
            random_erasing_scale_min=0.02,
            random_erasing_scale_max=0.10,
        ),
        ir_aug=IRAugConfig(
            hflip_prob=0.5,
            multiscale_min=0.5,
            multiscale_max=1.5,
            multiscale_target_h=target_h,
            multiscale_target_w=target_w,
            intensity_shift_prob=0.5,
            intensity_shift_mag=0.1,
            contrast_jitter_prob=0.5,
            contrast_jitter_mag=0.2,
            gamma_prob=0.3,
            gamma_min=0.7,
            gamma_max=1.3,
            gaussian_noise_prob=0.3,
            gaussian_noise_std=0.02,
        ),
        block_scheduler=BlockSchedulerConfig(
            ratio_schedule=a.ratio_schedule if a else BlockSchedulerConfig().ratio_schedule,
            cycle_iters=a.cycle_iters if a else BlockSchedulerConfig().cycle_iters,
        ),
        distill=DistillConfig(
            ratio=a.distill_ratio if a else DistillConfig().ratio,
            hm_alpha=a.hm_alpha if a else DistillConfig().hm_alpha,
            hm_beta=a.hm_beta if a else DistillConfig().hm_beta,
            un_alpha=a.un_alpha if a else DistillConfig().un_alpha,
            w_logits=a.dw_logits if a else DistillConfig().w_logits,
            w_deltas=a.dw_deltas if a else DistillConfig().w_deltas,
            w_quality=a.dw_quality if a else DistillConfig().w_quality,
            w_uhl=a.dw_uhl if a else DistillConfig().w_uhl,
        ),
        teacher_weight=TeacherWeightConfig(
            w_match=a.w_match if a else TeacherWeightConfig().w_match,
            w_cross=a.w_cross if a else TeacherWeightConfig().w_cross,
        ),
        disjoint_split=DisjointSplitConfig(
            enabled=a.disjoint_split if a else True,
            source_frac=a.source_frac if a else 0.5,
            split_seed=a.split_seed if a else -1,
        ),
        lambda_cfg=LambdaConfig(
            lambda_max=a.lambda_max if a else 1.0,
            ramp_iters=a.lambda_ramp_iters if a else 6_000,
        ),
        gt_weight=1.0,
        distill_num_classes=NUM_CLASSES,
        device=device,
        log_interval=100,
    )


# ---------------------------------------------------------------------------
# Teacher pretraining
# ---------------------------------------------------------------------------

def pretrain_teacher(
    model: torch.nn.Module,
    loader: DataLoader,
    total_iters: int,
    device: torch.device,
    lr_backbone: float,
    lr_head: float,
    label: str,
    save_path: str,
    ir_val_loader: Optional[DataLoader] = None,
    evaluator: Optional["DetectionEvaluator"] = None,
) -> None:
    """
    Supervised pretraining cho một teacher trên loader cho trước.

    Dùng cho:
      - rgb_teacher  : loader = rgb_loader  (labeled RGB images + GT boxes)
      - ir_teacher   : loader = mid_loader  (MID images do GAN sinh + GT boxes từ RGB)

    Nếu ir_val_loader và evaluator được cung cấp, tại mỗi log checkpoint sẽ
    eval mAP teacher trên IR val set và log kết quả.

    Args:
        model          : teacher model (FCOSDetector hoặc FasterRCNNDetector)
        loader         : DataLoader trả (images [B,3,H,W], targets List[Dict])
        total_iters    : số iteration pretrain
        device         : cuda/cpu
        lr_backbone, lr_head : learning rates
        label          : tên log (ví dụ "rgb_teacher", "ir_teacher")
        save_path      : đường dẫn lưu checkpoint sau pretrain
        ir_val_loader  : IR val DataLoader để eval mAP trong burn-in
        evaluator      : DetectionEvaluator instance (dùng chung, reset trước mỗi lần eval)
    """
    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD([
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad],
         "lr": lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    data_iter = iter(loader)
    log_every = max(1, total_iters // 20)
    do_ir_eval = ir_val_loader is not None and evaluator is not None

    logger.info(f"[Pretrain {label}] starting {total_iters} iters ...")
    for step in range(1, total_iters + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images  = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        lr_sched.step()

        if step % log_every == 0 or step == total_iters:
            logger.info(
                f"[Pretrain {label}] [{step:06d}/{total_iters}]  "
                f"loss={loss.item():.4f}  lr={optimizer.param_groups[-1]['lr']:.2e}"
            )
            if do_ir_eval:
                model.eval()
                evaluator.reset()
                with torch.no_grad():
                    for ir_images, ir_targets in ir_val_loader:
                        ir_images = ir_images.to(device)
                        preds = model(ir_images)
                        evaluator.update(preds, [{k: v for k, v in t.items()} for t in ir_targets])
                ir_results = evaluator.compute()
                map50   = ir_results.get("mAP@0.5", 0.0)
                per_cls = "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in sorted(ir_results.items()) if k.startswith("AP@0.5/")
                )
                logger.info(
                    f"[Pretrain {label}] [IR val @ {step:06d}]  "
                    f"mAP@0.5={map50:.4f}  {per_cls}"
                )
                model.train()

    # Freeze và eval sau pretrain
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save({"teacher": model.state_dict()}, save_path)
    logger.info(f"[Pretrain {label}] done — saved to {save_path}")


def load_or_pretrain_teacher(
    model: torch.nn.Module,
    weights_path: str,
    loader: DataLoader,
    total_iters: int,
    device: torch.device,
    lr_backbone: float,
    lr_head: float,
    label: str,
    save_path: str,
    ir_val_loader: Optional[DataLoader] = None,
    evaluator: Optional["DetectionEvaluator"] = None,
    skip_pretrain: bool = False,
) -> None:
    """
    Nếu `weights_path` được cung cấp: nạp checkpoint có sẵn (giống cách
    training_full_d3t_v2_*.py nhận --teacher-rgb-weights/--teacher-ir-weights),
    freeze và bỏ qua bước pretrain. Ngược lại, nếu `skip_pretrain=False`
    (mặc định): pretrain từ đầu như cũ. Nếu `skip_pretrain=True` và không có
    weights_path: KHÔNG pretrain, giữ nguyên trọng số khởi tạo hiện tại của
    model (giả định người dùng biết mình đang làm gì, ví dụ debug pipeline).

    Chấp nhận cả 2 định dạng checkpoint:
      - {"teacher": state_dict}  (định dạng pretrain_teacher() tự lưu — key đã có
                                    tiền tố "model." vì FCOSDetector/FasterRCNNDetector
                                    bọc torchvision model qua self.model)
      - state_dict thẳng của torchvision model KHÔNG bọc wrapper (định dạng
        training_full_d3t_v2_*.py: torch.save(fcos_resnet50_fpn(...).state_dict(), ...),
        key KHÔNG có tiền tố "model.") — tự thêm tiền tố "model." trước khi load.
    """
    if weights_path:
        logger.info(f"[{label}] Loading pretrained weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["teacher"] if isinstance(ckpt, dict) and "teacher" in ckpt else ckpt

        wrapper_keys = set(model.state_dict().keys())
        if not any(k in wrapper_keys for k in state_dict.keys()):
            # Checkpoint đến từ torchvision model trần (không có wrapper self.model.*)
            prefixed = {f"model.{k}": v for k, v in state_dict.items()}
            if any(k in wrapper_keys for k in prefixed.keys()):
                logger.info(f"[{label}] Checkpoint keys unprefixed — adding 'model.' prefix.")
                state_dict = prefixed

        model.load_state_dict(state_dict)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        logger.info(f"[{label}] Loaded (frozen) — skipping pretraining.")
        return

    if skip_pretrain:
        logger.warning(
            f"[{label}] --skip_teacher_pretrain set and no weights given — "
            f"model left at its current (random/COCO-backbone) initialization."
        )
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return

    pretrain_teacher(
        model, loader, total_iters, device, lr_backbone, lr_head, label,
        save_path, ir_val_loader=ir_val_loader, evaluator=evaluator,
    )


class MIDDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset: lấy ảnh RGB + GT từ FLIRRGBDataset,
    dùng GAN translator để sinh ảnh MID on-the-fly.

    Dùng để pretrain ir_teacher trên MID domain.
    """

    def __init__(self, rgb_dataset, gan_translator: "GANTranslator"):
        self.rgb_dataset    = rgb_dataset
        self.gan_translator = gan_translator

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        image, target = self.rgb_dataset[idx]         # [3, H, W], dict
        mid = self.gan_translator.apply_to_batch(
            image.unsqueeze(0)                        # [1, 3, H, W]
        ).squeeze(0)                                  # [3, H, W]
        return mid, target

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device_str = args.device
    device     = torch.device(device_str)
    data_root  = args.data_root

    logger.info("=== Curriculum DA — FLIR ADAS Aligned ===")
    logger.info(f"Data root : {data_root}")
    logger.info(f"Device    : {device_str}")
    logger.info(f"Classes   : {FLIR_CLASSES}  (num_classes={NUM_CLASSES})")

    # --- Datasets ---
    logger.info("Loading datasets ...")
    if args.disjoint_split:
        all_stems = _read_split_file(Path(data_root) / "align_train.txt")
        src_stems, tgt_stems = _split_disjoint(all_stems, args.source_frac, args.split_seed)
        rgb_train = FLIRRGBDataset(data_root, stems=src_stems)
        ir_train  = FLIRIRDataset( data_root, stems=tgt_stems)
        mode = "first-N" if args.split_seed < 0 else f"shuffled(seed={args.split_seed})"
        logger.info(
            f"DISJOINT split [{mode}]: RGB source={len(src_stems)} | "
            f"real-IR target={len(tgt_stems)} (no overlap — D3T paper sec. 4.1)"
        )
    else:
        rgb_train = FLIRRGBDataset(data_root, split="train")
        ir_train  = FLIRIRDataset( data_root, split="train")
        logger.warning(
            "--no_disjoint_split: RGB source and real-IR target share the same "
            "aligned images — NOT matching the D3T paper protocol."
        )
    ir_val    = FLIRIRValDataset(data_root, split="validation")
    rgb_val   = FLIRRGBDataset( data_root, split="validation")

    logger.info(
        f"  RGB train     : {len(rgb_train):>5} images  (labeled, source domain)\n"
        f"  IR  train     : {len(ir_train):>5} images  (unlabeled, target domain)\n"
        f"  IR  val       : {len(ir_val):>5} images  (labeled, for mAP)\n"
        f"  RGB val       : {len(rgb_val):>5} images  (labeled, reference eval)"
    )

    # --- DataLoaders ---
    rgb_loader = DataLoader(
        rgb_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=args.workers, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )
    ir_loader = DataLoader(
        ir_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=ir_collate, num_workers=args.workers, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )
    ir_val_loader = DataLoader(
        ir_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=ir_val_collate, num_workers=args.workers,
    )
    rgb_val_loader = DataLoader(
        rgb_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=rgb_collate, num_workers=args.workers,
    )

    # --- GAN Translator (cần trước khi build MIDDataset) ---
    gan_device = device
    if args.gan_checkpoint:
        logger.info(f"Loading GAN translator from: {args.gan_checkpoint}")
        gan_translator = GANTranslator.from_checkpoint(
            checkpoint_path=args.gan_checkpoint,
            input_nc=3,
            output_nc=1,
            ngf=64,
            n_blocks=9,
            device=gan_device,
            amp=args.gan_amp,
        )
    else:
        logger.warning(
            "No --gan_checkpoint provided — MID images will be identical to RGB (passthrough). "
            "Pass --gan_checkpoint to enable GAN-based RGB→MID translation."
        )
        gan_translator = GANTranslator(generator=None, device=gan_device)

    # --- Models ---
    logger.info(f"Building {args.model.upper()} trio ...")
    _trio_kwargs = dict(
        num_classes=NUM_CLASSES,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        min_size=args.min_size,
        max_size=args.max_size,
        ir_to_rgb=True,
        from_coco=args.from_coco,
        coco_src_indices=FLIR_TO_COCO_IDX if args.from_coco else None,
    )
    if args.model == "faster_rcnn":
        student, rgb_teacher, ir_teacher = build_faster_rcnn_trio(
            **_trio_kwargs, focal_gamma=args.focal_gamma
        )
    else:
        student, rgb_teacher, ir_teacher = build_fcos_trio(**_trio_kwargs)

    # --- MID DataLoader (GAN-translated RGB — used only to pretrain ir_teacher) ---
    mid_dataset = MIDDataset(rgb_train, gan_translator)
    mid_loader  = DataLoader(
        mid_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=0, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )

    # --- Optimizer cho student ---
    optimizer = torch.optim.SGD([
        {"params": [p for n, p in student.named_parameters() if "backbone" in n],
         "lr": args.lr_backbone},
        {"params": [p for n, p in student.named_parameters() if "backbone" not in n],
         "lr": args.lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iters,
    )

    # --- Evaluator ---
    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASSES,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    config = make_training_config(
        device_str, target_h=args.min_size, target_w=args.max_size,
        gan_checkpoint=args.gan_checkpoint, args=args,
    )
    phase_eval = PhaseEvaluator(
        evaluator=evaluator,
        ir_val_loader=ir_val_loader,
        device=device,
        eval_every_n=args.eval_every,
        rgb_val_loader=rgb_val_loader,
        vis_dir=os.path.join(args.output_dir, "vis"),
        vis_every_n=args.vis_every,
        vis_num_samples=16,
        class_names=FLIR_CLASSES,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
    )

    # --- Domain scheduler (BlockDomainScheduler: rgb / mid / ir short-cycle blocks) ---
    ratio_phases = parse_ratio_schedule(config.block_scheduler.ratio_schedule)
    block_scheduler = BlockDomainScheduler(
        ratio_phases, args.total_iters, cycle_iters=config.block_scheduler.cycle_iters,
    )
    logger.info(
        f"BlockDomainScheduler: ratio_schedule={ratio_phases}  "
        f"cycle_iters={config.block_scheduler.cycle_iters}"
    )

    # --- Teacher weights (unless resuming): load pretrained checkpoint if given,
    # otherwise pretrain from scratch — student inits from rgb_teacher either way ---
    if not args.resume:
        load_or_pretrain_teacher(
            rgb_teacher, args.teacher_rgb_weights, rgb_loader,
            args.teacher_pretrain_iters, device,
            args.lr_backbone, args.lr_head, label="rgb_teacher",
            save_path=f"{args.output_dir}/pretrain_rgb_teacher.pt",
            ir_val_loader=ir_val_loader, evaluator=evaluator,
            skip_pretrain=args.skip_teacher_pretrain,
        )
        load_or_pretrain_teacher(
            ir_teacher, args.teacher_ir_weights, mid_loader,
            args.teacher_pretrain_iters, device,
            args.lr_backbone, args.lr_head, label="ir_teacher",
            save_path=f"{args.output_dir}/pretrain_ir_teacher.pt",
            ir_val_loader=ir_val_loader, evaluator=evaluator,
            skip_pretrain=args.skip_teacher_pretrain,
        )
        # Student init = rgb_teacher weights (D3T convention).
        copy_student_to_teacher(teacher=student, student=rgb_teacher)
        logger.info("Student initialized from rgb_teacher weights.")

    # --- Trainer ---
    trainer = BlockDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        gan_translator=gan_translator,
        block_scheduler=block_scheduler,
        phase_evaluator=phase_eval,
    )

    # --- Best checkpoint callbacks ---
    os.makedirs(args.output_dir, exist_ok=True)

    def save_global_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        map50 = results["mAP@0.5"]
        path  = f"{args.output_dir}/best.pt"
        trainer.save_checkpoint(path)
        logger.info(f"[Global Best] mAP@0.5={map50:.4f}  phase={phase}  step={step}  → {path}")

    def save_phase_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        map50 = results["mAP@0.5"]
        path = f"{args.output_dir}/best_{phase}.pt"
        trainer.save_checkpoint(path)
        logger.info(f"[Phase Best] {phase}  mAP@0.5={map50:.4f}  step={step}  → {path}")

    phase_eval.register_best_fn(save_global_best)
    phase_eval.register_phase_best_fn(save_phase_best)

    # --- Resume ---
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        # Reset optimizer LRs to base values so CosineAnnealingLR reads correct base_lrs.
        optimizer.param_groups[0]["lr"] = args.lr_backbone
        optimizer.param_groups[1]["lr"] = args.lr_head
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.total_iters, last_epoch=-1,
        )
        for _ in range(trainer.global_step):
            lr_scheduler.step()
        logger.info(
            f"Resumed at global_step={trainer.global_step}  "
            f"remaining={args.total_iters - trainer.global_step} iters  "
            f"lr={optimizer.param_groups[-1]['lr']:.2e}"
        )
    else:
        # --- Baseline eval (only on fresh run) ---
        logger.info("\nBaseline evaluation (before training) ...")
        phase_eval.evaluate(student, global_step=0, phase_idx=0, trigger_reason="baseline")

    # --- Training loop ---
    remaining_iters = args.total_iters - trainer.global_step
    logger.info(
        f"\nStarting training: {remaining_iters} remaining iterations "
        f"(global_step {trainer.global_step} → {args.total_iters}) ..."
    )
    for _ in range(remaining_iters):
        log  = trainer.train_one_iteration()
        step = trainer.global_step  # incremented at end of train_one_iteration
        lr_scheduler.step()

        # Verbose log every 500 iters
        if step % 500 == 0:
            phase  = log.get("phase", "?")
            domain = log.get("domain", "?")
            lam    = log.get("lambda", 0.0)
            lr     = optimizer.param_groups[-1]["lr"]
            logger.info(
                f"[{step:07d}/{args.total_iters}]  "
                f"phase={phase:<8}  domain={domain:<4}  lambda={lam:.3f}  lr={lr:.2e}"
            )

        # Checkpoint
        if step % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            trainer.save_checkpoint(f"{args.output_dir}/ckpt_{step:07d}.pt")

    # --- Final eval + summary ---
    logger.info("\nFinal evaluation ...")
    final_phase_idx = block_scheduler.phase_for_iter(max(0, trainer.global_step - 1))
    phase_eval.evaluate(student, global_step=trainer.global_step,
                        phase_idx=final_phase_idx, trigger_reason="final")
    phase_eval.print_history()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_checkpoint(f"{args.output_dir}/final.pt")
    logger.info(f"Done.  global_step={trainer.global_step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True,
                   help="Path to align/ directory (contains JPEGImages/, Annotations/, ImageSets/)")
    p.add_argument("--output_dir",  default="./output")
    p.add_argument("--total_iters", type=int,   default=30_000)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head",     type=float, default=5e-4)
    p.add_argument("--min_size",    type=int,   default=512)
    p.add_argument("--max_size",    type=int,   default=640)
    p.add_argument("--eval_every",  type=int,   default=1_000)
    p.add_argument("--vis_every",   type=int,   default=1_000)
    p.add_argument("--save_every",  type=int,   default=5_000)
    p.add_argument("--model",       default="fcos",
                   choices=["fcos", "faster_rcnn"],
                   help="Detector backbone (default: fcos)")
    p.add_argument("--from_coco",   action="store_true",
                   help="Init head from COCO pretrained weights (91-class → replace head)")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss gamma for faster_rcnn classifier (default 2.0, 0=cross-entropy)")
    p.add_argument("--gan_checkpoint", default="",
                   help="Path to GAN generator checkpoint .pth (e.g. gan_mid/latest_net_G_A.pth). "
                        "If empty, MID = RGB passthrough (for debugging).")
    p.add_argument("--gan_amp",     action="store_true",
                   help="Use torch.autocast during GAN inference (saves VRAM)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from (e.g. output/best_phase0.pt)")
    p.add_argument("--device",      default="cuda",
                   choices=["cuda", "cpu", "mps"])

    # === BlockDomainScheduler ===
    p.add_argument("--ratio_schedule", default=BlockSchedulerConfig().ratio_schedule,
                   help="Comma-separated 'rgb:mid:ir' ratio per phase, e.g. "
                        "'7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8'")
    p.add_argument("--cycle_iters", type=int, default=BlockSchedulerConfig().cycle_iters,
                   help="Length (iters) of one rgb->mid->ir cycle within a phase.")

    # === Disjoint RGB source / real-IR target split (D3T paper 4.1) ===
    p.add_argument("--disjoint_split", dest="disjoint_split", action="store_true", default=True,
                   help="(default) split align_train into non-overlapping RGB-source / "
                        "real-IR-target halves.")
    p.add_argument("--no_disjoint_split", dest="disjoint_split", action="store_false",
                   help="Disable disjoint split (rgb & ir_train share the same aligned images).")
    p.add_argument("--source_frac", type=float, default=0.5,
                   help="Fraction of align_train used as RGB source (rest = real-IR target).")
    p.add_argument("--split_seed", type=int, default=-1,
                   help="-1 = 'first N' (paper); >=0 = deterministic shuffle before splitting.")

    # === Dense distillation (D3T-style, FCOS only) ===
    p.add_argument("--distill_ratio", type=float, default=DistillConfig().ratio,
                   help="Top-ratio region selection by harmony measure.")
    p.add_argument("--hm_alpha", type=float, default=DistillConfig().hm_alpha,
                   help="cls_prob exponent in harmony measure hm = cls_prob^a * iou^b.")
    p.add_argument("--hm_beta", type=float, default=DistillConfig().hm_beta,
                   help="iou (centerness) exponent in harmony measure.")
    p.add_argument("--un_alpha", type=float, default=DistillConfig().un_alpha,
                   help="Uncertainty weighting temperature: exp(-(1-hm)/un_alpha).")
    p.add_argument("--dw_logits", type=float, default=DistillConfig().w_logits,
                   help="QFLv2 (logits) distillation loss weight.")
    p.add_argument("--dw_deltas", type=float, default=DistillConfig().w_deltas,
                   help="GIoU (deltas) distillation loss weight.")
    p.add_argument("--dw_quality", type=float, default=DistillConfig().w_quality,
                   help="Centerness/quality BCE distillation loss weight.")
    p.add_argument("--dw_uhl", type=float, default=DistillConfig().w_uhl,
                   help="VFL/UHL distillation loss weight.")

    # === Two-teacher weighting + lambda ramp ===
    p.add_argument("--w_match", type=float, default=TeacherWeightConfig().w_match,
                   help="Weight for the teacher matching the current domain (high).")
    p.add_argument("--w_cross", type=float, default=TeacherWeightConfig().w_cross,
                   help="Weight for the cross-domain teacher (low).")
    p.add_argument("--lambda_max", type=float, default=LambdaConfig().lambda_max,
                   help="Max weight of the unsupervised distillation loss.")
    p.add_argument("--lambda_ramp_iters", type=int, default=LambdaConfig().ramp_iters,
                   help="Iterations to ramp lambda 0 -> lambda_max (0 = fixed from the start).")
    p.add_argument("--ema_alpha", type=float, default=0.9996,
                   help="EMA decay for teacher updates.")

    # === Teacher pretraining (before the block-scheduled curriculum) ===
    p.add_argument("--skip_teacher_pretrain", action="store_true",
                   help="Skip supervised teacher pretraining (assume teachers are "
                        "already trained, e.g. loaded via --resume, or via "
                        "--teacher_rgb_weights/--teacher_ir_weights below).")
    p.add_argument("--teacher_pretrain_iters", type=int, default=5_000,
                   help="Supervised pretraining iterations for each teacher "
                        "(rgb_teacher on RGB, ir_teacher on GAN-translated MID). "
                        "Ignored for a teacher whose --teacher_*_weights is given.")
    p.add_argument("--teacher_rgb_weights", default="",
                   help="Path to a pretrained rgb_teacher checkpoint (.pt/.pth). If given, "
                        "skips pretraining rgb_teacher and loads these weights instead "
                        "(same idea as training_full_d3t_v2_*.py's --teacher-rgb-weights). "
                        "Accepts either {'teacher': state_dict} or a plain state_dict.")
    p.add_argument("--teacher_ir_weights", default="",
                   help="Path to a pretrained ir_teacher checkpoint (.pt/.pth). If given, "
                        "skips pretraining ir_teacher and loads these weights instead "
                        "(same idea as training_full_d3t_v2_*.py's --teacher-ir-weights). "
                        "Accepts either {'teacher': state_dict} or a plain state_dict.")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

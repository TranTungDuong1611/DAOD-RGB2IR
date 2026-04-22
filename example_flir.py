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

# Đảm bảo thư mục của script luôn nằm trong sys.path,
# cho dù chạy từ thư mục nào.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from adaptive_threshold import AdaptiveThresholdConfig, AdaptiveThresholdScheduler, TeacherThresholds
from config import (
    AugConfig,
    CurriculumConfig,
    EMAConfig,
    LossConfig,
    MidRoutingConfig,
    SAGAConfig,
    SoftSAGAConfig,
    TeacherUpdateConfig,
    TrainingConfig,
)
from datasets import (
    FLIR_CLASSES,
    NUM_CLASSES,
    FLIRIRDataset,
    FLIRIRValDataset,
    FLIRRGBDataset,
    ir_collate,
    ir_val_collate,
    rgb_collate,
)
from ema import copy_student_to_teacher
from evaluator import DetectionEvaluator, PhaseEvaluator
from fcos_wrapper import build_fcos_trio
from scheduler import Phase
from trainer import CurriculumDomainAdaptationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_training_config(device: str) -> TrainingConfig:
    return TrainingConfig(
        ema=EMAConfig(alpha=0.9996, use_warmup=True),
        saga=SAGAConfig(apply_prob=0.8),
        soft_saga=SoftSAGAConfig(
            alpha_near_rgb=0.70,
            alpha_intermediate=0.50,
            alpha_near_ir=0.25,
        ),
        mid_routing=MidRoutingConfig(
            near_rgb_teacher_source="rgb",  near_rgb_ema_target="rgb",
            near_rgb_rgb_weight=1.0,        near_rgb_ir_weight=0.0,
            intermediate_teacher_source="both", intermediate_ema_target="ir",
            intermediate_ema_alpha=0.9998,
            intermediate_rgb_weight=0.5,    intermediate_ir_weight=0.5,
            near_ir_teacher_source="ir",    near_ir_ema_target="ir",
            near_ir_rgb_weight=0.0,         near_ir_ir_weight=1.0,
        ),
        aug=AugConfig(
            hflip_prob=0.5,
            blur_prob=0.5,
            blur_sigma_max=1.0,
            brightness_prob=0.3,
            brightness_mag=0.2,
            contrast_prob=0.3,
            contrast_mag=0.2,
        ),
        curriculum=CurriculumConfig(
            phase1_end=3_000,       # RGB warmup
            phase2_end=9_000,       # RGB + MID
            phase3_end=15_000,      # MID + IR
            phase2_rgb_ratio=0.67,  # RGB:MID = 2:1 → MID ít hơn, ổn định hơn
            phase3_mid_ratio=0.4,
            phase4_mid_every_n=5,
        ),
        loss=LossConfig(
            rgb_gt_weight=1.0,
            rgb_pseudo_weight=0.0,
            mid_rgb_weight=0.1,
            mid_ir_weight=0.5,
            mid_gt_weight=1.0,          # anchor MID step to GT → prevents forgetting
            ir_rgb_teacher_weight=0.4,
            ir_ir_teacher_weight=0.6,
        ),
        teacher_update=TeacherUpdateConfig(
            rgb_update_rgb_teacher=True,
            mid_update_rgb_teacher=False,
            mid_update_ir_teacher=True,   # MID step EMA update ir_teacher
            ir_update_ir_teacher=True,
        ),
        pseudo_label_conf_thresh=0.7,
        device=device,
        log_interval=100,
    )


def make_adaptive_threshold() -> AdaptiveThresholdScheduler:
    return AdaptiveThresholdScheduler(AdaptiveThresholdConfig(
        rgb_teacher=TeacherThresholds(
            phase1=0.90, phase2=0.88, phase3=0.80, phase4=0.70,
        ),
        ir_teacher=TeacherThresholds(
            phase1=0.95, phase2=0.90, phase3=0.85, phase4=0.78,
        ),
    ))


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
    rgb_train = FLIRRGBDataset(data_root, split="train")
    ir_train  = FLIRIRDataset( data_root, split="train")
    ir_val    = FLIRIRValDataset(data_root, split="validation")

    logger.info(
        f"  RGB train     : {len(rgb_train):>5} images  (labeled, source domain)\n"
        f"  IR  train     : {len(ir_train):>5} images  (unlabeled, target domain)\n"
        f"  IR  val       : {len(ir_val):>5} images  (labeled, for mAP)"
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

    # --- Models ---
    logger.info("Building FCOS trio ...")
    student, rgb_teacher, ir_teacher = build_fcos_trio(
        num_classes=NUM_CLASSES,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        min_size=args.min_size,
        max_size=args.max_size,
        ir_to_rgb=True,
        from_coco=args.from_coco,
    )
    copy_student_to_teacher(rgb_teacher, student)
    copy_student_to_teacher(ir_teacher,  student)

    # --- Optimizer ---
    optimizer = torch.optim.SGD([
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" not in n and p.requires_grad],
         "lr": args.lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iters,
    )

    # --- Adaptive threshold ---
    thresh = make_adaptive_threshold()
    logger.info("\n" + thresh.summary())

    # --- Evaluator ---
    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASSES,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    phase_eval = PhaseEvaluator(
        evaluator=evaluator,
        ir_val_loader=ir_val_loader,
        device=device,
        eval_every_n=args.eval_every,
        vis_dir=os.path.join(args.output_dir, "vis"),
        vis_num_samples=8,
        vis_score_thresh=0.3,
        class_names=FLIR_CLASSES,
    )

    # --- Trainer ---
    config = make_training_config(device_str)
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        threshold_scheduler=thresh,
        phase_evaluator=phase_eval,
    )

    # --- Best checkpoint callback ---
    os.makedirs(args.output_dir, exist_ok=True)

    def save_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        map50 = results["mAP@0.5"]
        path  = f"{args.output_dir}/best.pt"
        trainer.save_checkpoint(path)
        logger.info(f"[Best] mAP@0.5={map50:.4f}  phase={phase}  step={step}  → {path}")

    phase_eval.register_best_fn(save_best)

    # --- Baseline eval ---
    logger.info("\nBaseline evaluation (before training) ...")
    phase_eval.evaluate(student, global_step=0,
                        current_phase=Phase.PHASE1_RGB_WARMUP,
                        trigger_reason="baseline")

    # --- Training loop ---
    logger.info(f"\nStarting training: {args.total_iters} iterations ...")
    for i in range(args.total_iters):
        log = trainer.train_one_iteration()
        lr_scheduler.step()

        # Verbose log every 500 iters
        if i % 500 == 0:
            phase  = log.get("phase", "?")
            domain = log.get("domain", "?")
            t_rgb  = log.get("thresh_rgb", "?")
            t_ir   = log.get("thresh_ir",  "?")
            lr     = optimizer.param_groups[-1]["lr"]
            thresh_str = f"({t_rgb:.2f}/{t_ir:.2f})" if isinstance(t_rgb, float) else ""
            logger.info(
                f"[{i:07d}/{args.total_iters}]  "
                f"phase={phase:<22}  domain={domain}  "
                f"thresh={thresh_str}  lr={lr:.2e}"
            )

        # Checkpoint
        if i > 0 and i % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            trainer.save_checkpoint(f"{args.output_dir}/ckpt_{i:07d}.pt")

    # --- Final eval + summary ---
    logger.info("\nFinal evaluation ...")
    phase_eval.evaluate(student, global_step=trainer.global_step,
                        current_phase=Phase.PHASE4_IR_FOCUS,
                        trigger_reason="final")
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
    p.add_argument("--total_iters", type=int,   default=20_000)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head",     type=float, default=1e-3)
    p.add_argument("--min_size",    type=int,   default=512)
    p.add_argument("--max_size",    type=int,   default=640)
    p.add_argument("--eval_every",  type=int,   default=2_000)
    p.add_argument("--save_every",  type=int,   default=5_000)
    p.add_argument("--from_coco",   action="store_true",
                   help="Init head from COCO pretrained FCOS (91-class → replace head)")
    p.add_argument("--device",      default="cuda",
                   choices=["cuda", "cpu", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

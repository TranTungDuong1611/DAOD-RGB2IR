import argparse
import logging
import os
import sys
import torch
import time
from torch.utils.data import DataLoader

# Ensure the script directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TrainingConfig, FCOSModelConfig, DistillConfig,
    EMAConfig, SAGAConfig, SoftSAGAConfig, MidRoutingConfig,
    AugConfig, CurriculumConfig, LossConfig, TeacherUpdateConfig,
    DataConfig, DataLoaderConfig, TrainLoaderConfig, EvalLoaderConfig,
    Phase, TeacherSchedule
)
from data import (
    FLIR_CLASSES, NUM_CLASSES, FLIRIRDataset, FLIRIRValDataset,
    FLIRRGBDataset, ir_collate, ir_val_collate, rgb_collate
)
from ema import copy_student_to_teacher
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.custom_fcos import build_fcos_trio, FCOSDistillAdapter
from trainer import CurriculumDomainAdaptationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def make_training_config(args) -> TrainingConfig:
    """
    Creates the master TrainingConfig based on CLI arguments and 
    the new phase-based curriculum strategy.
    """
    # Calculate phase boundaries based on total iterations
    # Example: 10k total 
    b1, b2, b3 = 2000, 5000, 8000 

    return TrainingConfig(
        model=FCOSModelConfig(
            num_classes=NUM_CLASSES,
            from_coco=args.from_coco,
            min_size=args.min_size,
            max_size=args.max_size,
            vfl_alpha=0.75,
            vfl_gamma=2.0
        ),
        distill=DistillConfig(
            phase_boundaries=(b1, b2, b3),
            hm_alpha=1.0,
            hm_beta=1.0,
            un_regular_alpha=4.0,
            # Per-teacher adaptive parameters: (ratio, min_hm)
            rgb_teacher=TeacherSchedule(
                phase1=(0.010, 0.45), phase2=(0.020, 0.35), 
                phase3=(0.040, 0.25), phase4=(0.060, 0.20)
            ),
            ir_teacher=TeacherSchedule(
                phase1=(0.002, 0.60), phase2=(0.005, 0.50), 
                phase3=(0.015, 0.35), phase4=(0.030, 0.25)
            )
        ),
        curriculum=CurriculumConfig(
            phase1_end=2000,
            phase2_end=5000,
            phase3_end=8000,
            phase2_rgb_sampling_ratio=0.7,
            phase3_rgb_sampling_ratio=0.3,
        ),
        loss=LossConfig(
            p1_sup_weight=1.0, p1_distill_weight=0.0,
            p2_sup_weight=1.0, p2_distill_weight=0.7,
            p3_sup_weight=0.5, p3_distill_weight=1.0,
            p4_sup_weight=0.0, p4_distill_weight=1.0,
            weight_logits=4.0, weight_deltas=1.0, weight_quality=1.0
        ),
        ema=EMAConfig(alpha=0.9996),
        soft_saga=SoftSAGAConfig(
            alpha_near_rgb=0.75,
            alpha_intermediate=0.5,
            alpha_near_ir=0.25
        ),
        data=DataConfig(root=args.data_root),
        step2_start=2000,
        max_iter=10000,
        eval_period=args.eval_every,
        output_dir=args.output_dir,
        device=args.device
    )

# ---------------------------------------------------------------------------
# 2. Main Execution Logic
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device)
    
    logger.info("=== Curriculum Domain Adaptation - Upgraded FLIR ADAS ===")
    
    # --- 1. Load Config ---
    config = make_training_config(args)

    # --- 2. Datasets & Loaders ---
    logger.info("Preparing DataLoaders...")
    rgb_train = FLIRRGBDataset(config.data.root, split="train")
    ir_train  = FLIRIRDataset( config.data.root, split="train")
    ir_val    = FLIRIRValDataset(config.data.root, split="validation")
    rgb_val   = FLIRRGBDataset( config.data.root, split="validation")

    rgb_loader = DataLoader(
        rgb_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=args.workers, drop_last=True
    )
    ir_loader = DataLoader(
        ir_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=ir_collate, num_workers=args.workers, drop_last=True
    )
    # Note: Evaluator expects (images, targets) from val_loaders
    ir_val_loader = DataLoader(
        ir_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=ir_val_collate, num_workers=args.workers
    )
    rgb_val_loader = DataLoader(
        rgb_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=rgb_collate, num_workers=args.workers
    )

    # --- 3. Models (Trio) ---
    logger.info(f"Building FCOS Trio (Student + 2 Teachers)...")
    # build_fcos_trio returns FCOSDetector wrappers containing CustomFCOS cores
    student, rgb_teacher, ir_teacher = build_fcos_trio(
        num_classes=config.model.num_classes,
        pretrained_backbone=True,
        min_size=config.model.min_size,
        max_size=config.model.max_size,
        from_coco=config.model.from_coco,
        vfl_alpha=config.model.vfl_alpha,
        vfl_gamma=config.model.vfl_gamma
    )
    
    # Sync initial weights
    copy_student_to_teacher(rgb_teacher, student)
    copy_student_to_teacher(ir_teacher,  student)

    # --- 4. Optimizer & Scheduler ---
    optimizer = torch.optim.SGD(
        student.parameters(), 
        lr=args.lr_head, momentum=0.9, weight_decay=1e-4
    )

    # --- 5. Knowledge Distillation Adapter ---
    distill_adapter = FCOSDistillAdapter(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        config=config
    )

    # --- 6. Evaluator ---
    evaluator_core = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASSES,
        iou_thresholds=[0.5] # mAP@0.5 focus
    )
    phase_eval = PhaseEvaluator(
        evaluator=evaluator_core,
        ir_val_loader=ir_val_loader,
        rgb_val_loader=rgb_val_loader,
        device=device,
        eval_every_n=args.eval_every,
        vis_dir=os.path.join(args.output_dir, "vis")
    )

    # --- 7. Trainer ---
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        val_loader=ir_val_loader, # Primary evaluation on IR
        distill_adapter=distill_adapter
    )
    
    # Link phase_eval to trainer manually or pass it in
    trainer.phase_evaluator = phase_eval

    # --- 8. Resume / Baseline ---
    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        logger.info("Running baseline evaluation...")
        phase_eval.evaluate(student, global_step=0, current_phase=Phase.PHASE1_RGB_WARMUP)

    # --- 9. Start Training ---
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving final state...")
    finally:
        trainer.save_checkpoint("final_model.pth")
        phase_eval.print_history()
        logger.info("Done.")

# ---------------------------------------------------------------------------
# 3. CLI Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Curriculum DA Training for FLIR IR/RGB")
    p.add_argument("--data_root", required=True, help="Path to FLIR aligned dataset")
    p.add_argument("--output_dir", default="./output_flir")
    p.add_argument("--total_iters", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--min_size", type=int, default=512)
    p.add_argument("--max_size", type=int, default=640)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--from_coco", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
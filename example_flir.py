import argparse
import logging
import os
import sys
import torch
import time

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
    FLIRRGBDataset, ir_collate, ir_val_collate, rgb_collate, build_dataloaders
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
            from_coco=True,
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
        output_dir=args.output_dir,
        device=args.device,
        eval_period=500
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
    loaders = build_dataloaders(config)
    
    rgb_train_loader = loaders["rgb_train"]
    ir_train_loader  = loaders["ir_train"]
    ir_val_loader    = loaders["ir_val"]
    rgb_val_loader   = loaders["rgb_val"]

    logger.info(
        f"  RGB Train: {len(rgb_train_loader.dataset)} images\n"
        f"  IR  Train: {len(ir_train_loader.dataset)} images\n"
        f"  IR  Val  : {len(ir_val_loader.dataset)} images"
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

    student     = student.to(device)
    rgb_teacher = rgb_teacher.to(device)
    ir_teacher  = ir_teacher.to(device)

    # Debug checklist
    print("student device:", next(student.parameters()).device)
    print("student training:", student.training)  # doit être False pendant eval
    print("score_thresh:", student.model.score_thresh)

    # Test forward manual avec 1 image
    student.eval()
    student.to(device)
    with torch.no_grad():
        dummy = torch.randn(1, 3, 512, 640).to(device)
        out = student(dummy)
        print("num detections:", len(out[0]["boxes"]))
        if len(out[0]["boxes"]) > 0:
            print("scores:", out[0]["scores"][:5])
        else:
            print("→ aucune détection — score_thresh trop élevé ou model sur mauvais device")

    
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
        eval_every_n=1000,
        vis_dir=os.path.join(args.output_dir, "vis")
    )

    def on_best_found(results):
        trainer.save_checkpoint("best_model.pth")
        logger.info(f" >>> New best mAP saved: {results.get('mAP@0.5'):.4f}")

    phase_eval.register_best_fn(on_best_found)

    # --- 7. Trainer ---
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_train_loader,
        ir_loader=ir_train_loader,
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
        while trainer.global_step < config.max_iter:
            # Chạy 1 bước huấn luyện
            start_time = time.time()
            logs = trainer.train_one_iteration()

            iter_time = time.time() - start_time
            current_step = trainer.global_step

            if current_step % config.log_interval == 0:
                logs["iter_time"] = iter_time
                trainer._log(logs)
            
            # Lấy phase hiện tại từ config
            current_phase = config.get_phase(current_step)
            
            # GỌI EVALUATOR Ở ĐÂY
            # Hàm .step() sẽ tự kiểm tra: (step % 1000 == 0) HOẶC (Phase thay đổi)
            eval_results = phase_eval.step(
                model=student, 
                global_step=current_step, 
                current_phase=current_phase
            )
            
            # Nếu vừa chạy eval xong, bạn có thể thực hiện logic phụ ở đây (nếu muốn)
            if eval_results:
                logger.info(f"Iteration {current_step} evaluation completed.")

            if current_step % 5000 == 0:
                trainer.save_checkpoint(f"checkpoint_{current_step:06d}.pt")

    except KeyboardInterrupt:
        logger.info("Training manually interrupted.")
    finally:
        trainer.save_checkpoint("final_model.pth")
        phase_eval.print_history()

# ---------------------------------------------------------------------------
# 3. CLI Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Curriculum DA Training for FLIR IR/RGB")
    p.add_argument("--data_root", required=True, help="Path to FLIR aligned dataset")
    p.add_argument("--output_dir", default="./output_flir")
    p.add_argument("--total_iters", type=int, default=10000)
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
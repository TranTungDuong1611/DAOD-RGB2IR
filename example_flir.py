"""Command-line entrypoint for FLIR RGB baseline and D3T curriculum training."""

import argparse
import logging
import os

import torch

from config import (
    AugConfig,
    CurriculumConfig,
    DataConfig,
    DataLoaderConfig,
    DistillConfig,
    EMAConfig,
    EvalLoaderConfig,
    FCOSModelConfig,
    LossConfig,
    Phase,
    SoftSAGAConfig,
    TeacherSchedule,
    TrainLoaderConfig,
    TrainingConfig,
)
from data import FLIR_CLASSES, NUM_CLASSES, build_dataloaders
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.fcos_factory import build_fcos_d3t_trio
from models.torchvision_fcos_adapter import ClassificationInitMode
from trainer import CurriculumDomainAdaptationTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _phase_boundaries(total_iters: int) -> tuple[int, int, int]:
    """Create monotonic shortened boundaries for both normal and smoke runs."""

    first = min(total_iters, max(1, round(total_iters * 0.20)))
    second = min(total_iters, max(first, round(total_iters * 0.50)))
    third = min(total_iters, max(second, round(total_iters * 0.80)))
    return first, second, third


def _weight_identifier(value: str):
    return None if value.strip().lower() in {"none", ""} else value


def build_training_config(args) -> TrainingConfig:
    """Map every training-relevant CLI argument into one effective config."""

    total_iters = int(args.total_iters)
    phase1_end, phase2_end, phase3_end = _phase_boundaries(total_iters)
    weights = _weight_identifier(args.weights)
    pretrained_backbone = (
        bool(args.pretrained_backbone)
        if args.pretrained_backbone is not None
        else weights is not None
    )


    curriculum = CurriculumConfig(
        phase1_end=phase1_end,
        phase2_end=phase2_end,
        phase3_end=phase3_end,
        phase2_rgb_sampling_ratio=args.phase2_rgb_ratio,
        phase3_rgb_sampling_ratio=args.phase3_rgb_ratio,
    )

    return TrainingConfig(
        model=FCOSModelConfig(
            num_classes=NUM_CLASSES,
            class_names=tuple(FLIR_CLASSES),
            weights=weights,
            classification_init_mode=ClassificationInitMode(
                args.classification_init
            ),
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_size=args.min_size,
            max_size=args.max_size,
            center_sampling_radius=args.center_sampling_radius,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            topk_candidates=args.topk_candidates,
            detections_per_img=args.detections_per_img,
            from_coco=weights is not None,
            vfl_alpha=0.75,
            vfl_gamma=2.0,
        ),
        distill=DistillConfig(
            phase_boundaries=(phase1_end, phase2_end, phase3_end),
            hm_alpha=1.0,
            hm_beta=1.0,
            un_regular_alpha=4.0,
            rgb_teacher=TeacherSchedule(
                phase1=(0.010, 0.45),
                phase2=(0.020, 0.35),
                phase3=(0.040, 0.25),
                phase4=(0.060, 0.20),
            ),
            ir_teacher=TeacherSchedule(
                phase1=(0.002, 0.60),
                phase2=(0.005, 0.50),
                phase3=(0.015, 0.35),
                phase4=(0.030, 0.25),
            ),
        ),
        curriculum=curriculum,
        loss=LossConfig(
            p1_sup_weight=1.0,
            p1_distill_weight=0.0,
            p2_sup_weight=1.0,
            p2_distill_weight=0.7,
            p3_sup_weight=0.5,
            p3_distill_weight=1.0,
            p4_sup_weight=0.0,
            p4_distill_weight=1.0,
            weight_logits=4.0,
            weight_deltas=1.0,
            weight_quality=1.0,
        ),
        ema=EMAConfig(alpha=args.ema_alpha, start_steps=args.ema_start),
        soft_saga=SoftSAGAConfig(
            alpha_near_rgb=0.75,
            alpha_intermediate=0.5,
            alpha_near_ir=0.25,
        ),
        aug=AugConfig(
            hflip_prob=args.hflip_prob,
            blur_prob=args.blur_prob,
            brightness_prob=args.brightness_prob,
            contrast_prob=args.contrast_prob,
        ),
        data=DataConfig(root=args.data_root),
        loader=DataLoaderConfig(
            train=TrainLoaderConfig(
                batch_size=args.batch_size,
                num_workers=args.workers,
            ),
            eval=EvalLoaderConfig(
                batch_size=args.eval_batch_size,
                num_workers=args.workers,
            ),
        ),
        step2_start=phase1_end,
        max_iter=total_iters,
        total_iters=total_iters,
        output_dir=args.output_dir,
        device=args.device,
        workflow=args.workflow,
        teacher_mode=args.teacher_mode,
        log_interval=args.log_interval,
        eval_period=args.eval_every,
    )


# Backward-compatible name used by earlier local launch scripts.
make_training_config = build_training_config


def main(args) -> None:
    config = make_training_config(args)
    device = torch.device(config.device)
    logger.info(
        "FLIR D3T: workflow=%s teacher_mode=%s mode=%s weights=%s device=%s total_iters=%d",
        config.workflow,
        config.teacher_mode,
        config.model.classification_init_mode.value,
        config.model.weights or "none",
        device,
        config.total_iters,
    )

    loaders = build_dataloaders(config)
    rgb_train_loader = loaders["rgb_train"]
    ir_train_loader = loaders["ir_train"]
    rgb_val_loader = loaders["rgb_val"]
    ir_val_loader = loaders["ir_val"]
    logger.info(
        "RGB train=%d, IR train=%d, RGB val=%d, IR val=%d",
        len(rgb_train_loader.dataset),
        len(ir_train_loader.dataset),
        len(rgb_val_loader.dataset),
        len(ir_val_loader.dataset),
    )

    student, rgb_teacher, ir_teacher = build_fcos_d3t_trio(config)
    student.to(device)
    rgb_teacher.to(device)
    ir_teacher.to(device)
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr_head,
        momentum=0.9,
        weight_decay=1e-4,
    )

    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=list(FLIR_CLASSES),
        iou_thresholds=[0.5],
    )
    phase_evaluator = PhaseEvaluator(
        evaluator=evaluator,
        ir_val_loader=ir_val_loader,
        rgb_val_loader=rgb_val_loader,
        device=device,
        eval_every_n=config.eval_period,
        vis_dir=os.path.join(config.output_dir, "vis"),
    )
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_train_loader,
        ir_loader=ir_train_loader,
        val_loader=ir_val_loader,
        phase_evaluator=phase_evaluator,
    )

    def on_best_found(results):
        trainer.save_checkpoint("best_model.pth")
        logger.info(
            "New best IR mAP@0.5 saved: %.4f",
            results.get("mAP@0.5", 0.0),
        )

    phase_evaluator.register_best_fn(on_best_found)

    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        logger.info("Running initial IR evaluation")
        phase_evaluator.evaluate(
            student,
            global_step=trainer.global_step,
            current_phase=Phase.PHASE1_RGB_WARMUP,
            trigger_reason="initial",
        )

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        trainer.save_checkpoint("final_model.pth")
        phase_evaluator.print_history()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum D3T training for aligned FLIR RGB/IR data"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="./output_flir")
    parser.add_argument("--total_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--min_size", type=int, default=512)
    parser.add_argument("--max_size", type=int, default=640)
    parser.add_argument("--center_sampling_radius", type=float, default=1.5)
    parser.add_argument("--score_thresh", type=float, default=0.2)
    parser.add_argument("--nms_thresh", type=float, default=0.6)
    parser.add_argument("--topk_candidates", type=int, default=1000)
    parser.add_argument("--detections_per_img", type=int, default=100)
    parser.add_argument("--trainable_backbone_layers", type=int, default=3)
    parser.add_argument(
        "--weights",
        choices=("DEFAULT", "none"),
        default="DEFAULT",
        help="Torchvision FCOS COCO weights or none",
    )
    parser.add_argument(
        "--from_coco",
        action="store_true",
        help="Deprecated compatibility flag; COCO weights are selected explicitly with --weights DEFAULT",
    )
    parser.add_argument(
        "--pretrained-backbone",
        dest="pretrained_backbone",
        action="store_true",
    )
    parser.add_argument(
        "--no-pretrained-backbone",
        dest="pretrained_backbone",
        action="store_false",
    )
    parser.set_defaults(pretrained_backbone=None)
    parser.add_argument(
        "--classification-init",
        dest="classification_init",
        choices=[mode.value for mode in ClassificationInitMode],
        default=ClassificationInitMode.COCO_TOWER.value,
    )
    parser.add_argument(
        "--workflow",
        choices=("curriculum", "rgb_baseline"),
        default="curriculum",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("rgb", "ir", "two_teacher"),
        default="two_teacher",
        help="Enable one teacher for staged smoke/ablation or both for curriculum",
    )
    parser.add_argument("--phase2-rgb-ratio", type=float, default=0.7)
    parser.add_argument("--phase3-rgb-ratio", type=float, default=0.3)
    parser.add_argument("--ema-alpha", type=float, default=0.9996)
    parser.add_argument("--ema-start", type=int, default=6000)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--blur-prob", type=float, default=0.5)
    parser.add_argument("--brightness-prob", type=float, default=0.3)
    parser.add_argument("--contrast-prob", type=float, default=0.3)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.from_coco:
        args.weights = "DEFAULT"
    return args


if __name__ == "__main__":
    main(parse_args())

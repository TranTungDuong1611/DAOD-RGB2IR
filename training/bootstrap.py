"""Assemble training dependencies while the entrypoint owns the run lifecycle."""

import logging
import os

import torch

from config import TrainingConfig
from data import FLIR_CLASSES, NUM_CLASSES, build_dataloaders
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.fcos_factory import build_fcos_d3t_trio
from optimization import build_lr_scheduler, build_optimizer
from trainer import CurriculumDomainAdaptationTrainer


def move_model_trio_to_device(student, rgb_teacher, ir_teacher, device):
    """Move the student and whichever teachers are enabled to one device."""

    student = student.to(device)
    if rgb_teacher is not None:
        rgb_teacher = rgb_teacher.to(device)
    if ir_teacher is not None:
        ir_teacher = ir_teacher.to(device)
    return student, rgb_teacher, ir_teacher


def build_training(
    config: TrainingConfig, *, logger: logging.Logger
) -> CurriculumDomainAdaptationTrainer:
    """Build and wire the existing factories in their original order."""
    device = torch.device(config.device)
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

    student, rgb_teacher, ir_teacher = move_model_trio_to_device(
        *build_fcos_d3t_trio(config),
        device,
    )
    optimizer = build_optimizer(student, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)

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
        phase_evaluator=phase_evaluator,
        lr_scheduler=lr_scheduler,
    )

    # Bind the callback before initial evaluation or resume, as in the entrypoint.
    def on_best_found(results):
        trainer.save_checkpoint("best_model.pth")
        logger.info(
            "New best IR mAP@0.5 saved: %.4f",
            results.get("mAP@0.5", 0.0),
        )

    phase_evaluator.register_best_fn(on_best_found)

    return trainer

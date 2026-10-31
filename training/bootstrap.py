"""Assemble training dependencies while the entrypoint owns the run lifecycle."""

import logging
import os

import torch
from torch.utils.data import DataLoader, Subset

from config import TrainingConfig
from data import FLIR_CLASSES, NUM_CLASSES, build_dataloaders
from data.dataloaders import ir_collate
from data.ir_annotations import IRAnnotationReader
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.fcos_factory import build_fcos_d3t_trio
from optimization import build_lr_scheduler, build_optimizer
from trainer import CurriculumDomainAdaptationTrainer
from training.pseudo_label_eval import FixedPseudoLabelEvaluator, ValidationImages, select_fixed_subset


def move_model_trio_to_device(student, rgb_teacher, ir_teacher, device):
    """Move the student and whichever teachers are enabled to one device."""

    student = student.to(device)
    if rgb_teacher is not None:
        rgb_teacher = rgb_teacher.to(device)
    if ir_teacher is not None:
        ir_teacher = ir_teacher.to(device)
    return student, rgb_teacher, ir_teacher


def build_pseudo_label_evaluator(config, *, logger):
    """Assemble isolated validation loaders for each enabled teacher's domain."""
    audit = config.pseudo_label_eval
    if not audit.enabled or config.workflow == "rgb_baseline":
        return None
    reader = IRAnnotationReader(config.data.root)
    # Each loader owns a generator so iterator/worker seeding never consumes
    # the global RNG used by training. Retain the final incomplete batch.
    def loader(data):
        return DataLoader(
            data, batch_size=config.loader.eval.batch_size,
            num_workers=config.loader.eval.num_workers, shuffle=False, drop_last=False,
            pin_memory=config.device.startswith("cuda"), collate_fn=ir_collate,
            generator=torch.Generator().manual_seed(audit.subset_seed),
        )
    # Domain choice stays in assembly/trainer policy. No SAGA is applied to
    # audits, and disabled teacher domains do not create extra datasets/loaders.
    domains = ("rgb", "ir") if config.teacher_mode == "two_teacher" else (config.teacher_mode,)
    fixed_loaders, full_loaders, fixed_ids = {}, {}, {}
    for domain in domains:
        dataset = ValidationImages(config.data.root, domain)
        indices = select_fixed_subset(
            dataset.samples, reader, sample_size=audit.sample_size, seed=audit.subset_seed,
        )
        fixed_loaders[domain] = loader(Subset(dataset, indices))
        full_loaders[domain] = loader(dataset)
        fixed_ids[domain] = tuple(dataset.samples[i][1] for i in indices)
        logger.info("Pseudo-label %s validation: fixed_images=%d full_images=%d batch_size=%d seed=%d",
                    domain.upper(), len(indices), len(dataset), config.loader.eval.batch_size, audit.subset_seed)
    return FixedPseudoLabelEvaluator(
        fixed_loaders=fixed_loaders, full_loaders=full_loaders,
        reader=reader, device=torch.device(config.device),
        fixed_sample_ids=fixed_ids,
    )


def build_training(
    config: TrainingConfig, *, logger: logging.Logger
) -> CurriculumDomainAdaptationTrainer:
    """Build and wire the existing factories in their original order."""
    # Data assembly: create all source/target training and validation loaders
    # from the effective config, then expose their sizes in the existing log.
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

    # Model assembly: build the student and configured teachers before binding
    # optimization state to the student's actual device-resident parameters.
    student, rgb_teacher, ir_teacher = move_model_trio_to_device(
        *build_fcos_d3t_trio(config),
        device,
    )
    optimizer = build_optimizer(student, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)

    # Evaluation assembly: pair the existing AP50 metric implementation with
    # domain loaders, evaluation cadence, and best-result tracking.
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
    # Wire the assembled dependencies into the orchestrator. Run lifecycle
    # decisions such as resume, initial evaluation, and final save stay outside.
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
        pseudo_label_evaluator=build_pseudo_label_evaluator(config, logger=logger),
    )

    # Connect an evaluator improvement to saving the complete training state.
    # Bind it before initial evaluation or resume, as in the original entrypoint.
    def on_best_found(results):
        trainer.save_checkpoint("best_model.pth")
        logger.info(
            "New best IR mAP@0.5 saved: %.4f",
            results.get("mAP@0.5", 0.0),
        )

    phase_evaluator.register_best_fn(on_best_found)

    # Return a ready-to-run object without starting training during assembly.
    return trainer

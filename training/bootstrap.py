"""Assemble training dependencies while the entrypoint owns the run lifecycle."""

import logging
import os

import torch
from torch.utils.data import DataLoader, Subset

from config import TrainingConfig
from data import FLIR_CLASSES, NUM_CLASSES, build_dataloaders
from data.augmentations import StudentAugmentor
from data.datasets import FLIRIRDataset
from data.dataloaders import ir_collate
from data.ir_annotations import IRAnnotationReader
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.fcos_models import build_fcos_trio
from optimization import build_lr_scheduler, build_optimizer
from scheduler import BridgeDomainScheduler, CurriculumScheduler
from training.data_flow import RouteDataFlow
from trainer import CurriculumDomainAdaptationTrainer
from training.gan_bridge import load_gan_translator
from training.pseudo_label_eval import FixedIRPseudoLabelEvaluator, select_fixed_ir_subset


def move_model_trio_to_device(student, rgb_teacher, ir_teacher, device):
    """Move the student and whichever teachers are enabled to one device."""

    student = student.to(device)
    if rgb_teacher is not None:
        rgb_teacher = rgb_teacher.to(device)
    if ir_teacher is not None:
        ir_teacher = ir_teacher.to(device)
    return student, rgb_teacher, ir_teacher


def build_pseudo_label_evaluator(config, *, logger):
    """Resolve audit policy once and assemble dedicated, deterministic IR loaders."""
    audit = config.pseudo_label_eval
    if not audit.enabled or config.workflow == "rgb_baseline":
        return None
    dataset = FLIRIRDataset(config.data.root, split="validation")
    reader = IRAnnotationReader(config.data.root)
    indices = select_fixed_ir_subset(
        dataset.ir_paths, reader, sample_size=audit.sample_size, seed=audit.subset_seed,
    )
    # Each loader owns a generator so iterator/worker seeding never consumes
    # the global RNG used by training. Retain the final incomplete batch.
    def loader(data):
        return DataLoader(
            data, batch_size=config.loader.eval.batch_size,
            num_workers=config.loader.eval.num_workers, shuffle=False, drop_last=False,
            pin_memory=config.device.startswith("cuda"), collate_fn=ir_collate,
            generator=torch.Generator().manual_seed(audit.subset_seed),
        )
    logger.info("Pseudo-label IR validation: fixed_images=%d full_images=%d batch_size=%d seed=%d",
                len(indices), len(dataset), config.loader.eval.batch_size, audit.subset_seed)
    return FixedIRPseudoLabelEvaluator(
        fixed_loader=loader(Subset(dataset, indices)), full_loader=loader(dataset),
        reader=reader, device=torch.device(config.device),
        fixed_sample_ids=tuple(dataset.ir_paths[i].stem for i in indices),
    )


def build_training(
    config: TrainingConfig, *, logger: logging.Logger
) -> CurriculumDomainAdaptationTrainer:
    """Prepare models, data, optimizers and evaluators before handing off to training."""
    # Data assembly: create all source/target training and validation loaders
    # from the effective config, then expose their sizes in the existing log.
    device = torch.device(config.device)
    gan_translator = None
    if config.workflow == "gan_bridge":
        gan_translator = load_gan_translator(config.gan_bridge.checkpoint, device)
        logger.info(
            "GAN bridge: domains=rgb:saga:ir_gan:ir_real ratios=%s fractions=%s generator=%s",
            config.bridge_schedule.phase_ratios, config.bridge_schedule.phase_fractions,
            config.gan_bridge.checkpoint,
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

    # The factory freezes teachers; bootstrap places every model before binding
    # optimization state. The trainer receives these dependencies ready to run.
    student, rgb_teacher, ir_teacher = move_model_trio_to_device(
        *build_fcos_trio(config),
        device,
    )
    optimizer = build_optimizer(student, config)
    lr_scheduler = build_lr_scheduler(optimizer, config)

    # Evaluation assembly: pair the existing AP50 metric implementation with
    # domain loaders, evaluation cadence, and history tracking.
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
    data_flow = RouteDataFlow(
        rgb_train_loader, ir_train_loader, device=device,
        augmentor=StudentAugmentor(config), gan_translator=gan_translator,
    )
    scheduler = (
        BridgeDomainScheduler(config.bridge_schedule, config.max_iter)
        if config.workflow == "gan_bridge" else CurriculumScheduler(config)
    )
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        data_flow=data_flow,
        scheduler=scheduler,
        phase_evaluator=phase_evaluator,
        lr_scheduler=lr_scheduler,
        pseudo_label_evaluator=build_pseudo_label_evaluator(config, logger=logger),
    )

    # Return a ready-to-run object without starting training during assembly.
    return trainer

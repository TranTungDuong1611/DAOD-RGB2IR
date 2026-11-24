"""Command-line entrypoint for FLIR RGB baseline and D3T curriculum training."""

import argparse
import logging

import torch

from config import (
    AugConfig,
    BridgeScheduleConfig,
    CurriculumConfig,
    DataConfig,
    DataLoaderConfig,
    DistillConfig,
    EMAConfig,
    EvalLoaderConfig,
    FCOSModelConfig,
    GANBridgeConfig,
    IntermediateDomainConfig,
    OptimizerConfig,
    Phase,
    PseudoLabelEvalConfig,
    TrainLoaderConfig,
    TrainingConfig,
)
from data import FLIR_CLASSES, NUM_CLASSES
from models.fcos_adapter import ClassificationInitMode
from training.bootstrap import build_training
from training.observability import configure_training_logging


logger = logging.getLogger(__name__)


def build_training_config(args) -> TrainingConfig:
    """Map every training-relevant CLI argument into one effective config."""

    # Resolve iteration boundaries and weight identifiers before building the
    # effective configuration consumed by bootstrap and checkpoint metadata.
    total_iters = int(args.total_iters)
    if args.workflow == "gan_bridge" and any(
        value is not None for value in (args.phase1_end, args.phase2_end, args.phase3_end)
    ):
        raise ValueError("gan_bridge uses --bridge-phase-fractions, not legacy --phaseN-end flags")
    if args.phase3_end is not None and args.phase3_end > total_iters:
        raise ValueError("phase3_end must not exceed total_iters")
    # Collect phase boundaries and route proportions for the curriculum scheduler.
    curriculum = CurriculumConfig(
        phase1_end=args.phase1_end,
        phase2_end=args.phase2_end,
        phase3_end=args.phase3_end,
        phase2_rgb_sampling_ratio=args.phase2_rgb_ratio,
        phase3_rgb_sampling_ratio=args.phase3_rgb_ratio,
        sampling_mode=args.domain_sampling,
        phase4_rgb_sampling_ratio=args.phase4_rgb_ratio,
    )

    return TrainingConfig(
        # Preserve baseline detector construction, transform, and inference settings.
        model=FCOSModelConfig(
            num_classes=NUM_CLASSES,
            class_names=tuple(FLIR_CLASSES),
            weights=args.weights,
            classification_init_mode=ClassificationInitMode(
                args.classification_init
            ),
            pretrained_backbone=args.pretrained_backbone,
            trainable_backbone_layers=args.trainable_backbone_layers,
            min_sizes=tuple(args.min_sizes),
            max_size=args.max_size,
            center_sampling_radius=args.center_sampling_radius,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            topk_candidates=args.topk_candidates,
            detections_per_img=args.detections_per_img,
        ),
        # Teacher-specific KD selection schedules and shared distillation settings.
        distill=DistillConfig(
            hm_alpha=args.hm_alpha,
            hm_beta=args.hm_beta,
            un_regular_alpha=args.un_regular_alpha,
        ),
        curriculum=curriculum,
        bridge_schedule=BridgeScheduleConfig(
            phase_ratios=args.bridge_ratios,
            phase_fractions=tuple(args.bridge_phase_fractions),
        ),
        gan_bridge=GANBridgeConfig(
            checkpoint=args.gan_weights, source_sup_weight=args.bridge_source_sup_weight,
            saga_student_alphas=tuple(args.bridge_saga_student_alphas),
            saga_teacher_alphas=tuple(args.bridge_saga_teacher_alphas),
            saga_teachers=tuple(args.bridge_saga_teachers),
        ),
        optimizer=OptimizerConfig(
            base_lr=args.lr,
            warmup_iters=args.warmup_iters,
            warmup_factor=args.warmup_factor,
            milestones=tuple(args.lr_steps),
            gamma=args.lr_gamma,
        ),
        ema=EMAConfig(alpha=args.ema_alpha, start_steps=args.ema_start),
        intermediate_domain=IntermediateDomainConfig(
            mode=args.intermediate_domain,
            texture_strength=args.texture_strength,
            btf_radius=args.btf_radius,
            btf_iterations=args.btf_iterations,
        ),
        aug=AugConfig(
            hflip_prob=args.hflip_prob,
            blur_prob=args.blur_prob,
            brightness_prob=args.brightness_prob,
            contrast_prob=args.contrast_prob,
        ),
        # Dataset location and loader settings used by bootstrap's factories.
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
        # Run lifecycle, output, workflow, and monitoring cadence settings.
        total_iters=total_iters,
        output_dir=args.output_dir,
        device=args.device,
        workflow=args.workflow,
        teacher_mode=args.teacher_mode,
        log_interval=args.log_interval,
        eval_period=args.eval_every,
        pseudo_label_eval=PseudoLabelEvalConfig(
            enabled=args.eval_pseudo_labels,
            interval=args.pseudo_label_eval_interval,
            iou_threshold=args.pseudo_label_iou_threshold,
            sample_size=args.pseudo_label_eval_samples,
            subset_seed=args.pseudo_label_eval_seed,
        ),
    )


def main(args) -> None:
    # Configure the experiment's existing log destinations before constructing
    # the dependencies that emit setup and progress messages.
    config = build_training_config(args)
    log_path = configure_training_logging(config.output_dir)
    logger.info("Training log -> %s", log_path)
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

    logger.info("Intermediate domain: %s", config.intermediate_domain)
    if (
        config.workflow in {"curriculum", "gan_bridge"}
        and config.intermediate_domain.mode == "randommono_btf"
        and config.intermediate_domain.texture_strength
    ):
        from randommono_btf import require_btf
        require_btf()

    # Bootstrap wires dependencies; this entrypoint retains the sequence of
    # resume/initial evaluation, training, interruption handling, and final save.
    trainer = build_training(config, logger=logger)
    evaluator = trainer.evaluator

    # Resume saved progress or evaluate the initial model using the original
    # mutually exclusive paths and initial phase label.
    if args.resume:
        trainer.load_checkpoint(args.resume)
    else:
        logger.info("Running initial IR evaluation")
        trainer.evaluate(
            current_phase=Phase.PHASE1_RGB_WARMUP,
            trigger_reason="initial",
        )

    # An interrupted iteration can leave partial runtime state. Resume the last
    # completed checkpoint instead of publishing that state as a final model.
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted; resume from the last completed checkpoint")
    else:
        trainer.save_checkpoint("final_model.pth")
    finally:
        evaluator.print_history()


def _bridge_ratios(value):
    try:
        ratios = tuple(tuple(float(weight) for weight in row.split(":")) for row in value.split(","))
        BridgeScheduleConfig(phase_ratios=ratios)
        return ratios
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    defaults = TrainingConfig()
    # Declare launch options here; build_training_config translates them into
    # one effective configuration before any training dependency is assembled.
    parser = argparse.ArgumentParser(
        description="Curriculum D3T training for aligned FLIR RGB/IR data"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--intermediate-domain",
        choices=("randommono_btf", "saga"),
        default=defaults.intermediate_domain.mode,
    )
    parser.add_argument(
        "--texture-strength",
        type=float,
        default=defaults.intermediate_domain.texture_strength,
    )
    parser.add_argument(
        "--btf-radius",
        type=int,
        default=defaults.intermediate_domain.btf_radius,
    )
    parser.add_argument(
        "--btf-iterations",
        type=int,
        default=defaults.intermediate_domain.btf_iterations,
    )
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--total_iters", type=int, default=defaults.total_iters)
    parser.add_argument(
        "--phase1-end", "--phase1_end", dest="phase1_end", type=int, default=None
    )
    parser.add_argument(
        "--phase2-end", "--phase2_end", dest="phase2_end", type=int, default=None
    )
    parser.add_argument(
        "--phase3-end", "--phase3_end", dest="phase3_end", type=int, default=None
    )
    parser.add_argument("--batch_size", type=int, default=defaults.loader.train.batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=defaults.loader.eval.batch_size)
    parser.add_argument("--workers", type=int, default=defaults.loader.train.num_workers)
    parser.add_argument(
        "--lr",
        type=float,
        default=OptimizerConfig.base_lr,
        help="Actual LR; defaults to D3T's 0.016 scaled from batch 16",
    )
    parser.add_argument("--warmup-iters", type=int, default=OptimizerConfig.warmup_iters)
    parser.add_argument("--warmup-factor", type=float, default=defaults.optimizer.warmup_factor)
    parser.add_argument("--lr-steps", type=int, nargs="*", default=defaults.optimizer.milestones)
    parser.add_argument("--lr-gamma", type=float, default=defaults.optimizer.gamma)
    parser.add_argument(
        "--min-sizes",
        type=int,
        nargs="+",
        default=defaults.model.min_sizes,
        help="Training short-edge choices; evaluation uses the final value",
    )
    parser.add_argument("--max_size", type=int, default=defaults.model.max_size)
    parser.add_argument("--center_sampling_radius", type=float, default=defaults.model.center_sampling_radius)
    parser.add_argument("--score_thresh", type=float, default=defaults.model.score_thresh)
    parser.add_argument("--nms_thresh", type=float, default=defaults.model.nms_thresh)
    parser.add_argument("--topk_candidates", type=int, default=defaults.model.topk_candidates)
    parser.add_argument("--detections_per_img", type=int, default=defaults.model.detections_per_img)
    parser.add_argument("--trainable_backbone_layers", type=int, default=defaults.model.trainable_backbone_layers)
    parser.add_argument(
        "--weights",
        choices=("DEFAULT", "none"),
        default=defaults.model.weights,
        help="Torchvision FCOS COCO weights or none",
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
        default=defaults.model.classification_init_mode.value,
    )
    parser.add_argument(
        "--workflow",
        choices=("curriculum", "rgb_baseline", "gan_bridge"),
        default=defaults.workflow,
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("rgb", "ir", "two_teacher"),
        default=defaults.teacher_mode,
        help="Enable one teacher for staged smoke/ablation or both for curriculum",
    )
    parser.add_argument("--phase2-rgb-ratio", type=float, default=defaults.curriculum.phase2_rgb_sampling_ratio)
    parser.add_argument("--phase3-rgb-ratio", type=float, default=defaults.curriculum.phase3_rgb_sampling_ratio)
    parser.add_argument(
        "--domain-sampling", choices=("grouped", "interleaved"), default=defaults.curriculum.sampling_mode,
        help="Group existing curriculum routes in short cycles or spread them per iteration",
    )
    parser.add_argument("--gan-weights", default=defaults.gan_bridge.checkpoint, help="Pretrained RGB(3)->IR(1) CycleGAN G_A checkpoint")
    parser.add_argument("--bridge-ratios", type=_bridge_ratios, default=defaults.bridge_schedule.phase_ratios)
    parser.add_argument("--bridge-phase-fractions", type=float, nargs=5, default=defaults.bridge_schedule.phase_fractions)
    parser.add_argument("--bridge-source-sup-weight", type=float, default=defaults.gan_bridge.source_sup_weight)
    parser.add_argument("--bridge-saga-student-alphas", type=float, nargs=5, default=defaults.gan_bridge.saga_student_alphas)
    parser.add_argument("--bridge-saga-teacher-alphas", type=float, nargs=5, default=defaults.gan_bridge.saga_teacher_alphas)
    parser.add_argument("--bridge-saga-teachers", choices=("none", "rgb", "ir"), nargs=5,
                        default=defaults.gan_bridge.saga_teachers)
    parser.add_argument(
        "--phase4-rgb-ratio", type=float, default=defaults.curriculum.phase4_rgb_sampling_ratio,
        help="Fraction of phase-4 steps using supervised RGB; use 0 for IR-only",
    )
    parser.add_argument("--hm-alpha", type=float, default=defaults.distill.hm_alpha)
    parser.add_argument("--hm-beta", type=float, default=defaults.distill.hm_beta)
    parser.add_argument("--un-regular-alpha", type=float, default=defaults.distill.un_regular_alpha)
    parser.add_argument("--ema-alpha", type=float, default=defaults.ema.alpha)
    parser.add_argument(
        "--ema-start",
        type=int,
        default=EMAConfig.start_steps,
        help="EMA initialization step; defaults to the start of phase 2",
    )
    parser.add_argument("--hflip-prob", type=float, default=defaults.aug.hflip_prob)
    parser.add_argument("--blur-prob", type=float, default=defaults.aug.blur_prob)
    parser.add_argument("--brightness-prob", type=float, default=defaults.aug.brightness_prob)
    parser.add_argument("--contrast-prob", type=float, default=defaults.aug.contrast_prob)
    parser.add_argument("--eval_every", type=int, default=defaults.eval_period)
    # Audit labels are independent of supervised targets and student validation.
    parser.add_argument("--eval-pseudo-labels", action="store_true", default=defaults.pseudo_label_eval.enabled,
                        help="Compare selected KD rows with IR GT; never train on IR GT")
    parser.add_argument("--pseudo-label-eval-interval", type=int, default=defaults.pseudo_label_eval.interval,
                        help="Global iteration interval after EMA teachers are initialized")
    parser.add_argument("--pseudo-label-iou-threshold", type=float, default=defaults.pseudo_label_eval.iou_threshold)
    parser.add_argument("--pseudo-label-eval-samples", type=int, default=defaults.pseudo_label_eval.sample_size,
                        help="Maximum labeled IR validation images in each periodic teacher audit")
    parser.add_argument("--pseudo-label-eval-seed", type=int, default=defaults.pseudo_label_eval.subset_seed,
                        help="Local seed for reproducible fixed IR validation image selection")
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=defaults.device)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

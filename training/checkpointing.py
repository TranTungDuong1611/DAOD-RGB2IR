"""Save and restore trainer state, with model settings for standalone evaluation."""

import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def save_checkpoint(trainer, filename: str) -> None:
    """Persist runtime state and the model settings needed by COCO evaluation."""
    path = Path(trainer.config.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = trainer.config.model
    evaluator = trainer.evaluator
    payload = {
        "metadata": {"model": {
            **vars(model_config),
            "classification_init_mode": model_config.classification_init_mode.value,
        }},
        "student": trainer.student.state_dict(),
        "rgb_teacher": None if trainer.rgb_teacher is None else trainer.rgb_teacher.state_dict(),
        "ir_teacher": None if trainer.ir_teacher is None else trainer.ir_teacher.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "lr_scheduler": None if trainer.lr_scheduler is None else trainer.lr_scheduler.state_dict(),
        "global_step": trainer.global_step,
        "best_map": trainer.best_map,
        "best_rgb_map": trainer.best_rgb_map,
        "ema_initialized": trainer.ema_initialized,
        "evaluation": None if evaluator is None else evaluator.state_dict(),
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved -> %s", path)


def load_checkpoint(trainer, path: str) -> None:
    """Restore saved state directly; model loading retains PyTorch's strict checks."""
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    trainer.student.load_state_dict(checkpoint["student"])
    for name in ("rgb_teacher", "ir_teacher"):
        teacher = getattr(trainer, name)
        if teacher is not None:
            teacher.load_state_dict(checkpoint[name])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    if trainer.lr_scheduler is not None:
        trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    evaluator_state = checkpoint.get("evaluation")
    if evaluator_state is None:
        # Read the previous flat evaluation state once when resuming an older run.
        evaluator_state = {
            **(checkpoint.get("phase_evaluator") or {}),
            "last_pseudo_label_eval_step": checkpoint.get("last_pseudo_label_eval_step", 0),
            "last_full_pseudo_label_eval_step": checkpoint.get("last_full_pseudo_label_eval_step", 0),
            "pseudo_label_evaluator": checkpoint.get("pseudo_label_evaluator"),
        }
    else:
        evaluator_state = dict(evaluator_state)
    trainer.best_map = max(float(checkpoint.get("best_map", -1.0)),
                           float(evaluator_state.pop("best_ir_map", -1.0)))
    trainer.best_rgb_map = max(float(checkpoint.get("best_rgb_map", -1.0)),
                               float(evaluator_state.pop("best_rgb_map", -1.0)))
    if trainer.evaluator is not None:
        trainer.evaluator.load_state_dict(evaluator_state)
    trainer.global_step = int(checkpoint.get("global_step", 0))
    trainer.ema_initialized = bool(checkpoint.get("ema_initialized", False))
    logger.info("Resumed from step %s", trainer.global_step)


__all__ = ["save_checkpoint", "load_checkpoint"]

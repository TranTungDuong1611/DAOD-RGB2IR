"""Checkpoint metadata serialization and compatibility validation."""

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from config import TrainingConfig


CHECKPOINT_SCHEMA_VERSION = 1


def _serialize(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def serialize_effective_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert the effective dataclass configuration to checkpoint-safe data."""

    return _serialize(config)


def _bridge_signature(config, generator_identity):
    if not isinstance(generator_identity, dict) or not generator_identity.get("sha256"):
        raise ValueError("bridge checkpoint requires a GAN generator fingerprint")
    settings = _serialize(config.gan_bridge)
    settings.pop("checkpoint")  # Identical weights may live at a different path.
    return {
        "generator": _serialize(generator_identity),
        "schedule": _serialize(config.bridge_schedule),
        "settings": settings,
        "total_iters": config.max_iter,
        "ema": _serialize(config.ema),
        "loss": _serialize(config.loss),
        "distill": _serialize(config.distill),
        "model": _serialize(config.model),
        "augmentation": _serialize(config.aug),
        "optimizer": _serialize(config.optimizer),
    }


def build_checkpoint_metadata(
    config: TrainingConfig, *, generator_identity=None
) -> dict[str, Any]:
    """Build the compatibility metadata stored beside model and optimizer state."""

    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "num_classes": config.model.num_classes,
        "class_names": list(config.model.class_names),
        "classification_init_mode": config.model.classification_init_mode.value,
        "teacher_mode": config.teacher_mode,
        "config": serialize_effective_config(config),
        "workflow": config.workflow,
    }
    if config.workflow == "gan_bridge":
        metadata["bridge"] = _bridge_signature(config, generator_identity)
    return metadata


def validate_checkpoint_metadata(
    metadata: dict[str, Any],
    config: TrainingConfig,
    *,
    generator_identity=None,
) -> None:
    """Reject checkpoints whose schema or training setup is incompatible."""

    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema; expected "
            f"{CHECKPOINT_SCHEMA_VERSION}, got {metadata.get('schema_version')}"
        )
    if metadata.get("num_classes") != config.model.num_classes:
        raise ValueError("Checkpoint class count does not match the effective config")
    if tuple(metadata.get("class_names", ())) != tuple(config.model.class_names):
        raise ValueError("Checkpoint class order does not match the effective config")
    if metadata.get("classification_init_mode") != config.model.classification_init_mode.value:
        raise ValueError(
            "Checkpoint classification initialization mode does not match the effective config"
        )
    if metadata.get("teacher_mode") != config.teacher_mode:
        raise ValueError("Checkpoint teacher mode does not match the effective config")
    saved_workflow = metadata.get(
        "workflow", metadata.get("config", {}).get("workflow", "curriculum")
    )
    if "gan_bridge" in {saved_workflow, config.workflow}:
        if saved_workflow != config.workflow:
            raise ValueError("Checkpoint workflow does not match the GAN bridge workflow")
        if metadata.get("bridge") != _bridge_signature(config, generator_identity):
            raise ValueError(
                "Checkpoint bridge settings, schedule or GAN generator identity do not match"
            )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "build_checkpoint_metadata",
    "serialize_effective_config",
    "validate_checkpoint_metadata",
]

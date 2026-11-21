"""Checkpoint metadata serialization and compatibility checks."""

import unittest

import config as configs
from models.torchvision_fcos_adapter import ClassificationInitMode
from training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint_metadata,
    serialize_effective_config,
    validate_checkpoint_metadata,
)


def build_config(**kwargs):
    model = configs.FCOSModelConfig(
        weights=None,
        pretrained_backbone=False,
        classification_init_mode=ClassificationInitMode.COCO_TOWER,
    )
    return configs.TrainingConfig(model=model, device="cpu", **kwargs)


class CheckpointingTests(unittest.TestCase):
    def test_metadata_serializes_effective_config_and_rejects_incompatible_class_mode(self):
        config = build_config()
        metadata = build_checkpoint_metadata(config)

        self.assertEqual(metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(metadata["config"], serialize_effective_config(config))
        validate_checkpoint_metadata(metadata, config)

        incompatible = dict(metadata)
        incompatible["classification_init_mode"] = ClassificationInitMode.RANDOM_HEAD.value
        with self.assertRaisesRegex(ValueError, "classification"):
            validate_checkpoint_metadata(incompatible, config)

    def test_bridge_metadata_checks_generator_identity_but_ignores_checkpoint_path(self):
        identity = {"sha256": "fixture-hash", "architecture": "fixture"}
        config = build_config(
            gan_bridge=configs.GANBridgeConfig(checkpoint="first-path.pth"),
            workflow="gan_bridge",
        )
        metadata = build_checkpoint_metadata(config, generator_identity=identity)

        moved_config = build_config(
            gan_bridge=configs.GANBridgeConfig(checkpoint="moved-path.pth"),
            workflow="gan_bridge",
        )
        validate_checkpoint_metadata(metadata, moved_config, generator_identity=identity)

        changed_identity = {**identity, "sha256": "different-hash"}
        with self.assertRaisesRegex(ValueError, "bridge settings"):
            validate_checkpoint_metadata(
                metadata, moved_config, generator_identity=changed_identity
            )


if __name__ == "__main__":
    unittest.main()

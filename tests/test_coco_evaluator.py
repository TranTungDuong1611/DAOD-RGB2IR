import unittest
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch

from config import ClassificationInitMode, FCOSModelConfig, TrainingConfig
from evaluate_coco import _load_checkpoint_model, build_coco_inputs, summarize_coco_eval
from models.fcos_models import build_fcos_model
from test_fcos_adapter import TinyBackbone
from torchvision.models.detection import FCOS


class CocoEvaluatorTest(unittest.TestCase):
    def test_checkpoint_model_loads_minimal_and_legacy_metadata(self):
        config = TrainingConfig(model=FCOSModelConfig(
            weights=None, pretrained_backbone=False,
            classification_init_mode=ClassificationInitMode.RANDOM_HEAD,
            min_sizes=(64, 96), max_size=128, center_sampling_radius=1.5,
            nms_thresh=.45, topk_candidates=37,
        ), device="cpu")

        def tiny_model(effective_config):
            return build_fcos_model(effective_config, base_detector=FCOS(
                TinyBackbone(), num_classes=91, min_size=64, max_size=128,
            ))

        source = tiny_model(config)
        model_metadata = dict(vars(config.model), classification_init_mode="random_head")
        metadata_formats = (
            {"model": model_metadata},
            {
                "schema_version": 1, "num_classes": 3,
                "class_names": ["person", "car", "bicycle"],
                "classification_init_mode": "random_head", "teacher_mode": "two_teacher",
                "config": {"model": model_metadata},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for metadata in metadata_formats:
                with self.subTest(metadata=tuple(metadata)):
                    path = Path(directory) / "model.pth"
                    torch.save({"student": source.state_dict(), "metadata": metadata}, path)
                    with mock.patch("evaluate_coco.build_fcos_model", side_effect=tiny_model):
                        model, classes = _load_checkpoint_model(str(path), torch.device("cpu"), .001)
                    self.assertEqual(classes, ("person", "car", "bicycle"))
                    self.assertFalse(model.training)
                    effective = model.effective_config.model
                    self.assertEqual(effective.min_sizes, (64, 96))
                    self.assertEqual(effective.max_size, 128)
                    self.assertEqual(effective.center_sampling_radius, 1.5)
                    self.assertEqual(effective.nms_thresh, .45)
                    self.assertEqual(effective.topk_candidates, 37)
                    self.assertEqual(effective.score_thresh, .001)
                    self.assertIs(effective.classification_init_mode, ClassificationInitMode.RANDOM_HEAD)
                    for key, value in source.state_dict().items():
                        self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_build_coco_inputs_converts_zero_based_labels_and_xyxy_boxes(self):
        predictions = [{
            "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([2]),
        }]
        targets = [{
            "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
            "labels": torch.tensor([0]),
        }]

        dataset, detections = build_coco_inputs(
            predictions, targets, ("person", "car", "bicycle")
        )

        self.assertEqual(dataset["categories"][0], {"id": 1, "name": "person"})
        self.assertEqual(dataset["annotations"][0]["category_id"], 1)
        self.assertEqual(dataset["annotations"][0]["bbox"], [1.0, 2.0, 10.0, 20.0])
        self.assertEqual(dataset["annotations"][0]["area"], 200.0)
        self.assertEqual(detections[0]["category_id"], 3)
        self.assertEqual(detections[0]["bbox"], [10.0, 20.0, 30.0, 40.0])

    def test_summarize_coco_eval_reports_each_class(self):
        class FakeParams:
            iouThrs = np.array([0.5, 0.75])
            areaRngLbl = ["all"]
            maxDets = [1, 10, 100]

        class FakeEval:
            params = FakeParams()
            stats = np.arange(12, dtype=float) / 10.0
            precision = np.empty((2, 2, 2, 1, 3), dtype=float)
            precision[:, :, 0, 0, :] = np.array([0.2, 0.6])[:, None, None]
            precision[:, :, 1, 0, :] = np.array([0.4, 0.8])[:, None, None]
            eval = {"precision": precision}

        metrics = summarize_coco_eval(FakeEval(), ("person", "car"))

        self.assertAlmostEqual(metrics["AP"], 0.0)
        self.assertAlmostEqual(metrics["AP50/person"], 0.2)
        self.assertAlmostEqual(metrics["AP75/person"], 0.6)
        self.assertAlmostEqual(metrics["AP/person"], 0.4)
        self.assertAlmostEqual(metrics["AP/car"], 0.6)


if __name__ == "__main__":
    unittest.main()

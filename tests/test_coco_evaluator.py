import unittest

import numpy as np
import torch

from evaluate_coco import _collect, build_coco_inputs, summarize_coco_eval


class CocoEvaluatorTest(unittest.TestCase):
    def test_collect_forwards_the_evaluation_domain(self):
        class RecordingModel:
            def __init__(self):
                self.domains = []

            def __call__(self, images, sample_ids=None, domain="rgb"):
                self.domains.append(domain)
                return [
                    {
                        "boxes": torch.empty(0, 4),
                        "scores": torch.empty(0),
                        "labels": torch.empty(0, dtype=torch.long),
                    }
                    for _ in images
                ]

        model = RecordingModel()
        batch = (
            torch.zeros(1, 3, 8, 8),
            [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}],
            ("ir-1",),
        )

        _collect(model, [batch], torch.device("cpu"), domain="ir")

        self.assertEqual(model.domains, ["ir"])

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

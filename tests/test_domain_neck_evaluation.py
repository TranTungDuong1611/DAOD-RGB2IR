import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from config import FCOSModelConfig, TrainingConfig
from evaluate_coco import _collect, _load_checkpoint_model
from evaluator import DetectionEvaluator, PhaseEvaluator
from models.domain_residual_neck import SEBlock
from models.fcos_factory import build_checkpoint_metadata, build_fcos_d3t_model


class DomainEvaluationModel(nn.Module):
    supports_domain_routing = True

    def __init__(self):
        super().__init__()
        self.rgb_se, self.ir_se = SEBlock(1, 1), SEBlock(1, 1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            self.rgb_se.se[3].bias.fill_(math.log(3))
            self.ir_se.se[3].bias.fill_(-math.log(3))

    def forward(self, images, sample_ids=None, domain="rgb"):
        self.last_domain = domain
        attention = self.ir_se if domain == "ir" else self.rgb_se
        self.last_score = attention(torch.ones(1, 1, 2, 2)).mean().reshape(1)
        return [{"boxes": torch.tensor([[0., 0., 10., 10.]]),
                 "labels": torch.tensor([0]), "scores": self.last_score} for _ in images]


def loader_batch():
    return [(torch.ones(1, 3, 16, 16),
             [{"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([0])}],
             ("sample",))]


@pytest.mark.parametrize("domain, score", [("rgb", 0.75), ("ir", 0.25)])
def test_coco_collection_selects_matching_attention(domain, score):
    model = DomainEvaluationModel()
    predictions, targets = _collect(model, loader_batch(), torch.device("cpu"), domain=domain)
    assert predictions[0]["scores"].item() == pytest.approx(score)
    assert len(targets) == 1


@pytest.mark.parametrize("domain, score", [("RGB", 0.75), ("IR", 0.25)])
def test_phase_validation_selects_matching_attention(domain, score):
    model = DomainEvaluationModel()
    evaluator = PhaseEvaluator(DetectionEvaluator(1), loader_batch(), torch.device("cpu"))
    result = evaluator._run_eval_on_loader(model, loader_batch(), domain)
    assert model.last_domain == domain.lower()
    assert model.last_score.item() == pytest.approx(score)
    assert result["mAP@0.5"] == 1.0


def test_validation_does_not_hide_domain_aware_model_errors_by_retrying_rgb():
    class FailingModel(DomainEvaluationModel):
        def forward(self, images, sample_ids=None, domain="rgb"):
            if domain == "ir":
                raise TypeError("IR attention failed")
            return super().forward(images, sample_ids, domain)
    evaluator = PhaseEvaluator(DetectionEvaluator(1), loader_batch(), torch.device("cpu"))
    with pytest.raises(TypeError, match="IR attention failed"):
        evaluator._run_eval_on_loader(FailingModel(), loader_batch(), "IR")


@pytest.mark.parametrize("enabled", [False, True])
def test_coco_checkpoint_roundtrip_restores_architecture_and_raw_predictions(enabled):
    config = TrainingConfig(model=FCOSModelConfig(
        weights=None, pretrained_backbone=False, min_sizes=(64,), max_size=64,
        domain_neck_enabled=enabled, domain_neck_bottleneck_channels=8,
        domain_neck_se_reduction=8,
    ), device="cpu")
    model = build_fcos_d3t_model(config).eval()
    metadata = build_checkpoint_metadata(config)
    if not enabled:
        metadata.pop("domain_neck")
        for name in ("domain_neck_enabled", "domain_neck_bottleneck_channels", "domain_neck_se_reduction"):
            metadata["config"]["model"].pop(name)
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.pth")
        torch.save({"metadata": metadata, "student": model.state_dict()}, path)
        restored, names = _load_checkpoint_model(path, torch.device("cpu"), 0.05)
    assert names == config.model.class_names
    assert (restored.adapter.domain_neck is not None) == enabled
    image = torch.rand(3, 64, 64)
    with torch.no_grad():
        before = model.raw([image], domain="ir").predictions[0]
        after = restored.raw([image], domain="ir").predictions[0]
    torch.testing.assert_close(before.class_logits, after.class_logits)
    torch.testing.assert_close(before.quality_logits, after.quality_logits)
    torch.testing.assert_close(before.boxes, after.boxes)


def test_coco_loader_rejects_inconsistent_neck_metadata_before_tensor_loading():
    config = TrainingConfig(model=FCOSModelConfig(domain_neck_enabled=True))
    metadata = build_checkpoint_metadata(config)
    metadata["domain_neck"]["enabled"] = False
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "invalid.pth")
        torch.save({"metadata": metadata, "student": {}}, path)
        with pytest.raises(ValueError, match="neck"):
            _load_checkpoint_model(path, torch.device("cpu"), 0.05)

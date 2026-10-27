"""Monitoring cadence and persistence must not share student evaluation state."""

import json
import logging
from pathlib import Path
import sys
import tempfile
from unittest import mock

import pytest

from example_flir import build_training_config, parse_args
from scheduler import CurriculumScheduler


def test_cli_keeps_audit_opt_in_and_allows_separate_interval_and_label_directory():
    with mock.patch.object(sys, "argv", ["example_flir.py", "--data_root", "data"]):
        config = build_training_config(parse_args())
    assert not config.pseudo_label_eval.enabled
    with mock.patch.object(sys, "argv", [
        "example_flir.py", "--data_root", "data", "--eval-pseudo-labels",
        "--pseudo-label-eval-interval", "20", "--pseudo-label-iou-threshold", ".7",
        "--ir-annotation-dir", "audit-labels",
    ]):
        config = build_training_config(parse_args())
    assert config.pseudo_label_eval.enabled
    assert config.pseudo_label_eval.interval == 20
    assert config.pseudo_label_eval.iou_threshold == .7
    assert config.pseudo_label_eval.annotation_dir == "audit-labels"


def test_cadence_defers_to_ir_and_stays_on_global_500_iteration_boundaries():
    from config import PseudoLabelEvalConfig, TrainingConfig

    config = TrainingConfig(pseudo_label_eval=PseudoLabelEvalConfig(enabled=True))
    scheduler = CurriculumScheduler(config)
    due = scheduler.should_evaluate_pseudo_labels
    assert not due(498, "p4_ir_focus", last_eval_step=0)
    assert not due(499, "p3_rgb_flow", last_eval_step=0)
    assert not due(499, "p2_ir_flow", last_eval_step=0)  # Physical source is RGB.
    assert due(500, "p3_ir_flow", last_eval_step=0)
    assert not due(501, "p3_ir_flow", last_eval_step=501)
    assert due(999, "p4_ir_focus", last_eval_step=501)
    config.pseudo_label_eval.enabled = False
    assert not due(999, "p4_ir_focus", last_eval_step=501)


@pytest.mark.parametrize("kwargs", [{"interval": 0}, {"interval": -2},
                                    {"iou_threshold": -1.}, {"iou_threshold": float("nan")}])
def test_invalid_audit_config_is_rejected(kwargs):
    from config import PseudoLabelEvalConfig

    with pytest.raises(ValueError):
        PseudoLabelEvalConfig(**kwargs)


def test_observability_writes_jsonl_and_logs_even_outside_iteration_log_cadence():
    from training.observability import log_pseudo_label_evaluation

    record = {"global_step": 501, "teacher": "ir", "phase": "PHASE3_ADAPTATION",
              "metrics": {"valid_fraction": None, "missing_images": 1}}
    logger = logging.getLogger("test.pseudo.audit")
    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.object(logger, "info", wraps=logger.info) as info:
            log_pseudo_label_evaluation(record, output_dir=directory, logger=logger)
        assert info.call_count == 1
        saved = json.loads((Path(directory) / "logs" / "pseudo_label_eval.jsonl").read_text())
        assert saved == record

"""Logging setup and presentation, without changing training decisions."""

from datetime import datetime
import json
import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_HANDLER_MARKER = "_d3t_training_handler"


def configure_training_logging(output_dir: str) -> Path:
    """Log the current run to both the console and its output logs directory."""

    # Give this invocation its own timestamped destination under the configured
    # output directory, using the existing filename convention.
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"train_{timestamp}.log"

    # Replace only handlers owned by training so reconfiguration does not leave
    # duplicate outputs or open files from a previous invocation.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

    # Send the same records to console and UTF-8 file with the original format.
    # Other modules can use their normal loggers through root propagation.
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    )
    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARKER, True)
        root_logger.addHandler(handler)

    return log_path


def log_iteration(log: dict, global_step: int, *, logger: logging.Logger) -> None:
    """Present the current metrics with the original field selection and format."""
    # Keep the existing loss/KD field selection when constructing the detail
    # section; collecting a metric does not automatically add it to text output.
    components = []
    for key, value in sorted(log.items()):
        if (
            key != "total_loss"
            and isinstance(value, float)
            and ("loss" in key or "kd_" in key)
        ):
            components.append(f"{key}={value:.4f}")
    # Present the step context first, followed by optional timing and component
    # values. Formatting here does not modify the supplied training metrics.
    message = (
        f"[{int(log.get('global_step', global_step)):06d}] "
        f"Phase: {log.get('phase', 'N/A'):<22} | "
        f"Route: {log.get('step_type', 'N/A'):<18} | "
        f"Loss: {log.get('total_loss', 0.0):.4f} | "
        f"LR: {log.get('lr', 0.0):.6g}"
    )
    if isinstance(log.get("iter_time"), (int, float)):
        message += f" | Time: {log['iter_time']:.3f}s"
    if components:
        message += " | " + " | ".join(components)
    logger.info(message)


def log_pseudo_label_evaluation(record: dict, *, output_dir: str, logger: logging.Logger):
    """Persist audit records independently of regular iteration log cadence."""
    # Append supports resumed runs. Keep sample identity and selection settings
    # in JSONL while showing metric summaries through the existing log handlers.
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "pseudo_label_eval.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    logger.info(
        "[pseudo-label-eval] step=%s phase=%s teacher=%s metrics=%s",
        record["global_step"], record["phase"], record["teacher"],
        json.dumps(record["metrics"], allow_nan=False, sort_keys=True),
    )


__all__ = ["configure_training_logging", "log_iteration", "log_pseudo_label_evaluation"]

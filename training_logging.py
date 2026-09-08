"""Logging setup for a single training process."""

from datetime import datetime
import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_HANDLER_MARKER = "_d3t_training_handler"


def configure_training_logging(output_dir: str) -> Path:
    """Log the current run to both the console and its output logs directory."""

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"train_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()

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


__all__ = ["configure_training_logging"]

"""Build the standalone Kaggle notebook from this checkout's Python sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "kaggle_flir_object_color_consistency.ipynb"
EXCLUDED_DIRS = {".git", "__pycache__", "outputs", "tests", "tools"}


def _source_files() -> list[Path]:
    files = [
        path
        for path in ROOT.rglob("*.py")
        if path.name != Path(__file__).name
        and not any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts)
    ]
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _cell(cell_type: str, source: str, cell_id: str):
    if cell_type == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    sources = _source_files()
    relative_paths = [path.relative_to(ROOT).as_posix() for path in sources]
    cells = [
        _cell(
            "markdown",
            """# FLIR object color consistency training

Upload this notebook to Kaggle and run all cells. It embeds the training source, uses the attached `align` dataset, trains for 10,000 iterations, writes checkpoints under `/kaggle/working/flir-object-color-consistency-output`, and evaluates the final student in the last cell.

The trial adds normalized object-level L1 between the RGB EMA teacher and SAGA student at P3 with weight `0.1`. Set `COLOR_CONSISTENCY_WEIGHT` to `0.0` in the configuration cell to disable it.

Select a Kaggle GPU accelerator. The existing trainer uses Torchvision's default COCO weights; enable Kaggle Internet if those weights are not already cached.
""",
            "intro",
        ),
        _cell(
            "code",
            """from pathlib import Path
import sys

PROJECT_ROOT = Path("/kaggle/working/flir_object_color_consistency")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
MODULE_FILES = """ + repr(relative_paths) + """
for relative_path in MODULE_FILES:
    (PROJECT_ROOT / relative_path).parent.mkdir(parents=True, exist_ok=True)
%cd /kaggle/working/flir_object_color_consistency
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
""",
            "prepare-source-tree",
        ),
    ]

    for index, (path, relative_path) in enumerate(zip(sources, relative_paths)):
        text = path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"
        source = f"%%writefile {relative_path}\n{text}"
        slug = relative_path.replace("/", "-").replace(".", "-")
        cells.append(_cell("code", source, f"source-{index:02d}-{slug}"))

    cells.extend(
        [
            _cell(
                "markdown",
                "## Configure and build the existing trainer\n\nThe dataset root follows the repository's `align/` layout. Defaults for batch size, model, curriculum, and evaluation come from `example_flir.py`.\n",
                "configure-heading",
            ),
            _cell(
                "code",
                '''import sys
from pathlib import Path

from config import Phase
from example_flir import build_training_config, parse_args
from training.bootstrap import build_training
from training.observability import configure_training_logging

DATA_ROOT = Path("/kaggle/input/datasets/tungduong123qwe/flir-dataset/align")
OUTPUT_DIR = Path("/kaggle/working/flir-object-color-consistency-output")
TOTAL_ITERS = 10_000
COLOR_CONSISTENCY_WEIGHT = 0.1

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Kaggle dataset directory does not exist: {DATA_ROOT}")

sys.argv = [
    "example_flir.py",
    "--data_root", str(DATA_ROOT),
    "--output_dir", str(OUTPUT_DIR),
    "--total_iters", str(TOTAL_ITERS),
    "--color-consistency-weight", str(COLOR_CONSISTENCY_WEIGHT),
]
args = parse_args()
config = build_training_config(args)
configure_training_logging(str(OUTPUT_DIR))
trainer = build_training(config)

print(f"Training iterations: {config.total_iters}")
print(f"Color consistency weight: {config.loss.color_consistency_weight}")
print(f"Dataset: {config.data.root}")
print(f"Output: {config.output_dir}")
''',
                "configure-training",
            ),
            _cell(
                "markdown",
                "## Initial evaluation and training\n\nThis follows the existing entrypoint's initial evaluation and training loop. Periodic evaluation and 5,000-step checkpoints remain enabled.\n",
                "train-heading",
            ),
            _cell(
                "code",
                '''trainer.phase_evaluator.evaluate(
    trainer.student,
    global_step=trainer.global_step,
    current_phase=Phase.PHASE1_RGB_WARMUP,
    trigger_reason="initial",
)
''',
                "initial-evaluation",
            ),
            _cell(
                "code",
                '''try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted; saving the current checkpoint.")
finally:
    trainer.save_checkpoint("final_model.pth")
''',
                "train-10000-iterations",
            ),
            _cell(
                "markdown",
                "## Final evaluation\n\nThe final cell evaluates the trained student with the existing IR validation evaluator, then saves both the full resumable checkpoint and student weights in Kaggle output.\n",
                "final-evaluation-heading",
            ),
            _cell(
                "code",
                '''import torch

final_results = trainer.phase_evaluator.evaluate(
    trainer.student,
    global_step=trainer.global_step,
    current_phase=config.get_phase(trainer.global_step),
    trigger_reason="final",
)
trainer.save_checkpoint("final_model.pth")
student_weights_path = OUTPUT_DIR / "student_weights.pth"
torch.save(trainer.student.state_dict(), student_weights_path)

print(final_results)
print(f"Full checkpoint: {OUTPUT_DIR / 'final_model.pth'}")
print(f"Student weights: {student_weights_path}")
''',
                "final-evaluation-and-save",
            ),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")

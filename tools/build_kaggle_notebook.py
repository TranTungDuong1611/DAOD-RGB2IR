"""Build the standalone Kaggle experiment from canonical Python modules.

Usage: python tools/build_kaggle_notebook.py
No Jupyter authoring dependency is needed. The resulting notebook contains the
source and imports it in memory, never cloning a repo or launching project .py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pprint
import textwrap


ROOT_MODULES = (
    "config", "randommono_btf", "saga", "evaluator", "optimization",
    "scheduler", "trainer", "example_flir", "evaluate_coco", "notebook_support",
)
PACKAGES = ("configs", "data", "models", "loss", "training", "utils")

DEPENDENCY_SOURCE = r'''
# Keep the Kaggle CUDA torch/torchvision stack. Install only missing optional
# packages into a private directory, never another OpenCV alongside global cv2.
import importlib
import os
from pathlib import Path
import subprocess
import sys

DEPENDENCY_ROOT = Path("/kaggle/working/.flir_runtime_deps")
if not Path("/kaggle/working").is_dir():
    DEPENDENCY_ROOT = Path.cwd() / ".flir_runtime_deps"
if DEPENDENCY_ROOT.is_dir():
    sys.path.insert(0, str(DEPENDENCY_ROOT))

def ensure_notebook_dependencies():
    probes = {
        "opencv-contrib-python-headless==4.11.0.86":
            "import cv2; assert hasattr(getattr(cv2,'ximgproc',None),'bilateralTextureFilter')",
        "pycocotools==2.0.11": "from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval",
    }
    missing = []
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(DEPENDENCY_ROOT), env.get("PYTHONPATH", "")])
    for package, probe in probes.items():
        run = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
        if run.returncode:
            missing.append(package)
    if missing:
        if any(name == "cv2" or name.startswith("cv2.") for name in sys.modules) and any(p.startswith("opencv") for p in missing):
            raise RuntimeError("An incompatible cv2 is already loaded. Restart the kernel and Run All from the first cell.")
        DEPENDENCY_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-deps", "--only-binary=:all:", "--upgrade", "--target", str(DEPENDENCY_ROOT), *missing,
            ])
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Optional dependencies could not be installed. Enable Kaggle Internet or provide the required wheels in this private dependency directory, then restart and Run All.") from error
        if str(DEPENDENCY_ROOT) not in sys.path:
            sys.path.insert(0, str(DEPENDENCY_ROOT))
        importlib.invalidate_caches()
    import cv2
    from pycocotools.coco import COCO
    if not hasattr(getattr(cv2, "ximgproc", None), "bilateralTextureFilter"):
        raise RuntimeError("The active OpenCV still has no BTF; restart the kernel after dependency installation.")
    print("Dependencies ready. OpenCV:", cv2.__version__, "| torch/torchvision were not replaced.")

ensure_notebook_dependencies()
'''

LOADER_SOURCE = r'''
# Register the embedded source as real Python modules without project files.
import importlib.abc
import importlib.util
import linecache
import sys

class EmbeddedProjectFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    _flir_embedded_finder = True

    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        item = self.sources.get(fullname)
        if item is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=item["is_package"])

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return self.sources[fullname]["source"]

    def get_filename(self, fullname):
        item = self.sources[fullname]
        path = item["path"] or fullname.replace(".", "/") + "/__init__.py"
        return "/__flir_embedded__/" + path

    def exec_module(self, module):
        source = self.get_source(module.__name__)
        filename = self.get_filename(module.__name__)
        module.__file__ = filename
        module.__embedded_flir__ = True
        if self.sources[module.__name__]["is_package"]:
            module.__path__ = [filename.rsplit("/", 1)[0]]
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, "exec"), module.__dict__)


def install_embedded_project(sources):
    sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, "_flir_embedded_finder", False)]
    for name in sorted(sources, key=len, reverse=True):
        sys.modules.pop(name, None)
    finder = EmbeddedProjectFinder(sources)
    sys.meta_path.insert(0, finder)
    return finder

missing_modules = sorted(set(EXPECTED_SOURCE_MANIFEST) - set(EMBEDDED_SOURCES))
if missing_modules:
    raise RuntimeError("Run all module cells before this loader cell. Missing: " + ", ".join(missing_modules))
install_embedded_project(EMBEDDED_SOURCES)
SOURCE_MANIFEST = {name: {key: value for key, value in item.items() if key != "source"}
                   for name, item in EMBEDDED_SOURCES.items()}
print("Loaded", len(EMBEDDED_SOURCES), "embedded project modules; no checkout is required.")
'''

SETTINGS = {
    "data_root": "/kaggle/input/datasets/tungduong123qwe/flir-dataset/align",
    "output_dir": "/kaggle/working/output_randommono_btf",
    "total_iters": 10000,
    "batch_size": 4,
    "eval_batch_size": 4,
    "workers": 0,
    "device": "cuda",
    "weights": "DEFAULT",
    "workflow": "curriculum",
    "gan_weights": None,
    "teacher_mode": "two_teacher",
    "intermediate_domain": "randommono_btf",
    "texture_strength": 0.25,
    "btf_radius": 2,
    "btf_iterations": 1,
    "color_consistency_weight": 0.1,
    "color_consistency_levels": ("p3", "p4", "p5"),
    "color_consistency_level_weights": (1.0, 1.0, 1.0),
    "eval_every": 500,
    "log_interval": 50,
    "eval_pseudo_labels": False,
    "resume": None,
    "eval_checkpoint": None,
    "seed": 42,
}


def collect_sources(project_root):
    root = Path(project_root)
    sources = {}

    def add(name, path, is_package):
        source = path.read_text(encoding="utf-8") if path is not None else ""
        sources[name] = {
            "path": path.relative_to(root).as_posix() if path is not None else None,
            "is_package": is_package,
            "source": source,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        }

    for name in ROOT_MODULES:
        add(name, root / f"{name}.py", False)
    for package in PACKAGES:
        folder = root / package
        for directory in sorted([folder, *(p for p in folder.rglob("*") if p.is_dir() and "__pycache__" not in p.parts)]):
            name = ".".join(directory.relative_to(root).parts)
            init = directory / "__init__.py"
            add(name, init if init.is_file() else None, True)
        for path in sorted(folder.rglob("*.py")):
            if path.name != "__init__.py" and "__pycache__" not in path.parts:
                add(".".join(path.relative_to(root).with_suffix("").parts), path, False)
    return dict(sorted(sources.items()))


def _cell(source, role, *, name="", markdown=False):
    # Embedded source must remain byte-for-byte identical, including blank-line whitespace.
    if role != "source":
        source = textwrap.dedent(source).lstrip("\n")
    cell = {
        "cell_type": "markdown" if markdown else "code",
        "id": hashlib.sha256((role + ":" + name).encode()).hexdigest()[:12],
        "metadata": {"flir_role": role},
        "source": source.splitlines(keepends=True),
    }
    if not markdown:
        cell.update(execution_count=None, outputs=[])
    return cell


MODULE_SETUP_SOURCE = r'''
# Register one notebook cell as one importable Python module.
# Module bodies stay visible/editable Python; no quoted source dictionaries.
import hashlib
from IPython.core.magic import register_cell_magic

EMBEDDED_SOURCES = {
    name: dict(item, source="") for name, item in EXPECTED_SOURCE_MANIFEST.items()
    if item["path"] is None
}

@register_cell_magic
def flir_module(line, cell):
    name = line.strip()
    if name not in EXPECTED_SOURCE_MANIFEST:
        raise ValueError(f"Unknown FLIR module: {name}")
    item = EXPECTED_SOURCE_MANIFEST[name]
    # Preserve a canonical missing final newline when possible. IPython also
    # normalizes blank-line whitespace; record the source actually received.
    if cell.endswith("\n") and hashlib.sha256(cell[:-1].encode()).hexdigest() == item["sha256"]:
        cell = cell[:-1]
    filename = "/__flir_embedded__/" + (item["path"] or name.replace(".", "/") + "/__init__.py")
    compile(cell, filename, "exec")
    EMBEDDED_SOURCES[name] = dict(item, source=cell, sha256=hashlib.sha256(cell.encode()).hexdigest())

print("Module cells are ready. Run every %%flir_module cell, then the loader cell.")
'''


def build_notebook(project_root, output_path=None):
    sources = collect_sources(project_root)
    manifest = {name: {k: v for k, v in item.items() if k != "source"} for name, item in sources.items()}
    cells = [
        _cell('''
        # FLIR: RandomMono + BTF / 10k iterations

        Attach the FLIR Kaggle dataset, select **GPU** and enable **Internet** for
        uncached dependencies/pretrained FCOS weights, then **Run All** in a fresh
        kernel. Only this notebook is needed; project source is embedded below.

        Base: `origin/feat/randommono-btf-kaggle` at `feff39f`.
        The experiment uses RandomMono/BTF source views. Existing detection/KD losses,
        teacher EMA, phase-four RGB supervision and optional pseudo-label audits
        are retained. Object consistency adds normalized feature L1 on P3-P5
        against the pure RGB EMA reference, averaged across the three levels.

        Edit the **Experiment settings** cell for batch size, paths, resume, or
        ablations. Default training is 10,000 iterations. Checkpoints, training
        logs, configuration, source manifest, preview and final COCO metrics are
        saved under `/kaggle/working/output_randommono_btf`.
        ''', "intro", markdown=True),
        _cell(DEPENDENCY_SOURCE, "dependencies"),
        _cell('''
        ## Embedded canonical source
        Each `%%flir_module <name>` code cell below contains visible Python:
        imports, classes and functions from the matching CLI module. You can edit
        this code directly. The first line preserves its module namespace and
        imports. Run every module cell and the loader before training. After a
        code edit, run the edited module cell, then the loader and subsequent
        cells, or use Run All.
        No project .py files or git clone are needed.
        ''', "source_intro", markdown=True),
    ]
    cells.append(_cell(
        "EXPECTED_SOURCE_MANIFEST = " + pprint.pformat(manifest, sort_dicts=False, width=100)
        + "\n\n" + textwrap.dedent(MODULE_SETUP_SOURCE).lstrip("\n"), "module_setup"
    ))
    for name, item in sources.items():
        if item["path"] is None:
            continue  # Empty namespace packages are registered in setup.
        cell = _cell(f"%%flir_module {name}\n" + item["source"], "source", name=name)
        cell["metadata"]["flir_module"] = name
        cells.append(cell)
    cells.extend([
        _cell(LOADER_SOURCE, "loader"),
        _cell('''
        ## Experiment settings
        The original CLI defaults remain unless overridden here. `workers=0`
        avoids multiprocessing import issues with in-memory notebook modules.
        Set `intermediate_domain="saga"` for legacy SAGA, or
        `texture_strength=0.0` for a RandomMono-only ablation.
        `color_consistency_weight=0.0` disables object feature consistency.
        Default levels are P3, P4, P5 with equal weights. Loss activates after
        teacher initialization on labeled appearance-shifted source views;
        pure RGB, GAN-IR, real IR and RGB baseline steps do not activate it.
        `resume` resumes training; `eval_checkpoint` selects a checkpoint for the
        final eval cell and otherwise defaults to best, falling back to final.
        ''', "settings_intro", markdown=True),
        _cell("SETTINGS = " + pprint.pformat(SETTINGS, sort_dicts=False, width=95) + "\n", "settings"),
        _cell('''
        import example_flir
        import notebook_support

        train_args = notebook_support.make_training_args(SETTINGS)
        effective_config = notebook_support.preflight(train_args, SOURCE_MANIFEST)
        preview_path = notebook_support.save_bridge_preview(train_args)
        from IPython.display import display, Image as DisplayImage
        display(DisplayImage(filename=str(preview_path)))
        ''', "preflight"),
        _cell('''
        # Same entrypoint as `python example_flir.py ...`, called directly.
        # Re-seed after the preview so visualization cannot alter training RNG.
        notebook_support.seed_everything(SETTINGS["seed"])
        example_flir.main(train_args)
        notebook_support.release_training_memory()
        print("Training completed. Checkpoints are in:", train_args.output_dir)
        ''', "train"),
        _cell('''
        ## Final IR COCO evaluation
        The last cell loads the saved student through the original evaluator.
        It reports AP/AP50/AP75, size metrics and per-class metrics, and writes a
        JSON result. You can rerun this cell with another `eval_checkpoint`.
        ''', "eval_intro", markdown=True),
        _cell('''
        import json
        from pathlib import Path
        import evaluate_coco

        notebook_support.release_training_memory()
        eval_checkpoint = notebook_support.select_checkpoint(
            SETTINGS["output_dir"], SETTINGS.get("eval_checkpoint")
        )
        metrics_path = Path(SETTINGS["output_dir"]) / f"coco_ir_{eval_checkpoint.stem}.json"
        eval_args = evaluate_coco.parse_args([
            "--checkpoint", str(eval_checkpoint), "--data-root", str(SETTINGS["data_root"]),
            "--domain", "ir", "--batch-size", str(SETTINGS["eval_batch_size"]),
            "--workers", str(SETTINGS["workers"]), "--device", SETTINGS["device"],
            "--output", str(metrics_path),
        ])
        coco_metrics = evaluate_coco.evaluate(eval_args)
        print("Evaluated checkpoint:", eval_checkpoint)
        print("Metrics saved:", metrics_path)
        print(json.dumps(coco_metrics, indent=2))
        ''', "eval"),
    ])
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "flir_source_manifest": manifest,
            "flir_base_ref": "origin/feat/randommono-btf-kaggle",
            "flir_base_commit": "feff39f6d2019be0903a9a36834a80b3f4ce117a",
        },
        "cells": cells,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return notebook


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "notebooks/flir_randommono_btf_10k.ipynb")
    args = parser.parse_args()
    notebook = build_notebook(root, args.output)
    print(f"Wrote {args.output} ({len(notebook['cells'])} cells, {len(notebook['metadata']['flir_source_manifest'])} modules)")


if __name__ == "__main__":
    main()

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.build_kaggle_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def cell_source(cell):
    return "".join(cell["source"])


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = build_notebook(ROOT)

    def test_notebook_is_deterministic_and_all_code_cells_compile(self):
        self.assertEqual(self.notebook, build_notebook(ROOT))
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 5)
        ids = []
        for cell in self.notebook["cells"]:
            ids.append(cell["id"])
            if cell["cell_type"] == "code":
                compile(cell_source(cell), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
        self.assertEqual(len(ids), len(set(ids)))

    def test_source_cells_match_canonical_files_and_cover_all_runtime_modules(self):
        manifests = self.notebook["metadata"]["flir_source_manifest"]
        self.assertIn("example_flir", manifests)
        self.assertIn("evaluate_coco", manifests)
        self.assertIn("intermediate_domain", manifests)
        for item in manifests.values():
            if item["path"] is not None:
                data = (ROOT / item["path"]).read_text(encoding="utf-8")
                self.assertEqual(hashlib.sha256(data.encode()).hexdigest(), item["sha256"])
        ns = {}
        for cell in self.notebook["cells"]:
            if cell["metadata"].get("flir_role") == "source":
                exec(cell_source(cell), ns)
        self.assertEqual(set(ns["EMBEDDED_SOURCES"]), set(manifests))
        for name, item in ns["EMBEDDED_SOURCES"].items():
            self.assertEqual(hashlib.sha256(item["source"].encode()).hexdigest(), manifests[name]["sha256"])

    def test_kaggle_defaults_and_final_cell(self):
        settings_cell = next(c for c in self.notebook["cells"] if c["metadata"].get("flir_role") == "settings")
        ns = {}
        exec(cell_source(settings_cell), ns)
        settings = ns["SETTINGS"]
        self.assertEqual(settings["data_root"], "/kaggle/input/datasets/tungduong123qwe/flir-dataset/align")
        self.assertEqual(settings["output_dir"], "/kaggle/working/output_randommono_btf")
        self.assertEqual(settings["total_iters"], 10000)
        self.assertEqual(settings["workers"], 0)
        self.assertEqual(settings["intermediate_domain"], "randommono_btf")
        last = self.notebook["cells"][-1]
        self.assertEqual(last["cell_type"], "code")
        self.assertEqual(last["metadata"]["flir_role"], "eval")
        self.assertIn("evaluate_coco.evaluate", cell_source(last))
        training = next(c for c in self.notebook["cells"] if c["metadata"].get("flir_role") == "train")
        self.assertIn("example_flir.main", cell_source(training))
        self.assertNotIn("subprocess", cell_source(training))

    def test_embedded_modules_import_without_checkout_and_reload(self):
        code = "\n".join(cell_source(c) for c in self.notebook["cells"] if c["metadata"].get("flir_role") in {"source", "loader"})
        code += "\nimport example_flir, evaluate_coco, notebook_support, config, sys\n"
        code += "assert sys.modules['config'].__embedded_flir__\n"
        code += "a=example_flir.parse_args(['--data_root','unused','--weights','none'])\n"
        code += "assert example_flir.build_training_config(a).intermediate_domain.mode=='randommono_btf'\n"
        code += "assert evaluate_coco.parse_args(['--checkpoint','x','--data-root','unused']).domain=='ir'\n"
        code += "old=config; install_embedded_project(EMBEDDED_SOURCES); import config; assert config is not old\n"
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            # Retain isolated third-party test deps, but never the project checkout.
            env["PYTHONPATH"] = os.pathsep.join(x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x and Path(x).resolve() != ROOT)
            run = subprocess.run([sys.executable, "-X", "utf8", "-B", "-"], input=code, cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_checked_in_notebook_is_current(self):
        path = ROOT / "notebooks/flir_randommono_btf_10k.ipynb"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.notebook)

    def test_checkpoint_selection_and_dataset_error(self):
        from notebook_support import select_checkpoint, validate_dataset
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                select_checkpoint(tmp)
            final = Path(tmp) / "final_model.pth"
            final.touch()
            self.assertEqual(select_checkpoint(tmp), final)
            best = Path(tmp) / "best_model.pth"
            best.touch()
            self.assertEqual(select_checkpoint(tmp), best)
            self.assertEqual(select_checkpoint(tmp, str(final)), final)
            with self.assertRaises(FileNotFoundError):
                select_checkpoint(tmp, str(Path(tmp) / "absent.pth"))
            with self.assertRaisesRegex(FileNotFoundError, "JPEGImages"):
                validate_dataset(tmp, 4)

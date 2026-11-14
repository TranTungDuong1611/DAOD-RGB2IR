import ast
import hashlib
import importlib.util
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
                source = cell_source(cell)
                if cell["metadata"].get("flir_role") == "source":
                    header, source = source.split("\n", 1)
                    self.assertEqual(header, "%%flir_module " + cell["metadata"]["flir_module"])
                compile(source, cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
        self.assertEqual(len(ids), len(set(ids)))

    def test_source_cells_match_canonical_files_and_cover_all_runtime_modules(self):
        manifests = self.notebook["metadata"]["flir_source_manifest"]
        for name in ("example_flir", "evaluate_coco", "intermediate_domain"):
            self.assertIn(name, manifests)
        cells = {c["metadata"]["flir_module"]: c for c in self.notebook["cells"]
                 if c["metadata"].get("flir_role") == "source"}
        self.assertEqual(set(cells), {name for name, item in manifests.items() if item["path"] is not None})
        for name, item in manifests.items():
            if item["path"] is None:
                continue
            canonical = (ROOT / item["path"]).read_text(encoding="utf-8")
            body = cell_source(cells[name]).split("\n", 1)[1]
            self.assertEqual(body, canonical, name)
            self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), item["sha256"])

    def test_module_code_is_visible_python_not_a_quoted_source_value(self):
        cell = next(c for c in self.notebook["cells"] if c["metadata"].get("flir_module") == "config")
        header, body = cell_source(cell).split("\n", 1)
        self.assertEqual(header, "%%flir_module config")
        self.assertNotIn("EMBEDDED_SOURCES[", body)
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == "TrainingConfig"
                            for node in ast.parse(body).body))

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

    @unittest.skipUnless(importlib.util.find_spec("IPython"), "requires IPython, as provided by Kaggle/Jupyter")
    def test_visible_module_cells_run_in_ipython_without_checkout_and_accept_edits(self):
        cells = [(c["metadata"].get("flir_role"), cell_source(c)) for c in self.notebook["cells"]
                 if c["metadata"].get("flir_role") in {"module_setup", "source", "loader"}]
        code = "CELLS = " + repr(cells) + "\n" + r'''
import hashlib
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
for role, source in CELLS:
    result = shell.run_cell(source, silent=True)
    if result.error_before_exec or result.error_in_exec:
        raise result.error_before_exec or result.error_in_exec
ns = shell.user_ns
exec("import example_flir, evaluate_coco, config; assert config.__embedded_flir__", ns)
exec("a=example_flir.parse_args(['--data_root','unused','--weights','none']); assert example_flir.build_training_config(a).intermediate_domain.mode=='randommono_btf'", ns)
exec("assert evaluate_coco.parse_args(['--checkpoint','x','--data-root','unused']).domain=='ir'", ns)
for name, expected in ns['EXPECTED_SOURCE_MANIFEST'].items():
    actual = ns['EMBEDDED_SOURCES'][name]
    assert actual['sha256'] == hashlib.sha256(actual['source'].encode()).hexdigest(), name
    canonical = next((source.split('\n', 1)[1] for role, source in CELLS
                      if role == 'source' and source.startswith('%%flir_module ' + name + '\n')), '')
    # IPython normalizes whitespace-only lines, as it does for normal code cells.
    assert [line for line in actual['source'].splitlines() if line.strip()] == [line for line in canonical.splitlines() if line.strip()], name
old_config = ns['config']
source = ns['EMBEDDED_SOURCES']['config']['source']
edited = source.replace('texture_strength: float = 0.25', 'texture_strength: float = 0.4')
assert edited != source
shell.run_cell_magic('flir_module', 'config', edited)
exec(next(source for role, source in CELLS if role == 'loader'), ns)
exec('import config; assert config.IntermediateDomainConfig().texture_strength==0.4', ns)
assert ns['config'] is not old_config
assert ns['SOURCE_MANIFEST']['config']['sha256'] == hashlib.sha256(edited.encode()).hexdigest()
try:
    shell.run_cell_magic('flir_module', 'unknown', 'x = 1')
except ValueError:
    pass
else:
    raise AssertionError('unknown module accepted')
registered = ns['EMBEDDED_SOURCES']['config']['source']
try:
    shell.run_cell_magic('flir_module', 'config', 'def invalid(')
except SyntaxError:
    pass
else:
    raise AssertionError('invalid Python accepted')
assert ns['EMBEDDED_SOURCES']['config']['source'] == registered
'''
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(x for x in env.get("PYTHONPATH", "").split(os.pathsep)
                                               if x and Path(x).resolve() != ROOT)
            run = subprocess.run([sys.executable, "-X", "utf8", "-B", "-"], input=code,
                                 cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
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
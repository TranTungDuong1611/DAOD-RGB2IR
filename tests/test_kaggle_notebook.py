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
        for name in ("example_flir", "evaluate_coco", "randommono_btf"):
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
        cell = next(c for c in self.notebook["cells"] if c["metadata"].get("flir_module") == "configs.training")
        header, body = cell_source(cell).split("\n", 1)
        self.assertEqual(header, "%%flir_module configs.training")
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
source = ns['EMBEDDED_SOURCES']['configs.data']['source']
edited = source.replace('texture_strength: float = 0.25', 'texture_strength: float = 0.4')
assert edited != source
shell.run_cell_magic('flir_module', 'configs.data', edited)
exec(next(source for role, source in CELLS if role == 'loader'), ns)
exec('import config; assert config.IntermediateDomainConfig().texture_strength==0.4', ns)
assert ns['config'] is not old_config
assert ns['SOURCE_MANIFEST']['configs.data']['sha256'] == hashlib.sha256(edited.encode()).hexdigest()
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

    def test_validate_dataset_uses_disjoint_rgb_ir_split(self):
        from PIL import Image
        from notebook_support import validate_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "JPEGImages").mkdir()
            (root / "Annotations").mkdir()
            train_stems = [f"FLIR_{i:05d}_PreviewData" for i in range(6)]
            val_stems = ["FLIR_00010_PreviewData"]
            (root / "align_train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
            (root / "align_validation.txt").write_text("\n".join(val_stems) + "\n", encoding="utf-8")
            for stem in train_stems + val_stems:
                rgb_name = stem.replace("_PreviewData", "") + "_RGB.jpg"
                Image.new("RGB", (8, 8)).save(root / "JPEGImages" / rgb_name)
                Image.new("L", (8, 8)).save(root / "JPEGImages" / f"{stem}.jpeg")
                (root / "Annotations" / f"{stem}.xml").write_text("<annotation></annotation>", encoding="utf-8")
            # Remove XML from second half (unlabeled IR train) to prove only RGB half requires GT XML.
            for stem in train_stems[3:]:
                (root / "Annotations" / f"{stem}.xml").unlink()
            counts = validate_dataset(tmp, 3)
            self.assertEqual(counts, {"rgb_train": 3, "ir_train": 3, "ir_validation": 1})
            with self.assertRaisesRegex(ValueError, "drop_last"):
                validate_dataset(tmp, 4)

    def test_make_training_args_forwards_gan_bridge_settings(self):
        from example_flir import build_training_config
        from notebook_support import make_training_args

        args = make_training_args({
            "data_root": "unused",
            "output_dir": "out",
            "workflow": "gan_bridge",
            "gan_weights": "/kaggle/input/gan/latest_net_G_A.pth",
            "domain_sampling": "interleaved",
            "bridge_ratios": ((10, 0, 0, 0), (5, 3, 2, 0), (3, 3, 3, 1), (2, 2, 4, 2), (1, 1, 2, 6)),
            "bridge_phase_fractions": (0.1, 0.3, 0.5, 0.7, 1.0),
            "bridge_source_sup_weight": 0.5,
            "bridge_saga_student_alphas": (1.0, 0.8, 0.6, 0.4, 0.2),
            "bridge_saga_teacher_alphas": (1.0, 0.8, 0.6, 0.6, 0.2),
            "bridge_saga_teachers": ("none", "rgb", "ir", "ir", "ir"),
        })
        cfg = build_training_config(args)
        self.assertEqual(cfg.workflow, "gan_bridge")
        self.assertEqual(cfg.gan_bridge.checkpoint, "/kaggle/input/gan/latest_net_G_A.pth")
        self.assertEqual(cfg.curriculum.sampling_mode, "interleaved")
        self.assertEqual(cfg.bridge_schedule.phase_fractions, (0.1, 0.3, 0.5, 0.7, 1.0))
        self.assertEqual(cfg.gan_bridge.source_sup_weight, 0.5)
        self.assertEqual(cfg.gan_bridge.saga_student_alphas, (1.0, 0.8, 0.6, 0.4, 0.2))
        self.assertEqual(cfg.gan_bridge.saga_teacher_alphas, (1.0, 0.8, 0.6, 0.6, 0.2))
        self.assertEqual(cfg.gan_bridge.saga_teachers, ("none", "rgb", "ir", "ir", "ir"))
        self.assertEqual(cfg.intermediate_domain.mode, "randommono_btf")
"""Keep configuration usable without importing detector implementations."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class ConfigImportTests(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent(code)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_configuration_can_be_built_without_torch_or_models(self):
        self.run_isolated("""
            import importlib.abc
            import sys

            class BlockDetectorPackages(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'torchvision', 'models'}:
                        raise ImportError('configuration imported detector package: ' + fullname)

            sys.meta_path.insert(0, BlockDetectorPackages())
            from config import GANBridgeConfig, Phase, TrainingConfig
            config = TrainingConfig(
                workflow='gan_bridge', total_iters=100,
                gan_bridge=GANBridgeConfig(checkpoint='G_A.pth'),
            )
            assert config.ema.start_steps == 20
            assert config.get_phase(20) is Phase.PHASE2_TRANSITION
            assert not any(name in sys.modules for name in ('torch', 'torchvision', 'models'))
        """)

    def test_legacy_adapter_enum_and_config_enum_are_identical_in_both_import_orders(self):
        for first_import in (
            "import config",
            "import models.fcos_adapter",
        ):
            with self.subTest(first_import=first_import):
                self.run_isolated(first_import + "\n" + textwrap.dedent("""
                    from config import ClassificationInitMode, FCOSModelConfig
                    from models.fcos_adapter import ClassificationInitMode as AdapterMode
                    assert ClassificationInitMode is AdapterMode
                    config = FCOSModelConfig(classification_init_mode=AdapterMode.RANDOM_HEAD)
                    assert config.classification_init_mode is AdapterMode.RANDOM_HEAD
                """))


if __name__ == "__main__":
    unittest.main()

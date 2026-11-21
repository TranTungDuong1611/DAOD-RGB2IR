"""Verify ordered, disjoint source/target selection through public loaders."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from config import DataConfig, TrainingConfig
from data.dataloaders import (
    build_dataloaders,
    build_ir_train_dataloader,
    build_rgb_train_dataloader,
)


class DisjointTrainDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "JPEGImages").mkdir()
        (self.root / "Annotations").mkdir()
        # Deliberately unsorted: partitioning must preserve the manifest order.
        self.stems = [f"FLIR_{i:05d}_PreviewData" for i in (4, 1, 5, 2, 6, 3)]
        self.validation_stems = ["FLIR_00008_PreviewData", "FLIR_00007_PreviewData"]
        self._write_train(self.stems)
        (self.root / "align_validation.txt").write_text(
            "\n".join(self.validation_stems) + "\n", encoding="utf-8"
        )
        for stem in self.stems + self.validation_stems:
            self._write_pair(stem)
        self.config = TrainingConfig(data=DataConfig(root=str(self.root)), device="cpu")
        self.config.loader.train.batch_size = 1
        self.config.loader.train.num_workers = 0
        self.config.loader.train.shuffle = False
        self.config.loader.train.drop_last = False
        self.config.loader.eval.num_workers = 0

    def _write_train(self, stems):
        (self.root / "align_train.txt").write_text(
            "\n".join(stems) + "\n", encoding="utf-8"
        )

    def _write_pair(self, stem):
        rgb_name = stem.replace("_PreviewData", "") + "_RGB.jpg"
        Image.new("RGB", (8, 8), color=(128, 64, 32)).save(
            self.root / "JPEGImages" / rgb_name
        )
        Image.new("L", (8, 8), color=128).save(
            self.root / "JPEGImages" / f"{stem}.jpeg"
        )
        (self.root / "Annotations" / f"{stem}.xml").write_text(
            "<annotation><object><name>person</name><bndbox>"
            "<xmin>1</xmin><ymin>1</ymin><xmax>7</xmax><ymax>7</ymax>"
            "</bndbox></object></annotation>", encoding="utf-8"
        )

    def test_train_loaders_use_first_half_rgb_and_second_half_unlabeled_ir(self):
        for workflow in ("curriculum", "gan_bridge", "rgb_baseline"):
            with self.subTest(workflow=workflow):
                self.config.workflow = workflow
                rgb = build_rgb_train_dataloader(self.config).dataset
                ir = build_ir_train_dataloader(self.config).dataset
                self.assertEqual(rgb.stems, self.stems[:3])
                self.assertEqual([p.stem for p in ir.ir_paths], self.stems[3:])
                self.assertTrue(set(rgb.stems).isdisjoint(p.stem for p in ir.ir_paths))
                _, target, sample_id = rgb[0]
                self.assertEqual(sample_id, self.stems[0])
                self.assertEqual(target["labels"].tolist(), [0])
                with mock.patch("data.datasets.parse_voc_xml", side_effect=AssertionError("IR training read GT")):
                    item = ir[0]
                self.assertEqual(len(item), 2)
                self.assertEqual(item[1], self.stems[3])

    def test_missing_images_do_not_move_the_partition_boundary(self):
        rgb_name = self.stems[0].replace("_PreviewData", "") + "_RGB.jpg"
        (self.root / "JPEGImages" / rgb_name).unlink()
        (self.root / "JPEGImages" / f"{self.stems[3]}.jpeg").unlink()
        rgb = build_rgb_train_dataloader(self.config).dataset
        ir = build_ir_train_dataloader(self.config).dataset
        self.assertEqual(rgb.stems, self.stems[1:3])
        self.assertEqual([p.stem for p in ir.ir_paths], self.stems[4:])

    def test_validation_keeps_the_complete_manifest_and_files_are_not_rewritten(self):
        before = (self.root / "align_train.txt").read_bytes()
        loaders = build_dataloaders(self.config)
        self.assertEqual(loaders["rgb_val"].dataset.stems, self.validation_stems)
        self.assertEqual(loaders["ir_val"].dataset.stems, self.validation_stems)
        self.assertEqual((self.root / "align_train.txt").read_bytes(), before)

    def test_odd_manifest_keeps_the_extra_image_in_target(self):
        extra = "FLIR_00009_PreviewData"
        self._write_pair(extra)
        self._write_train(self.stems + [extra])
        rgb = build_rgb_train_dataloader(self.config).dataset
        ir = build_ir_train_dataloader(self.config).dataset
        self.assertEqual(rgb.stems, self.stems[:3])
        self.assertEqual([p.stem for p in ir.ir_paths], self.stems[3:] + [extra])

    def test_duplicate_ids_are_rejected_before_they_can_cross_domains(self):
        self._write_train(self.stems[:3] + [self.stems[0]] + self.stems[4:])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_rgb_train_dataloader(self.config)

    def test_nested_voc_manifest_uses_the_same_ordered_partition(self):
        nested = self.root / "ImageSets" / "Main"
        nested.mkdir(parents=True)
        (self.root / "align_train.txt").rename(nested / "align_train.txt")
        self.assertEqual(build_rgb_train_dataloader(self.config).dataset.stems, self.stems[:3])
        ir = build_ir_train_dataloader(self.config).dataset
        self.assertEqual([p.stem for p in ir.ir_paths], self.stems[3:])

    def test_manifest_must_have_at_least_two_images(self):
        self._write_train(self.stems[:1])
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_ir_train_dataloader(self.config)


if __name__ == "__main__":
    unittest.main()

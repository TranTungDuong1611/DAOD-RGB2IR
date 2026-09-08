import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.datasets import FLIRIRDataset, FLIRIRValDataset, FLIRRGBDataset


class FLIRDatasetTests(unittest.TestCase):
    def test_root_split_takes_precedence_over_nested_voc_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "JPEGImages").mkdir()
            (root / "ImageSets" / "Main").mkdir(parents=True)
            root_stem = "FLIR_00001_PreviewData"
            nested_stem = "FLIR_00002_PreviewData"
            (root / "align_train.txt").write_text(
                f"{root_stem}\n", encoding="utf-8"
            )
            (root / "ImageSets" / "Main" / "align_train.txt").write_text(
                f"{nested_stem}\n", encoding="utf-8"
            )
            for stem in (root_stem, nested_stem):
                Image.new("L", (8, 8), color=128).save(
                    root / "JPEGImages" / f"{stem}.jpeg"
                )

            dataset = FLIRIRDataset(root)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0][1], root_stem)

    def test_missing_split_error_lists_both_supported_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"align_train\.txt.*ImageSets.*Main.*align_train\.txt",
            ):
                FLIRIRDataset(temp_dir)

    def test_all_datasets_accept_split_files_at_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "JPEGImages").mkdir()
            (root / "Annotations").mkdir()
            stem = "FLIR_00001_PreviewData"
            for split in ("train", "validation"):
                (root / f"align_{split}.txt").write_text(
                    f"{stem}\n", encoding="utf-8"
                )
            Image.new("RGB", (8, 8), color=(128, 64, 32)).save(
                root / "JPEGImages" / "FLIR_00001_RGB.jpg"
            )
            Image.new("L", (8, 8), color=128).save(
                root / "JPEGImages" / f"{stem}.jpeg"
            )
            (root / "Annotations" / f"{stem}.xml").write_text(
                """<annotation>
                <object><name>person</name><bndbox>
                <xmin>1</xmin><ymin>1</ymin><xmax>7</xmax><ymax>7</ymax>
                </bndbox></object>
                </annotation>""",
                encoding="utf-8",
            )

            self.assertEqual(len(FLIRRGBDataset(root)), 1)
            self.assertEqual(len(FLIRIRDataset(root)), 1)
            self.assertEqual(len(FLIRIRValDataset(root)), 1)

    def test_ir_validation_keeps_valid_boxes_smaller_than_training_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "JPEGImages").mkdir(parents=True)
            (root / "Annotations").mkdir()
            (root / "ImageSets" / "Main").mkdir(parents=True)
            stem = "FLIR_00001_PreviewData"
            (root / "ImageSets" / "Main" / "align_validation.txt").write_text(
                f"{stem}\n", encoding="utf-8"
            )
            Image.new("L", (8, 8), color=128).save(
                root / "JPEGImages" / f"{stem}.jpeg"
            )
            (root / "Annotations" / f"{stem}.xml").write_text(
                """<annotation>
                <object><name>person</name><bndbox>
                <xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax>
                </bndbox></object>
                </annotation>""",
                encoding="utf-8",
            )

            _, target, _ = FLIRIRValDataset(root)[0]

            self.assertEqual(target["boxes"].shape, (1, 4))
            self.assertEqual(target["labels"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()

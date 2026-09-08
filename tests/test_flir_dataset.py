import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.datasets import FLIRIRValDataset


class FLIRDatasetTests(unittest.TestCase):
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

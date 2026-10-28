"""IR audit labels must remain separate from supervised training targets."""

from pathlib import Path
import tempfile

import pytest
import torch

from data.augmentations import StudentAugmentor
from config import AugConfig, TrainingConfig
from training.data_flow import RouteDataFlow


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "Annotations").mkdir()
        yield Path(directory)


def test_ir_reader_preserves_small_boxes_and_aligns_horizontal_flip(tmp_path):
    from data.ir_annotations import IRAnnotationReader

    (tmp_path / "Annotations" / "a.xml").write_text(
        "<annotation><object><name>person</name><bndbox>"
        "<xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax>"
        "</bndbox></object></annotation>"
    )
    record, = IRAnnotationReader(tmp_path).read(("a",), ((10, 20),), hflipped=True)
    assert record.status == "available"
    torch.testing.assert_close(record.target["boxes"], torch.tensor([[17., 2., 19., 4.]]))
    assert record.target["labels"].tolist() == [0]


def test_ir_reader_distinguishes_missing_invalid_and_annotated_empty(tmp_path):
    from data.ir_annotations import IRAnnotationReader

    (tmp_path / "Annotations" / "empty.xml").write_text("<annotation/>")
    (tmp_path / "Annotations" / "broken.xml").write_text("<annotation>")
    (tmp_path / "Annotations" / "badbox.xml").write_text(
        "<annotation><object><name>car</name><bndbox>"
        "<xmin>nan</xmin><ymin>0</ymin><xmax>4</xmax><ymax>5</ymax>"
        "</bndbox></object></annotation>"
    )
    records = IRAnnotationReader(tmp_path).read(
        ("missing", "empty", "broken", "badbox"), ((10, 20),) * 4, hflipped=False
    )
    assert [r.status for r in records] == ["missing", "available", "invalid", "invalid"]
    assert records[0].target is None and records[2].target is None
    assert records[1].target["boxes"].shape == (0, 4)


def test_ir_geometry_capture_never_exposes_supervised_targets():
    config = TrainingConfig(device="cpu", aug=AugConfig(hflip_prob=1., blur_prob=0.,
                                                       brightness_prob=0., contrast_prob=0.))
    images = torch.arange(24.).reshape(1, 3, 2, 4)
    flow = RouteDataFlow([], [(images, ("a",))], device="cpu", augmentor=StudentAugmentor(config))
    data = flow.prepare(source_domain="ir", use_gt=True, student_alpha=None,
                        teacher_alpha=None, capture_geometry=True)
    assert data["targets"] is None
    assert data["diagnostic_hflip"] is True
    torch.testing.assert_close(data["teacher_images"], images.flip(-1))


def test_ir_reader_rejects_xml_dimensions_from_a_different_image(tmp_path):
    from data.ir_annotations import IRAnnotationReader

    Path(tmp_path / "Annotations" / "a.xml").write_text(
        "<annotation><size><width>99</width><height>10</height></size></annotation>"
    )
    record, = IRAnnotationReader(tmp_path).read(("a",), ((10, 20),), hflipped=False)
    assert record.status == "invalid"


@pytest.mark.parametrize("root_tag", ["annotation", "Annotation"])
def test_ir_reader_accepts_flir_annotation_root_casing(tmp_path, root_tag):
    from data.ir_annotations import IRAnnotationReader

    # Real FLIR exports use a capitalized root with lowercase child elements.
    # Keep image dimensions and a box from the reported failing sample.
    stem = "FLIR_01726_PreviewData"
    (tmp_path / "Annotations" / f"{stem}.xml").write_text(
        f"<{root_tag}><size><width>640</width><height>512</height></size>"
        "<object><name>car</name><bndbox>"
        "<xmin>526</xmin><ymin>225</ymin><xmax>638</xmax><ymax>328</ymax>"
        f"</bndbox></object></{root_tag}>"
    )
    record, = IRAnnotationReader(tmp_path).read((stem,), ((512, 640),), hflipped=False)
    assert record.status == "available"
    torch.testing.assert_close(record.target["boxes"], torch.tensor([[526., 225., 638., 328.]]))
    assert record.target["labels"].tolist() == [1]

import pytest
from spark.converters.labels import default_to_yolo_format, yolo_to_default_format


def test_default_to_yolo_label_format():
    img_w, img_h = 640, 640
    bbox = (30, 40, 70, 100)
    expected = (70.0 / img_w, 50.0 / img_h, 60 / img_w, 40 / img_h)
    output = default_to_yolo_format(img_w, img_h, *bbox)

    for i, val in enumerate(output):
        assert expected[i] == pytest.approx(val, rel=1e-9)


def test_yolo_to_default_label_format():
    img_w, img_h = 640, 640
    bbox = (70.0 / img_w, 50.0 / img_h, 60 / img_w, 40 / img_h)
    expected = (30, 40, 70, 100)

    output = yolo_to_default_format(img_w, img_h, *bbox)

    assert expected == output

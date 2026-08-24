import math

import pytest
from pydantic import ValidationError

from dashcam_ai.domain.geometry import BBox, FrameTransform, Point2D


def test_bbox_calculates_geometry() -> None:
    bbox = BBox(x1=10, y1=20, x2=50, y2=80)

    assert bbox.width == 40
    assert bbox.height == 60
    assert bbox.area == 2400
    assert bbox.center == Point2D(x=30, y=50)
    assert bbox.bottom_center == Point2D(x=30, y=80)


def test_bbox_rejects_inverted_extents() -> None:
    with pytest.raises(ValidationError):
        BBox(x1=10, y1=0, x2=5, y2=10)


def test_bbox_clips_to_frame() -> None:
    bbox = BBox(x1=-5, y1=-10, x2=120, y2=90)
    assert bbox.clip(100, 80) == BBox(x1=0, y1=0, x2=100, y2=80)


def test_letterbox_bbox_round_trip() -> None:
    transform = FrameTransform.letterbox(2560, 1440, 1280, 1280)
    original = BBox(x1=823.5, y1=302.25, x2=1050.75, y2=527.5)

    restored = transform.to_original_bbox(transform.to_inference_bbox(original))

    for actual, expected in zip(restored.as_xyxy(), original.as_xyxy(), strict=True):
        assert math.isclose(actual, expected, abs_tol=1e-9)
    assert transform.scale == 0.5
    assert transform.pad_y == 280


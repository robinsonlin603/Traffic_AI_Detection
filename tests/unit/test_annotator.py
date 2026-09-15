import pytest

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.visualization.annotator import OpenCVAnnotator


@pytest.mark.parametrize(
    ("class_name", "code"),
    [
        ("car", "C"),
        ("truck", "T"),
        ("bus", "B"),
        ("motorcycle", "M"),
        ("person", "P"),
        ("bicycle", "BC"),
    ],
)
def test_track_label_uses_compact_class_code(class_name: str, code: str) -> None:
    obj = TrackedObject(
        track_id=17,
        class_id=1,
        class_name=class_name,
        confidence=0.9,
        bbox=BBox(x1=10, y1=10, x2=20, y2=20),
    )

    assert OpenCVAnnotator._track_label_lines(obj) == (f"#17 {code}",)

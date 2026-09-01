from types import SimpleNamespace

import pytest

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.temporal import LaneChangeDirection
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

    assert OpenCVAnnotator._track_label_lines(obj, None) == (f"#17 {code}",)


def test_event_status_controls_box_color_without_extra_label_lines() -> None:
    state = SimpleNamespace(
        temporal=SimpleNamespace(lane_change_status=SimpleNamespace(value="confirmed")),
        membership=SimpleNamespace(membership=SimpleNamespace(value="inside_lane")),
    )
    obj = TrackedObject(
        track_id=4,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=BBox(x1=10, y1=10, x2=20, y2=20),
    )

    assert OpenCVAnnotator._track_box_color(state) == (0, 60, 255)
    assert OpenCVAnnotator._track_label_lines(obj, state) == ("#4 C",)


def test_event_banner_shows_semantic_direction() -> None:
    event = SimpleNamespace(
        direction=LaneChangeDirection.LEFT,
        track_id=9,
        status=SimpleNamespace(value="candidate"),
    )

    assert OpenCVAnnotator._event_banner(event) == "LANE CHANGE LEFT #9 CANDIDATE"

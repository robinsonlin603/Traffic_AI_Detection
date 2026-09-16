from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.visualization.annotator import OpenCVAnnotator


def test_track_label_only_displays_id() -> None:
    obj = TrackedObject(
        track_id=17,
        class_id=80,
        class_name="vehicle",
        confidence=0.9,
        bbox=BBox(x1=10, y1=10, x2=20, y2=20),
    )

    assert OpenCVAnnotator._track_label_lines(obj) == ("#17",)

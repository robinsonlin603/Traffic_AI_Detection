import numpy as np

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane_lines import LaneCurve, LaneLineFrame, LaneLineStatus
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


def test_lane_curve_is_split_at_vehicle_box() -> None:
    obj = TrackedObject(
        track_id=17,
        class_id=80,
        class_name="vehicle",
        confidence=0.9,
        bbox=BBox(x1=40, y1=30, x2=60, y2=70),
    )
    curve = LaneCurve(
        boundary_id="lane-1",
        points=tuple(Point2D(x=50, y=y) for y in (10, 20, 40, 60, 80, 90)),
        confidence=0.8,
    )

    segments = OpenCVAnnotator._visible_curve_segments(curve, [obj])

    assert segments == [[(50, 10), (50, 20)], [(50, 80), (50, 90)]]


def test_raster_occlusion_clips_between_curve_samples() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, :] = 255
    curve = LaneCurve(
        boundary_id="lane-1", confidence=0.8,
        points=(Point2D(x=50, y=10), Point2D(x=50, y=90)),
    )
    oversized_box = TrackedObject(
        track_id=17, class_id=80, class_name="vehicle", confidence=0.8,
        bbox=BBox(x1=0, y1=0, x2=99, y2=99),
    )
    result = OpenCVAnnotator().annotate(
        frame, [oversized_box],
        LaneLineFrame(status=LaneLineStatus.DEGRADED, curves=(curve,), reason="test"),
        lane_occlusion_mask=mask,
    )
    assert result[30, 50, 0] == 255
    assert not result[40:60, 40:60].any()

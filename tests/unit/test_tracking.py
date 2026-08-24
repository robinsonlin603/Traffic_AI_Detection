from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import Detection, TrackObservation
from dashcam_ai.tracking.centroid import CentroidTracker


def detection(x: float) -> Detection:
    return Detection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=BBox(x1=x, y1=20, x2=x + 20, y2=60),
    )


def test_centroid_tracker_preserves_identity_for_nearby_detection() -> None:
    tracker = CentroidTracker(maximum_distance=30)

    first = tracker.update([detection(10)], frame=None)
    second = tracker.update([detection(20)], frame=None)

    assert first[0].track_id == second[0].track_id


def test_centroid_tracker_creates_new_identity_for_distant_detection() -> None:
    tracker = CentroidTracker(maximum_distance=30)

    first = tracker.update([detection(10)], frame=None)
    second = tracker.update([detection(200)], frame=None)

    assert first[0].track_id != second[0].track_id


def test_track_observation_uses_bottom_center() -> None:
    obj = CentroidTracker().update([detection(10)], frame=None)[0]
    observation = TrackObservation.from_tracked_object(obj, frame_id=4, timestamp=0.2)

    assert observation.bottom_center.x == 20
    assert observation.bottom_center.y == 60


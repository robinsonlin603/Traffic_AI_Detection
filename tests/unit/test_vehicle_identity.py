from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.tracking.identity import VehicleIdentityResolver


def tracked(track_id: int, bbox: BBox, confidence: float = 0.8) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=80,
        class_name="vehicle",
        confidence=confidence,
        bbox=bbox,
    )


def box(cx: float, cy: float, width: float, height: float) -> BBox:
    return BBox(
        x1=cx - width / 2,
        y1=cy - height / 2,
        x2=cx + width / 2,
        y2=cy + height / 2,
    )


def test_nested_tracker_outputs_keep_one_canonical_id() -> None:
    resolver = VehicleIdentityResolver()
    smaller = box(100, 100, 50, 50)
    including_cargo = box(101, 110.45, 100, 100)

    first = resolver.update([tracked(8, smaller, 0.9), tracked(9, including_cargo, 0.8)])
    second = resolver.update([tracked(9, including_cargo, 0.8)])

    assert [item.track_id for item in first] == [8]
    assert [item.track_id for item in second] == [8]


def test_concurrent_neighboring_vehicles_are_not_merged() -> None:
    resolver = VehicleIdentityResolver()

    result = resolver.update(
        [
            tracked(177, box(100, 100, 100, 100)),
            tracked(197, box(125.2, 105.4, 40, 40)),
        ]
    )

    assert {item.track_id for item in result} == {177, 197}


def test_adjacent_frame_parked_vehicles_are_not_stitched() -> None:
    resolver = VehicleIdentityResolver()

    resolver.update([tracked(275, box(2514.8, 713.2, 73.8, 100.0))])
    result = resolver.update([tracked(294, box(2497.5, 706.7, 109.4, 135.2))])

    assert [item.track_id for item in result] == [294]


def test_new_id_after_one_missing_frame_is_stitched() -> None:
    resolver = VehicleIdentityResolver()

    resolver.update([tracked(508, box(1672.9, 649.5, 78.0, 94.3))])
    resolver.update([])
    result = resolver.update([tracked(523, box(1670.3, 673.0, 60.1, 73.5))])

    assert [item.track_id for item in result] == [508]


def test_unrelated_id_after_one_missing_frame_remains_new() -> None:
    resolver = VehicleIdentityResolver()

    resolver.update([tracked(10, box(100, 100, 50, 50))])
    resolver.update([])
    result = resolver.update([tracked(20, box(400, 400, 50, 50))])

    assert [item.track_id for item in result] == [20]


def test_new_id_after_two_missing_frames_is_stitched() -> None:
    resolver = VehicleIdentityResolver()

    resolver.update([tracked(37, box(100, 100, 100, 100))])
    resolver.update([])
    resolver.update([])
    result = resolver.update([tracked(214, box(118, 116, 100, 100))])

    assert [item.track_id for item in result] == [37]


def test_displaced_id_after_one_missing_frame_is_stitched_by_iou() -> None:
    resolver = VehicleIdentityResolver()

    resolver.update([tracked(393, box(100, 100, 100, 100))])
    resolver.update([])
    result = resolver.update([tracked(397, box(120, 104, 100, 100))])

    assert [item.track_id for item in result] == [393]


def test_canonical_collision_restores_distinct_raw_ids() -> None:
    for aliased_confidence, original_confidence in [(0.9, 0.8), (0.8, 0.9)]:
        resolver = VehicleIdentityResolver()
        original_box = box(100, 100, 100, 100)
        aliased_box = box(110, 100, 100, 100)
        distinct_box = box(400, 100, 100, 100)

        resolver.update([tracked(275, original_box)])
        resolver.update([])
        stitched = resolver.update([tracked(300, aliased_box)])
        collision = resolver.update(
            [
                tracked(300, aliased_box, aliased_confidence),
                tracked(275, distinct_box, original_confidence),
            ]
        )
        after_collision = resolver.update([tracked(300, aliased_box)])

        assert [item.track_id for item in stitched] == [275]
        assert {item.track_id for item in collision} == {275, 300}
        assert [item.track_id for item in after_collision] == [300]
        assert len({item.track_id for item in collision}) == len(collision)

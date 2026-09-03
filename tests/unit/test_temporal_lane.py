from __future__ import annotations

import json

import pytest

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import LaneChangeDirection, LaneChangePhase, LaneChangeStatus
from dashcam_ai.lane.temporal import TemporalLaneTracker


def feature(
    membership: LaneMembership,
    lane_id: str | None,
    *,
    distance: float | None = 30,
    boundary_lanes: tuple[str, str] = ("lane_left", "lane_center"),
    boundary_orders: tuple[int, int] = (0, 1),
) -> LaneMembershipFeature:
    orders = {"lane_left": 0, "lane_center": 1, "lane_right": 2}
    return LaneMembershipFeature(
        membership=membership,
        anchor=Point2D(x=100, y=200),
        lane_id=lane_id,
        lane_lateral_order=orders.get(lane_id),
        boundary_lane_ids=boundary_lanes if distance is not None else None,
        boundary_lane_orders=boundary_orders if distance is not None else None,
        signed_boundary_distance=distance,
        nearest_boundary_id="boundary" if distance is not None else None,
        geometry_confidence=1,
    )


def update(
    tracker: TemporalLaneTracker,
    frame_id: int,
    value: LaneMembershipFeature,
    *,
    track_id: int = 1,
    motion: EgoMotionStatus = EgoMotionStatus.VALID,
):
    return tracker.update(track_id, frame_id, frame_id * 0.1, value, motion)


def test_lane_id_requires_debounce_before_becoming_stable() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=2)
    first = update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"))
    second = update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_left"))

    assert first.stable_lane_id is None
    assert second.stable_lane_id == "lane_left"


def test_reset_discards_state_across_topology_versions() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    before = update(tracker, 10, feature(LaneMembership.INSIDE_LANE, "lane_left"))

    tracker.reset()
    after = update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_center"))

    assert before.stable_lane_id == "lane_left"
    assert after.stable_lane_id == "lane_center"
    assert after.source_lane is None


def test_boundary_jitter_preserves_stable_lane() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=2)
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_center"))
    update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_center"))
    jitter = update(
        tracker,
        2,
        feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2),
    )

    assert jitter.stable_lane_id == "lane_center"


@pytest.mark.parametrize(
    ("source", "target", "lanes", "orders", "direction"),
    [
        (
            "lane_center",
            "lane_left",
            ("lane_left", "lane_center"),
            (0, 1),
            LaneChangeDirection.LEFT,
        ),
        (
            "lane_center",
            "lane_right",
            ("lane_center", "lane_right"),
            (1, 2),
            LaneChangeDirection.RIGHT,
        ),
    ],
)
def test_adjacent_lane_change_records_ordered_timeline(
    source: str,
    target: str,
    lanes: tuple[str, str],
    orders: tuple[int, int],
    direction: LaneChangeDirection,
) -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
    )
    states = [
        update(tracker, 0, feature(LaneMembership.INSIDE_LANE, source)),
        update(
            tracker,
            1,
            feature(
                LaneMembership.NEAR_BOUNDARY,
                source,
                distance=2,
                boundary_lanes=lanes,
                boundary_orders=orders,
            ),
        ),
        update(
            tracker,
            2,
            feature(
                LaneMembership.NEAR_BOUNDARY,
                target,
                distance=1,
                boundary_lanes=lanes,
                boundary_orders=orders,
            ),
        ),
        update(tracker, 3, feature(LaneMembership.INSIDE_LANE, target)),
        update(tracker, 4, feature(LaneMembership.INSIDE_LANE, target)),
    ]

    assert states[1].lane_change_phase is LaneChangePhase.APPROACHING_BOUNDARY
    assert states[2].lane_change_phase is LaneChangePhase.CROSSING_BOUNDARY
    completed = states[-1]
    assert completed.lane_change_status is LaneChangeStatus.CONFIRMED
    assert completed.source_lane == source
    assert completed.target_lane == target
    assert completed.direction is direction
    assert completed.candidate_started_frame == 1
    assert completed.boundary_crossed_frame == 2
    assert completed.entered_started_frame == 3
    assert completed.completed_frame == 4


def test_candidate_returning_to_source_is_rejected() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"))
    update(tracker, 1, feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2))
    returned = update(tracker, 2, feature(LaneMembership.INSIDE_LANE, "lane_left"))

    assert returned.lane_change_status is LaneChangeStatus.REJECTED
    assert returned.reason == "vehicle returned to source lane"


def test_lane_jump_without_boundary_evidence_is_unknown() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"))
    jumped = update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_right"))

    assert jumped.lane_change_status is LaneChangeStatus.UNKNOWN
    assert jumped.completed_frame is None


def test_short_missing_observation_is_tolerated_then_becomes_unknown() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1, debounce_frames=1, maximum_missing_frames=1
    )
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"))
    update(tracker, 1, feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2))
    short = update(
        tracker,
        2,
        feature(LaneMembership.UNKNOWN, None, distance=None),
        motion=EgoMotionStatus.UNKNOWN,
    )
    expired = update(
        tracker,
        3,
        feature(LaneMembership.UNKNOWN, None, distance=None),
        motion=EgoMotionStatus.UNKNOWN,
    )

    assert short.lane_change_status is LaneChangeStatus.CANDIDATE
    assert expired.lane_change_status is LaneChangeStatus.UNKNOWN
    assert expired.stable_lane_id is None


def test_candidate_timeout_continues_during_missing_observations() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        maximum_missing_frames=10,
        candidate_timeout_seconds=0.15,
    )
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"))
    update(tracker, 1, feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2))
    timed_out = update(
        tracker,
        3,
        feature(LaneMembership.UNKNOWN, None, distance=None),
        motion=EgoMotionStatus.UNKNOWN,
    )

    assert timed_out.lane_change_status is LaneChangeStatus.REJECTED
    assert timed_out.reason == "candidate timed out"


def test_tracks_are_isolated_and_history_is_bounded_and_serializable() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1, history_size=2)
    update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_left"), track_id=1)
    update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_left"), track_id=1)
    latest = update(tracker, 2, feature(LaneMembership.INSIDE_LANE, "lane_left"), track_id=1)
    other = update(tracker, 0, feature(LaneMembership.INSIDE_LANE, "lane_right"), track_id=2)

    assert len(latest.history) == 2
    assert other.stable_lane_id == "lane_right"
    assert json.loads(latest.model_dump_json())["history"][-1]["anchor"] == {
        "x": 100.0,
        "y": 200.0,
    }


def test_frame_and_timestamp_must_increase_per_track() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_left"))

    with pytest.raises(ValueError, match="frame_id"):
        update(tracker, 1, feature(LaneMembership.INSIDE_LANE, "lane_left"))

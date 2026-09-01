from __future__ import annotations

import json

import pytest

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import (
    LaneChangeDirection,
    LaneChangePhase,
    LaneChangeStatus,
    LanePosition,
    LaneRelationPhase,
    ManeuverRelation,
)
from dashcam_ai.lane.temporal import TemporalLaneTracker


def feature(
    distance: float | None,
    membership: LaneMembership | None = None,
    boundary_id: str = "left",
    lane_id: str | None = None,
) -> LaneMembershipFeature:
    if membership is None:
        if distance is None:
            membership = LaneMembership.UNKNOWN
        elif distance > 0:
            membership = LaneMembership.INSIDE
        else:
            membership = LaneMembership.OUTSIDE
    return LaneMembershipFeature(
        membership=membership,
        anchor=Point2D(x=100, y=200),
        lane_id=lane_id,
        signed_boundary_distance=distance,
        nearest_boundary_id=boundary_id if distance is not None else None,
        geometry_confidence=1.0,
    )


def general_feature(
    membership: LaneMembership,
    lane_id: str | None,
    *,
    distance: float = 30,
    boundary_id: str = "boundary_left_center",
    boundary_lanes: tuple[str, str] = ("lane_left", "lane_center"),
    boundary_orders: tuple[int, int] = (0, 1),
) -> LaneMembershipFeature:
    lane_orders = {"lane_left": 0, "lane_center": 1, "lane_right": 2}
    return LaneMembershipFeature(
        membership=membership,
        anchor=Point2D(x=100, y=200),
        lane_id=lane_id,
        lane_lateral_order=lane_orders.get(lane_id),
        boundary_lane_ids=boundary_lanes,
        boundary_lane_orders=boundary_orders,
        signed_boundary_distance=distance,
        nearest_boundary_id=boundary_id,
        geometry_confidence=1.0,
    )


def general_update(
    tracker: TemporalLaneTracker,
    track_id: int,
    frame_id: int,
    feature_value: LaneMembershipFeature,
):
    return tracker.update(
        track_id,
        frame_id,
        frame_id * 0.1,
        feature_value,
        EgoMotionStatus.VALID,
    )


def update_sequence(
    tracker: TemporalLaneTracker,
    distances: list[float],
    *,
    track_id: int = 1,
    start_frame: int = 0,
    start_time: float = 0.0,
) -> list:
    return [
        tracker.update(
            track_id,
            start_frame + index,
            start_time + index * 0.1,
            feature(distance),
            EgoMotionStatus.VALID,
        )
        for index, distance in enumerate(distances)
    ]


def test_boundary_jitter_does_not_create_candidate() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=3, debounce_frames=2)

    states = update_sequence(tracker, [-80, -82, -6, 4, -5, 3, -75, -78])

    assert all(state.status is LaneChangeStatus.IDLE for state in states)


def test_stable_adjacent_vehicle_remains_idle() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)

    states = update_sequence(tracker, [-90, -85, -95, -88])

    assert states[-1].phase is LaneRelationPhase.ADJACENT
    assert states[-1].status is LaneChangeStatus.IDLE


def test_crossing_sequence_becomes_confirmed() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=3,
        minimum_confirmation_duration_seconds=0.2,
    )

    states = update_sequence(tracker, [-80, -25, -5, 5, 25, 30, 35])

    assert states[1].status is LaneChangeStatus.CANDIDATE
    assert states[-1].phase is LaneRelationPhase.ENTERED
    assert states[-1].status is LaneChangeStatus.CONFIRMED
    assert states[-1].candidate_started_frame == 1
    assert states[-1].entered_started_frame == 4
    assert states[-1].maneuver_relation is ManeuverRelation.ENTERING_EGO
    assert states[-1].from_lane is LanePosition.LEFT_ADJACENT
    assert states[-1].to_lane is LanePosition.EGO


def test_candidate_returning_to_adjacent_is_rejected() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)

    states = update_sequence(tracker, [-80, -20, -3, -90])

    assert states[-1].status is LaneChangeStatus.REJECTED
    assert states[-1].reason == "vehicle returned to origin lane"


def test_vehicle_can_leave_ego_lane_without_becoming_a_cutin() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
    )

    states = update_sequence(tracker, [40, 35, 5, -10, -70, -80])

    assert states[-1].status is LaneChangeStatus.CONFIRMED
    assert states[-1].maneuver_relation is ManeuverRelation.LEAVING_EGO
    assert states[-1].from_lane is LanePosition.EGO
    assert states[-1].to_lane is LanePosition.LEFT_ADJACENT


def test_same_track_can_enter_then_leave_as_two_maneuvers() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
    )

    states = update_sequence(
        tracker,
        [-80, -20, -5, 25, 30, 5, -10, -70, -80],
    )

    confirmed = [state for state in states if state.status is LaneChangeStatus.CONFIRMED]
    assert [state.maneuver_relation for state in confirmed] == [
        ManeuverRelation.ENTERING_EGO,
        ManeuverRelation.LEAVING_EGO,
    ]


def test_short_occlusion_preserves_candidate_and_can_recover() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
        maximum_missing_frames=1,
    )
    update_sequence(tracker, [-80, -20])
    missing = tracker.update(
        1, 2, 0.2, feature(None), EgoMotionStatus.UNKNOWN
    )
    entered = update_sequence(
        tracker, [25, 30], track_id=1, start_frame=3, start_time=0.3
    )

    assert missing.status is LaneChangeStatus.CANDIDATE
    assert missing.missing_observations == 1
    assert entered[-1].status is LaneChangeStatus.CONFIRMED


def test_missing_motion_beyond_tolerance_rejects_candidate() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1, debounce_frames=1, maximum_missing_frames=1
    )
    update_sequence(tracker, [-80, -20])

    first = tracker.update(1, 2, 0.2, feature(2), EgoMotionStatus.UNKNOWN)
    second = tracker.update(1, 3, 0.3, feature(5), EgoMotionStatus.UNKNOWN)

    assert first.status is LaneChangeStatus.CANDIDATE
    assert second.status is LaneChangeStatus.REJECTED
    assert second.phase is LaneRelationPhase.UNKNOWN
    assert second.reason == "temporal evidence missing beyond tolerance"


def test_candidate_timeout_continues_during_missing_observations() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        maximum_missing_frames=10,
        candidate_timeout_seconds=0.15,
    )
    update_sequence(tracker, [-80, -20])

    result = tracker.update(1, 3, 0.3, feature(None), EgoMotionStatus.UNKNOWN)

    assert result.status is LaneChangeStatus.REJECTED
    assert result.reason == "candidate timed out"


def test_frame_and_timestamp_must_increase_per_track() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    tracker.update(1, 5, 0.5, feature(-80), EgoMotionStatus.VALID)

    with pytest.raises(ValueError, match="frame_id"):
        tracker.update(1, 5, 0.6, feature(-70), EgoMotionStatus.VALID)
    with pytest.raises(ValueError, match="timestamp"):
        tracker.update(1, 6, 0.5, feature(-70), EgoMotionStatus.VALID)


def test_tracks_are_isolated_and_history_is_bounded_and_serializable() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1, debounce_frames=1, history_size=3
    )
    update_sequence(tracker, [-80, -20, -5, 25], track_id=1)
    other = update_sequence(tracker, [-90, -85], track_id=2)[-1]
    latest = tracker.update(1, 4, 0.4, feature(30), EgoMotionStatus.VALID)

    assert other.status is LaneChangeStatus.IDLE
    assert len(latest.history) == 3
    payload = json.loads(latest.model_dump_json())
    assert payload["history"][-1]["ego_motion_status"] == "valid"
    assert payload["boundary_id"] == "left"


def test_general_lane_id_requires_debounce_before_becoming_stable() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=2)

    first = tracker.update(
        9,
        0,
        0.0,
        feature(30, LaneMembership.INSIDE_LANE, lane_id="lane_left"),
        EgoMotionStatus.VALID,
    )
    second = tracker.update(
        9,
        1,
        0.1,
        feature(32, LaneMembership.INSIDE_LANE, lane_id="lane_left"),
        EgoMotionStatus.VALID,
    )

    assert first.observed_lane_id == "lane_left"
    assert first.stable_lane_id is None
    assert second.stable_lane_id == "lane_left"
    assert second.history[-1].stable_lane_id == "lane_left"


def test_boundary_jitter_does_not_replace_stable_general_lane() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=2)
    tracker.update(
        10,
        0,
        0.0,
        feature(30, LaneMembership.INSIDE_LANE, lane_id="lane_center"),
        EgoMotionStatus.VALID,
    )
    stable = tracker.update(
        10,
        1,
        0.1,
        feature(30, LaneMembership.INSIDE_LANE, lane_id="lane_center"),
        EgoMotionStatus.VALID,
    )
    jitter = tracker.update(
        10,
        2,
        0.2,
        feature(2, LaneMembership.NEAR_BOUNDARY, lane_id="lane_left"),
        EgoMotionStatus.VALID,
    )

    assert stable.stable_lane_id == "lane_center"
    assert jitter.stable_lane_id == "lane_center"


def test_short_missing_observation_preserves_stable_lane_then_clears_it() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1, debounce_frames=1, maximum_missing_frames=1
    )
    stable = tracker.update(
        11,
        0,
        0.0,
        feature(30, LaneMembership.INSIDE_LANE, lane_id="lane_right"),
        EgoMotionStatus.VALID,
    )
    short = tracker.update(11, 1, 0.1, feature(None), EgoMotionStatus.UNKNOWN)
    expired = tracker.update(11, 2, 0.2, feature(None), EgoMotionStatus.UNKNOWN)

    assert stable.stable_lane_id == "lane_right"
    assert short.stable_lane_id == "lane_right"
    assert expired.stable_lane_id is None


@pytest.mark.parametrize(
    ("source", "target", "boundary_lanes", "boundary_orders", "direction"),
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
def test_general_adjacent_lane_change_records_ordered_timeline(
    source: str,
    target: str,
    boundary_lanes: tuple[str, str],
    boundary_orders: tuple[int, int],
    direction: LaneChangeDirection,
) -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
    )
    states = [
        general_update(
            tracker,
            20,
            0,
            general_feature(LaneMembership.INSIDE_LANE, source),
        ),
        general_update(
            tracker,
            20,
            1,
            general_feature(
                LaneMembership.NEAR_BOUNDARY,
                source,
                distance=2,
                boundary_lanes=boundary_lanes,
                boundary_orders=boundary_orders,
            ),
        ),
        general_update(
            tracker,
            20,
            2,
            general_feature(
                LaneMembership.NEAR_BOUNDARY,
                target,
                distance=1,
                boundary_lanes=boundary_lanes,
                boundary_orders=boundary_orders,
            ),
        ),
        general_update(
            tracker,
            20,
            3,
            general_feature(LaneMembership.INSIDE_LANE, target),
        ),
        general_update(
            tracker,
            20,
            4,
            general_feature(LaneMembership.INSIDE_LANE, target),
        ),
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
    assert (
        completed.candidate_started_timestamp
        <= completed.boundary_crossed_timestamp
        <= completed.completed_timestamp
    )


def test_general_candidate_returning_to_source_is_rejected() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    general_update(
        tracker,
        21,
        0,
        general_feature(LaneMembership.INSIDE_LANE, "lane_left"),
    )
    candidate = general_update(
        tracker,
        21,
        1,
        general_feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2),
    )
    returned = general_update(
        tracker,
        21,
        2,
        general_feature(LaneMembership.INSIDE_LANE, "lane_left"),
    )

    assert candidate.lane_change_status is LaneChangeStatus.CANDIDATE
    assert returned.lane_change_status is LaneChangeStatus.REJECTED
    assert returned.reason == "vehicle returned to source lane"


def test_lane_jump_without_boundary_evidence_is_unknown_not_confirmed() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)
    general_update(
        tracker,
        22,
        0,
        general_feature(LaneMembership.INSIDE_LANE, "lane_left"),
    )
    jumped = general_update(
        tracker,
        22,
        1,
        general_feature(LaneMembership.INSIDE_LANE, "lane_right"),
    )

    assert jumped.lane_change_phase is LaneChangePhase.UNKNOWN
    assert jumped.lane_change_status is LaneChangeStatus.UNKNOWN
    assert jumped.reason == "lane changed without shared-boundary evidence"


def test_general_candidate_becomes_unknown_after_motion_gap() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1, debounce_frames=1, maximum_missing_frames=0
    )
    general_update(
        tracker,
        23,
        0,
        general_feature(LaneMembership.INSIDE_LANE, "lane_left"),
    )
    general_update(
        tracker,
        23,
        1,
        general_feature(LaneMembership.NEAR_BOUNDARY, "lane_left", distance=2),
    )
    unknown = tracker.update(
        23,
        2,
        0.2,
        general_feature(LaneMembership.UNKNOWN, None, distance=0),
        EgoMotionStatus.UNKNOWN,
    )

    assert unknown.lane_change_phase is LaneChangePhase.UNKNOWN
    assert unknown.lane_change_status is LaneChangeStatus.UNKNOWN
    assert unknown.completed_frame is None

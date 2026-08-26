from __future__ import annotations

import json

import pytest

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import LaneChangeStatus, LaneRelationPhase
from dashcam_ai.lane.temporal import TemporalLaneTracker


def feature(
    distance: float | None,
    membership: LaneMembership | None = None,
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
        signed_boundary_distance=distance,
        nearest_boundary_id="left" if distance is not None else None,
        geometry_confidence=1.0,
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


def test_candidate_returning_to_adjacent_is_rejected() -> None:
    tracker = TemporalLaneTracker(smoothing_window_frames=1, debounce_frames=1)

    states = update_sequence(tracker, [-80, -20, -3, -90])

    assert states[-1].status is LaneChangeStatus.REJECTED
    assert states[-1].reason == "vehicle returned to adjacent lane"


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

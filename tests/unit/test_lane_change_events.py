from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dashcam_ai.domain.events import (
    ConfidenceBreakdown,
    EventEvidence,
    EventStatus,
    LaneChangeEvent,
)
from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import (
    LaneChangeDirection,
    LaneChangePhase,
    LaneChangeStatus,
    TemporalLaneObservation,
    TemporalLaneState,
)
from dashcam_ai.events.lane_change import LaneChangeEventBuilder


def observation(frame_id: int, lane_id: str) -> TemporalLaneObservation:
    return TemporalLaneObservation(
        frame_id=frame_id,
        timestamp=frame_id * 0.1,
        membership=LaneMembership.INSIDE_LANE,
        anchor=Point2D(x=100 + frame_id, y=200),
        lane_id=lane_id,
        stable_lane_id=lane_id,
        signed_boundary_distance=20,
        smoothed_signed_boundary_distance=20,
        nearest_boundary_id="boundary_center_right",
        ego_motion_status=EgoMotionStatus.VALID,
    )


def temporal_state(
    status: LaneChangeStatus = LaneChangeStatus.CONFIRMED,
) -> TemporalLaneState:
    completed = status is LaneChangeStatus.CONFIRMED
    return TemporalLaneState(
        track_id=7,
        frame_id=4,
        timestamp=0.4,
        observed_lane_id="lane_right",
        stable_lane_id="lane_right",
        lane_change_phase=LaneChangePhase.ENTERED_NEW_LANE,
        lane_change_status=status,
        source_lane="lane_center",
        target_lane="lane_right",
        direction=LaneChangeDirection.RIGHT,
        candidate_started_frame=1,
        candidate_started_timestamp=0.1,
        boundary_crossed_frame=2,
        boundary_crossed_timestamp=0.2,
        entered_started_frame=3,
        entered_started_timestamp=0.3,
        completed_frame=4 if completed else None,
        completed_timestamp=0.4 if completed else None,
        missing_observations=0,
        valid_motion_observations=4,
        boundary_id="boundary_center_right",
        history=tuple(
            observation(frame, "lane_right" if frame >= 3 else "lane_center")
            for frame in range(1, 5)
        ),
    )


def test_confirmed_state_builds_general_serializable_event() -> None:
    event = LaneChangeEventBuilder().build(temporal_state())

    assert event is not None
    assert event.status is EventStatus.CONFIRMED
    assert event.source_lane == "lane_center"
    assert event.target_lane == "lane_right"
    assert event.direction is LaneChangeDirection.RIGHT
    assert event.started_at == 0.1
    assert event.lane_crossed_at == 0.2
    assert event.completed_at == 0.4
    assert event.evidence.frame_ids == (1, 2, 3, 4)
    assert len(event.evidence.trajectory) == 4
    payload = json.loads(event.model_dump_json())
    assert payload["event_type"] == "lane_change"
    assert "maneuver_relation" not in payload
    assert "cut_in" not in payload


def test_candidate_preserves_incomplete_timeline() -> None:
    event = LaneChangeEventBuilder().build(
        temporal_state(LaneChangeStatus.CANDIDATE).model_copy(
            update={
                "boundary_crossed_frame": None,
                "boundary_crossed_timestamp": None,
            }
        )
    )

    assert event is not None
    assert event.status is EventStatus.CANDIDATE
    assert event.lane_crossed_at is None
    assert event.completed_at is None


def test_idle_state_does_not_build_event() -> None:
    state = temporal_state(LaneChangeStatus.IDLE).model_copy(
        update={"candidate_started_frame": None, "candidate_started_timestamp": None}
    )
    assert LaneChangeEventBuilder().build(state) is None


def test_event_rejects_invalid_timeline_order() -> None:
    confidence = ConfidenceBreakdown(
        timeline=1,
        membership_stability=1,
        ego_motion=1,
        relative_motion=1,
        overall=1,
    )
    with pytest.raises(ValidationError, match="lane crossing"):
        LaneChangeEvent(
            event_id="lane-change:1:10",
            status=EventStatus.CONFIRMED,
            track_id=1,
            source_lane="lane_left",
            target_lane="lane_center",
            direction=LaneChangeDirection.RIGHT,
            started_frame=10,
            started_at=1.0,
            lane_crossed_frame=9,
            lane_crossed_at=0.9,
            completed_frame=12,
            completed_at=1.2,
            confidence=1,
            confidence_breakdown=confidence,
            evidence=EventEvidence(),
        )

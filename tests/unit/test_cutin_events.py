from __future__ import annotations

import json

from dashcam_ai.domain.events import EventStatus
from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import LaneMembership, NormalizedPoint2D
from dashcam_ai.domain.motion import EgoMotionStatus, RelativeMotionSummary
from dashcam_ai.domain.temporal import (
    LaneChangeStatus,
    LanePosition,
    LaneRelationPhase,
    ManeuverRelation,
    TemporalLaneObservation,
    TemporalLaneState,
)
from dashcam_ai.events.corridor import ConfiguredForwardCorridor
from dashcam_ai.events.cutin import CutInDetector
from dashcam_ai.events.lane_change import LaneChangeEventBuilder


def observation(
    frame_id: int,
    distance: float,
    motion: EgoMotionStatus = EgoMotionStatus.VALID,
) -> TemporalLaneObservation:
    return TemporalLaneObservation(
        frame_id=frame_id,
        timestamp=frame_id * 0.1,
        membership=LaneMembership.INSIDE if distance > 0 else LaneMembership.OUTSIDE,
        signed_boundary_distance=distance,
        smoothed_signed_boundary_distance=distance,
        nearest_boundary_id="left",
        ego_motion_status=motion,
    )


def temporal_state(
    status: LaneChangeStatus = LaneChangeStatus.CONFIRMED,
    motions: tuple[EgoMotionStatus, ...] = (
        EgoMotionStatus.VALID,
        EgoMotionStatus.VALID,
        EgoMotionStatus.VALID,
    ),
    relative_motion: RelativeMotionSummary | None = None,
) -> TemporalLaneState:
    distances = (-20.0, 5.0, 30.0)
    history = tuple(
        observation(index + 1, distance, motion)
        for index, (distance, motion) in enumerate(zip(distances, motions, strict=True))
    )
    return TemporalLaneState(
        track_id=7,
        phase=LaneRelationPhase.ENTERED,
        status=status,
        frame_id=3,
        timestamp=0.3,
        candidate_started_frame=1,
        candidate_started_timestamp=0.1,
        entered_started_frame=3,
        entered_started_timestamp=0.3,
        missing_observations=0,
        valid_motion_observations=sum(
            motion is EgoMotionStatus.VALID for motion in motions
        ),
        boundary_id="left",
        maneuver_relation=ManeuverRelation.ENTERING_EGO,
        from_lane=LanePosition.LEFT_ADJACENT,
        to_lane=LanePosition.EGO,
        relative_motion=relative_motion,
        reason="returned outside" if status is LaneChangeStatus.REJECTED else None,
        history=history,
    )


def corridor():
    return ConfiguredForwardCorridor(
        [
            NormalizedPoint2D(x=0.4, y=0.5),
            NormalizedPoint2D(x=0.6, y=0.5),
            NormalizedPoint2D(x=0.8, y=1.0),
            NormalizedPoint2D(x=0.2, y=1.0),
        ]
    ).resolve(1000, 1000)


def test_configured_corridor_maps_to_original_resolution() -> None:
    result = corridor()

    assert result.polygon[0] == Point2D(x=400, y=500)
    assert result.polygon[2] == Point2D(x=800, y=1000)
    assert result.contains(Point2D(x=500, y=800))
    assert result.contains(Point2D(x=500, y=1000))
    assert not result.contains(Point2D(x=100, y=800))


def test_confirmed_temporal_state_builds_serializable_lane_change_event() -> None:
    event = LaneChangeEventBuilder().build(temporal_state())

    assert event is not None
    assert event.status is EventStatus.CONFIRMED
    assert event.event_id == "lane-change:7:1"
    assert event.maneuver_relation is ManeuverRelation.ENTERING_EGO
    assert event.from_lane is LanePosition.LEFT_ADJACENT
    assert event.to_lane is LanePosition.EGO
    assert len(event.evidence.frames) == 3
    payload = json.loads(event.model_dump_json())
    assert payload["evidence"]["boundary_id"] == "left"


def test_idle_temporal_state_does_not_build_event() -> None:
    state = temporal_state(status=LaneChangeStatus.IDLE).model_copy(
        update={"candidate_started_frame": None, "candidate_started_timestamp": None}
    )

    assert LaneChangeEventBuilder().build(state) is None


def test_invalid_current_motion_cannot_build_confirmed_lane_change() -> None:
    event = LaneChangeEventBuilder().build(
        temporal_state(
            motions=(
                EgoMotionStatus.VALID,
                EgoMotionStatus.VALID,
                EgoMotionStatus.UNKNOWN,
            )
        )
    )

    assert event is not None
    assert event.status is EventStatus.CANDIDATE
    assert event.reason == "current ego-motion evidence is invalid"


def test_true_image_space_cutin_is_confirmed() -> None:
    lane_change = LaneChangeEventBuilder().build(temporal_state())
    assert lane_change is not None

    event = CutInDetector().detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.CONFIRMED
    assert event.corridor_interaction is True
    assert event.bbox_expansion_ratio is not None
    assert event.bbox_expansion_ratio > 0
    assert event.confidence.overall >= 0.65
    assert event.evidence.frames[-1].bbox is not None


def test_supported_relative_motion_contributes_to_cutin_confidence() -> None:
    summary = RelativeMotionSummary(
        valid_observations=3,
        cumulative_lateral_displacement=0.01,
        expected_lateral_progress=0.01,
        directional_consistency=1,
        motion_quality=0.9,
        scene_consistency=1,
        stationary_ratio=0,
        supported=True,
        confidence=0.95,
    )
    lane_change = LaneChangeEventBuilder().build(
        temporal_state(relative_motion=summary)
    )
    assert lane_change is not None

    event = CutInDetector(require_relative_motion=True).detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.CONFIRMED
    assert event.confidence.relative_motion == 0.95
    assert event.confidence.lateral_progress == 1
    assert event.confidence.direction_compatibility == 1
    assert event.confidence.scene_consistency == 1


def test_stationary_relative_motion_rejects_cutin() -> None:
    summary = RelativeMotionSummary(
        valid_observations=3,
        cumulative_lateral_displacement=0,
        expected_lateral_progress=0,
        directional_consistency=0,
        motion_quality=0.9,
        scene_consistency=1,
        stationary_ratio=1,
        supported=False,
        confidence=0.38,
        reason="vehicle is stationary relative to the background",
    )
    lane_change = LaneChangeEventBuilder().build(
        temporal_state(relative_motion=summary)
    )
    assert lane_change is not None

    event = CutInDetector(require_relative_motion=True).detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.REJECTED
    assert event.reason == "vehicle is stationary relative to the background"


def test_lane_change_outside_forward_corridor_is_rejected() -> None:
    lane_change = LaneChangeEventBuilder().build(temporal_state())
    assert lane_change is not None

    event = CutInDetector().detect(
        lane_change,
        current_bbox=BBox(x1=40, y1=700, x2=140, y2=850),
        previous_bbox=BBox(x1=50, y1=720, x2=130, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.REJECTED
    assert event.reason == "vehicle bottom-center is outside forward corridor"


def test_leaving_lane_change_is_not_a_cutin() -> None:
    state = temporal_state().model_copy(
        update={
            "maneuver_relation": ManeuverRelation.LEAVING_EGO,
            "from_lane": LanePosition.EGO,
            "to_lane": LanePosition.RIGHT_ADJACENT,
        }
    )
    lane_change = LaneChangeEventBuilder().build(state)
    assert lane_change is not None

    event = CutInDetector().detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.REJECTED
    assert event.reason == "lane change is not entering the ego lane"


def test_non_expanding_bbox_is_rejected() -> None:
    lane_change = LaneChangeEventBuilder().build(temporal_state())
    assert lane_change is not None
    box = BBox(x1=460, y1=700, x2=540, y2=850)

    event = CutInDetector().detect(lane_change, box, box, corridor())

    assert event.status is EventStatus.REJECTED
    assert event.reason == "bbox does not show forward image-space interaction"


def test_small_bbox_expansion_remains_candidate() -> None:
    lane_change = LaneChangeEventBuilder().build(temporal_state())
    assert lane_change is not None

    event = CutInDetector(minimum_bbox_expansion_ratio=0.2).detect(
        lane_change,
        current_bbox=BBox(x1=459, y1=699, x2=541, y2=851),
        previous_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        corridor=corridor(),
    )

    assert event.status is EventStatus.CANDIDATE
    assert event.reason == "bbox expansion is below confirmation threshold"


def test_incomplete_motion_evidence_remains_candidate() -> None:
    lane_change = LaneChangeEventBuilder().build(
        temporal_state(
            motions=(
                EgoMotionStatus.UNKNOWN,
                EgoMotionStatus.VALID,
                EgoMotionStatus.VALID,
            )
        )
    )
    assert lane_change is not None

    event = CutInDetector().detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.CANDIDATE
    assert event.reason == "ego-motion evidence is incomplete"


def test_rejected_lane_change_produces_rejected_cutin() -> None:
    lane_change = LaneChangeEventBuilder().build(
        temporal_state(status=LaneChangeStatus.REJECTED)
    )
    assert lane_change is not None

    event = CutInDetector().detect(
        lane_change,
        current_bbox=BBox(x1=460, y1=700, x2=540, y2=850),
        previous_bbox=BBox(x1=470, y1=720, x2=530, y2=830),
        corridor=corridor(),
    )

    assert event.status is EventStatus.REJECTED
    assert event.reason == "lane change was rejected"

"""可序列化的一般車道歸屬與換道時間狀態。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import (
    EgoMotionStatus,
    RelativeMotionEvidence,
    RelativeMotionSummary,
)


class LaneChangePhase(StrEnum):
    UNKNOWN = "unknown"
    STABLE_IN_LANE = "stable_in_lane"
    APPROACHING_BOUNDARY = "approaching_boundary"
    CROSSING_BOUNDARY = "crossing_boundary"
    ENTERED_NEW_LANE = "entered_new_lane"


class LaneChangeDirection(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class LaneChangeStatus(StrEnum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class TemporalLaneObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    membership: LaneMembership
    anchor: Point2D
    lane_id: str | None = None
    stable_lane_id: str | None = None
    signed_boundary_distance: float | None = None
    smoothed_signed_boundary_distance: float | None = None
    nearest_boundary_id: str | None = None
    ego_motion_status: EgoMotionStatus
    relative_motion: RelativeMotionEvidence | None = None


class TemporalLaneState(BaseModel):
    model_config = ConfigDict(frozen=True)
    track_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    observed_lane_id: str | None = None
    stable_lane_id: str | None = None
    lane_change_phase: LaneChangePhase = LaneChangePhase.UNKNOWN
    lane_change_status: LaneChangeStatus = LaneChangeStatus.IDLE
    source_lane: str | None = None
    target_lane: str | None = None
    direction: LaneChangeDirection = LaneChangeDirection.UNKNOWN
    candidate_started_frame: int | None = Field(default=None, ge=0)
    candidate_started_timestamp: float | None = Field(default=None, ge=0)
    boundary_crossed_frame: int | None = Field(default=None, ge=0)
    boundary_crossed_timestamp: float | None = Field(default=None, ge=0)
    entered_started_frame: int | None = Field(default=None, ge=0)
    entered_started_timestamp: float | None = Field(default=None, ge=0)
    completed_frame: int | None = Field(default=None, ge=0)
    completed_timestamp: float | None = Field(default=None, ge=0)
    missing_observations: int = Field(ge=0)
    valid_motion_observations: int = Field(ge=0)
    boundary_id: str | None = None
    relative_motion: RelativeMotionSummary | None = None
    reason: str | None = None
    history: tuple[TemporalLaneObservation, ...] = ()

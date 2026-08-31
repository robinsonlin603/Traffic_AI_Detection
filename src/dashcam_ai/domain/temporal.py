"""可序列化的時間車道歸屬與換道狀態模型。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import EgoMotionStatus


class LaneRelationPhase(StrEnum):
    """追蹤物件相對自車車道的時間階段。"""

    UNKNOWN = "unknown"
    ADJACENT = "adjacent"
    APPROACHING = "approaching"
    CROSSING = "crossing"
    ENTERED = "entered"


class LaneChangeStatus(StrEnum):
    """換道候選的生命週期。"""

    IDLE = "idle"
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class LanePosition(StrEnum):
    """以自車車道為中心描述換道前後的位置。"""

    EGO = "ego"
    LEFT_ADJACENT = "left_adjacent"
    RIGHT_ADJACENT = "right_adjacent"
    UNKNOWN = "unknown"


class ManeuverRelation(StrEnum):
    """描述一次換道是進入或離開自車車道。"""

    ENTERING_EGO = "entering_ego"
    LEAVING_EGO = "leaving_ego"
    UNKNOWN = "unknown"


class TemporalLaneObservation(BaseModel):
    """單一追蹤物件在某幀的 temporal lane 證據。"""

    model_config = ConfigDict(frozen=True)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    membership: LaneMembership
    signed_boundary_distance: float | None = None
    smoothed_signed_boundary_distance: float | None = None
    nearest_boundary_id: str | None = None
    ego_motion_status: EgoMotionStatus


class TemporalLaneState(BaseModel):
    """特定 track 的有界換道狀態快照。"""

    model_config = ConfigDict(frozen=True)
    track_id: int = Field(ge=0)
    phase: LaneRelationPhase
    status: LaneChangeStatus
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    candidate_started_frame: int | None = Field(default=None, ge=0)
    candidate_started_timestamp: float | None = Field(default=None, ge=0)
    entered_started_frame: int | None = Field(default=None, ge=0)
    entered_started_timestamp: float | None = Field(default=None, ge=0)
    missing_observations: int = Field(ge=0)
    valid_motion_observations: int = Field(ge=0)
    boundary_id: str | None = None
    maneuver_relation: ManeuverRelation = ManeuverRelation.UNKNOWN
    from_lane: LanePosition = LanePosition.UNKNOWN
    to_lane: LanePosition = LanePosition.UNKNOWN
    reason: str | None = None
    history: tuple[TemporalLaneObservation, ...] = ()

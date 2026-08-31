"""可序列化的換道、切入事件與影像空間證據模型。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import LanePosition, ManeuverRelation


class EventStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class EventEvidenceFrame(BaseModel):
    """事件判斷所引用的單幀 temporal 與位置證據。"""

    model_config = ConfigDict(frozen=True)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    membership: LaneMembership
    signed_boundary_distance: float | None = None
    smoothed_signed_boundary_distance: float | None = None
    ego_motion_status: EgoMotionStatus
    bbox: BBox | None = None


class EventEvidence(BaseModel):
    """事件的有界證據影格及其適用邊界。"""

    model_config = ConfigDict(frozen=True)
    boundary_id: str | None = None
    frames: tuple[EventEvidenceFrame, ...] = ()


class ConfidenceBreakdown(BaseModel):
    """切入信心分數的可解釋 image-space 組成。"""

    model_config = ConfigDict(frozen=True)
    lane_change: float = Field(ge=0, le=1)
    corridor_interaction: float = Field(ge=0, le=1)
    bbox_expansion: float = Field(ge=0, le=1)
    motion_quality: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class LaneChangeEvent(BaseModel):
    """由 temporal lane state 形成的換道事件。"""

    model_config = ConfigDict(frozen=True)
    event_type: Literal["lane_change"] = "lane_change"
    event_id: str = Field(min_length=1)
    status: EventStatus
    track_id: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_timestamp: float = Field(ge=0)
    end_timestamp: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    maneuver_relation: ManeuverRelation
    from_lane: LanePosition
    to_lane: LanePosition
    evidence: EventEvidence
    reason: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> LaneChangeEvent:
        if self.end_frame < self.start_frame or self.end_timestamp < self.start_timestamp:
            raise ValueError("event end must not precede event start")
        return self


class CutInEvent(BaseModel):
    """進入前方 corridor 的可解釋 image-space 切入事件。"""

    model_config = ConfigDict(frozen=True)
    event_type: Literal["cut_in"] = "cut_in"
    event_id: str = Field(min_length=1)
    lane_change_event_id: str = Field(min_length=1)
    status: EventStatus
    track_id: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    anchor: Point2D
    corridor_interaction: bool
    bbox_expansion_ratio: float | None = None
    confidence: ConfidenceBreakdown
    evidence: EventEvidence
    reason: str | None = None

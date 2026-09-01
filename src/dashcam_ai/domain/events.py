"""可序列化的一般換道事件與影像空間證據模型。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import EgoMotionStatus, RelativeMotionEvidence, RelativeMotionSummary
from dashcam_ai.domain.temporal import LaneChangeDirection


class EventStatus(StrEnum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class EventEvidenceFrame(BaseModel):
    model_config = ConfigDict(frozen=True)
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    membership: LaneMembership
    lane_id: str | None = None
    stable_lane_id: str | None = None
    anchor: Point2D
    signed_boundary_distance: float | None = None
    smoothed_signed_boundary_distance: float | None = None
    ego_motion_status: EgoMotionStatus
    relative_motion: RelativeMotionEvidence | None = None


class EventEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)
    boundary_id: str | None = None
    frame_ids: tuple[int, ...] = ()
    trajectory: tuple[Point2D, ...] = ()
    frames: tuple[EventEvidenceFrame, ...] = ()


class ConfidenceBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)
    timeline: float = Field(ge=0, le=1)
    membership_stability: float = Field(ge=0, le=1)
    ego_motion: float = Field(ge=0, le=1)
    relative_motion: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class LaneChangeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    event_type: Literal["lane_change"] = "lane_change"
    event_id: str = Field(min_length=1)
    status: EventStatus
    track_id: int = Field(ge=0)
    source_lane: str = Field(min_length=1)
    target_lane: str = Field(min_length=1)
    direction: LaneChangeDirection
    started_frame: int = Field(ge=0)
    started_at: float = Field(ge=0)
    lane_crossed_frame: int | None = Field(default=None, ge=0)
    lane_crossed_at: float | None = Field(default=None, ge=0)
    completed_frame: int | None = Field(default=None, ge=0)
    completed_at: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    confidence_breakdown: ConfidenceBreakdown
    relative_motion: RelativeMotionSummary | None = None
    evidence: EventEvidence
    reason: str | None = None

    @model_validator(mode="after")
    def validate_timeline(self) -> LaneChangeEvent:
        if (self.lane_crossed_frame is None) != (self.lane_crossed_at is None):
            raise ValueError("lane crossing frame and timestamp must be provided together")
        if (self.completed_frame is None) != (self.completed_at is None):
            raise ValueError("completion frame and timestamp must be provided together")
        if self.lane_crossed_frame is not None:
            assert self.lane_crossed_at is not None
            if (
                self.lane_crossed_frame < self.started_frame
                or self.lane_crossed_at < self.started_at
            ):
                raise ValueError("lane crossing must not precede event start")
        if self.completed_frame is not None:
            assert self.completed_at is not None
            if self.lane_crossed_frame is None:
                raise ValueError("completion requires lane crossing evidence")
            assert self.lane_crossed_at is not None
            if (
                self.completed_frame < self.lane_crossed_frame
                or self.completed_at < self.lane_crossed_at
            ):
                raise ValueError("completion must not precede lane crossing")
        if self.status is EventStatus.CONFIRMED and self.completed_frame is None:
            raise ValueError("confirmed event requires completion evidence")
        return self

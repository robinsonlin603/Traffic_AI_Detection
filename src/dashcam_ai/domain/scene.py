"""單幀場景分析、track 車道狀態及事件快照。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.events import LaneChangeEvent
from dashcam_ai.domain.lane import LaneGeometry, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionEstimate
from dashcam_ai.domain.temporal import TemporalLaneState


class TrackSceneAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)
    track_id: int = Field(ge=0)
    membership: LaneMembershipFeature
    temporal: TemporalLaneState


class FrameSceneAnalysis(BaseModel):
    """附加於 FrameRecord 的 optional Milestone 2 分析欄位。"""

    model_config = ConfigDict(frozen=True)
    lane_geometry: LaneGeometry
    ego_motion: EgoMotionEstimate
    tracks: tuple[TrackSceneAnalysis, ...] = ()
    lane_change_events: tuple[LaneChangeEvent, ...] = ()

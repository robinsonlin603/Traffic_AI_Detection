"""單幀場景分析、track 車道狀態及事件快照。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.events import CutInEvent, LaneChangeEvent
from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import LaneGeometry, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionEstimate
from dashcam_ai.domain.temporal import TemporalLaneState


def _contains(point: Point2D, polygon: tuple[Point2D, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        cross_product = (point.x - previous.x) * (current.y - previous.y) - (
            point.y - previous.y
        ) * (current.x - previous.x)
        if (
            abs(cross_product) <= 1e-9
            and min(previous.x, current.x) <= point.x <= max(previous.x, current.x)
            and min(previous.y, current.y) <= point.y <= max(previous.y, current.y)
        ):
            return True
        crosses_y = (current.y > point.y) != (previous.y > point.y)
        if crosses_y:
            crossing_x = (previous.x - current.x) * (point.y - current.y) / (
                previous.y - current.y
            ) + current.x
            if point.x < crossing_x:
                inside = not inside
        previous = current
    return inside


class ForwardCorridor(BaseModel):
    """原始影像座標中的前方互動區域。"""

    model_config = ConfigDict(frozen=True)
    polygon: tuple[Point2D, ...] = Field(min_length=3)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)

    def contains(self, point: Point2D) -> bool:
        return _contains(point, self.polygon)

    def interaction_score(self, bbox: BBox) -> float:
        if self.contains(bbox.bottom_center):
            return 1.0
        if self.contains(bbox.center):
            return 0.5
        return 0.0


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
    forward_corridor: ForwardCorridor
    tracks: tuple[TrackSceneAnalysis, ...] = ()
    lane_change_events: tuple[LaneChangeEvent, ...] = ()
    cut_in_events: tuple[CutInEvent, ...] = ()

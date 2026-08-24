"""定義偵測、逐幀追蹤物件及完整軌跡的領域模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.geometry import BBox, Point2D


class Detection(BaseModel):
    """尚未分配追蹤 ID 的單幀物件偵測結果。"""
    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BBox


class TrackedObject(BaseModel):
    """已分配持續性追蹤 ID 的單幀物件。"""
    model_config = ConfigDict(frozen=True)

    track_id: int = Field(ge=0)
    class_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BBox


class TrackObservation(BaseModel):
    """特定追蹤物件在某個時間點的位置與信心分數。"""
    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    center: Point2D
    bottom_center: Point2D

    @classmethod
    def from_tracked_object(
        cls, obj: TrackedObject, frame_id: int, timestamp: float
    ) -> TrackObservation:
        """由單幀追蹤物件建立可長期保存的觀察紀錄。"""
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            confidence=obj.confidence,
            bbox=obj.bbox,
            center=obj.bbox.center,
            bottom_center=obj.bbox.bottom_center,
        )


class Track(BaseModel):
    """同一物件從首次到最後出現的完整跨幀軌跡。"""
    track_id: int = Field(ge=0)
    class_name: str = Field(min_length=1)
    first_seen: float = Field(ge=0)
    last_seen: float = Field(ge=0)
    observations: list[TrackObservation] = Field(default_factory=list)

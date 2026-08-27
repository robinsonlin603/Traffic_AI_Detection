"""定義影片中繼資料與逐幀分析結果。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.runtime import RuntimeMetadata
from dashcam_ai.domain.scene import FrameSceneAnalysis


class VideoMetadata(BaseModel):
    """來源影片的尺寸、幀率、幀數及長度。"""
    model_config = ConfigDict(frozen=True)

    source: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    runtime: RuntimeMetadata | None = None


class FrameRecord(BaseModel):
    """單一影格的時間資訊及其中所有追蹤物件。"""
    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    objects: list[TrackedObject] = Field(default_factory=list)
    analysis: FrameSceneAnalysis | None = None

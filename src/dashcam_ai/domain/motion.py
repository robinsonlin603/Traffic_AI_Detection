"""可序列化的自車相機運動估算模型。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EgoMotionStatus(StrEnum):
    """單次相鄰影格運動估算的可用狀態。"""

    VALID = "valid"
    UNKNOWN = "unknown"


class HomographyTransform(BaseModel):
    """以前一影格像素映射至目前影格像素的 3×3 homography。"""

    model_config = ConfigDict(frozen=True)
    values: tuple[float, float, float, float, float, float, float, float, float]


class EgoMotionQuality(BaseModel):
    """背景特徵追蹤及 RANSAC 解的可解釋品質指標。"""

    model_config = ConfigDict(frozen=True)
    detected_features: int = Field(ge=0)
    tracked_features: int = Field(ge=0)
    inlier_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    mean_reprojection_error: float | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)


class EgoMotionEstimate(BaseModel):
    """相鄰原始影格間的相機運動與品質結果。"""

    model_config = ConfigDict(frozen=True)
    status: EgoMotionStatus
    transform: HomographyTransform | None = None
    quality: EgoMotionQuality
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> EgoMotionEstimate:
        if self.status is EgoMotionStatus.VALID and self.transform is None:
            raise ValueError("valid ego-motion estimate requires a transform")
        if self.status is EgoMotionStatus.UNKNOWN and self.transform is not None:
            raise ValueError("unknown ego-motion estimate cannot contain a transform")
        return self

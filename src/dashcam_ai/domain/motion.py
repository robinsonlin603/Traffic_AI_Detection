"""可序列化的自車相機運動估算模型。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D


class EgoMotionStatus(StrEnum):
    """單次相鄰影格運動估算的可用狀態。"""

    VALID = "valid"
    UNKNOWN = "unknown"


class RelativeMotionStatus(StrEnum):
    """Track 錨點扣除背景 homography 後的可用狀態。"""

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


class RelativeMotionEvidence(BaseModel):
    """相鄰影格間可解釋且解析度正規化的 Track 相對運動。"""

    model_config = ConfigDict(frozen=True)
    status: RelativeMotionStatus
    previous_anchor: Point2D | None = None
    current_anchor: Point2D
    predicted_background_anchor: Point2D | None = None
    observed_displacement: Point2D | None = None
    predicted_background_displacement: Point2D | None = None
    compensated_displacement: Point2D | None = None
    normalized_lateral_displacement: float | None = None
    normalized_longitudinal_displacement: float | None = None
    stationary: bool | None = None
    scene_consistent: bool | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> RelativeMotionEvidence:
        required = (
            self.previous_anchor,
            self.predicted_background_anchor,
            self.observed_displacement,
            self.predicted_background_displacement,
            self.compensated_displacement,
            self.normalized_lateral_displacement,
            self.normalized_longitudinal_displacement,
            self.stationary,
            self.scene_consistent,
        )
        if self.status is RelativeMotionStatus.VALID and any(
            value is None for value in required
        ):
            raise ValueError("valid relative motion requires complete displacement evidence")
        if self.status is RelativeMotionStatus.UNKNOWN and any(
            value is not None for value in required
        ):
            raise ValueError("unknown relative motion cannot contain displacement evidence")
        return self


class RelativeMotionSummary(BaseModel):
    """一次換道候選期間的累積相對運動與安全閘門結果。"""

    model_config = ConfigDict(frozen=True)
    valid_observations: int = Field(ge=0)
    cumulative_lateral_displacement: float
    expected_lateral_progress: float
    directional_consistency: float = Field(ge=0, le=1)
    motion_quality: float = Field(ge=0, le=1)
    scene_consistency: float = Field(ge=0, le=1)
    stationary_ratio: float = Field(ge=0, le=1)
    supported: bool
    confidence: float = Field(ge=0, le=1)
    reason: str | None = None

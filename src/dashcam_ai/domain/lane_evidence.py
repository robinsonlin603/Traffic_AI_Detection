"""單幀動態車道 evidence 的可序列化模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneGeometryStatus


class LaneCurveEvidence(BaseModel):
    """由影像證據擬合的一條候選車道曲線。"""

    model_config = ConfigDict(frozen=True)
    evidence_id: str = Field(min_length=1)
    points: tuple[Point2D, ...] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)
    supporting_segments: int = Field(gt=0)


class LaneEvidenceFrame(BaseModel):
    """單幀候選曲線與品質；尚未宣稱穩定 LaneGeometry。"""

    model_config = ConfigDict(frozen=True)
    status: LaneGeometryStatus
    backend: str = Field(min_length=1)
    weight_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    confidence: float = Field(ge=0, le=1)
    curves: tuple[LaneCurveEvidence, ...] = ()
    estimated_lane_count: int | None = Field(default=None, gt=0)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_quality_payload(self) -> LaneEvidenceFrame:
        if self.status is LaneGeometryStatus.VALID and not self.curves:
            raise ValueError("valid lane evidence requires candidate curves")
        if self.status is LaneGeometryStatus.DEGRADED and not self.reason:
            raise ValueError("degraded lane evidence requires a reason")
        if self.status is LaneGeometryStatus.UNKNOWN:
            if self.curves or self.confidence != 0:
                raise ValueError("unknown lane evidence cannot contain curves or confidence")
            if not self.reason:
                raise ValueError("unknown lane evidence requires a reason")
        return self

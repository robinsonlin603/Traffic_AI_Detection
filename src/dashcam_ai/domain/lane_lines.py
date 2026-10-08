"""定義 Milestone 2 動態道路白線的逐幀輸出。"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D


class LaneLineStatus(StrEnum):
    """白線證據在目前影格的可用程度。"""

    VALID = "valid"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class LaneCurve(BaseModel):
    """影像座標中的一條左右道路白線曲線。"""

    model_config = ConfigDict(frozen=True)

    boundary_id: str = Field(min_length=1)
    points: tuple[Point2D, ...] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)
    carried_frames: int = Field(default=0, ge=0)
    confirmed_frames: int = Field(default=0, ge=0)
    lane_probability: float | None = Field(default=None, ge=0, le=1)
    drivable_probability: float | None = Field(default=None, ge=0, le=1)
    fit_error_ratio: float | None = Field(default=None, ge=0)
    component_area_ratio: float | None = Field(default=None, ge=0, le=1)
    component_width_ratio: float | None = Field(default=None, ge=0, le=1)
    component_height_ratio: float | None = Field(default=None, ge=0, le=1)
    median_row_width_ratio: float | None = Field(default=None, ge=0, le=1)
    maximum_row_width_ratio: float | None = Field(default=None, ge=0, le=1)
    row_width_variation_ratio: float | None = Field(default=None, ge=0)


class LaneCandidateDecision(BaseModel):
    """Locate a rejected or trimmed curve and the stage responsible."""

    model_config = ConfigDict(frozen=True)

    stage: Literal["context", "merge", "length", "confirmation", "carry"]
    reason: str
    curve: LaneCurve


class LaneLineDiagnostics(BaseModel):
    """單幀候選形成與淘汰原因，供實拍調校使用。"""

    model_config = ConfigDict(frozen=True)

    lane_pixel_count: int = Field(default=0, ge=0)
    white_supported_pixel_count: int = Field(default=0, ge=0)
    vehicle_excluded_pixel_count: int = Field(default=0, ge=0)
    component_count: int = Field(default=0, ge=0)
    accepted_component_count: int = Field(default=0, ge=0)
    merged_candidate_count: int = Field(default=0, ge=0)
    deduplicated_candidate_count: int = Field(default=0, ge=0)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    candidate_decisions: tuple[LaneCandidateDecision, ...] = ()


class LaneLineFrame(BaseModel):
    """單一影格的白線曲線與品質狀態。"""

    model_config = ConfigDict(frozen=True)

    status: LaneLineStatus
    curves: tuple[LaneCurve, ...] = ()
    reason: str | None = None
    diagnostics: LaneLineDiagnostics = Field(default_factory=LaneLineDiagnostics)

    @model_validator(mode="after")
    def validate_payload(self) -> LaneLineFrame:
        boundary_ids = [curve.boundary_id for curve in self.curves]
        if len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError("lane line frame cannot contain duplicate boundary IDs")
        if self.status is LaneLineStatus.VALID and len(self.curves) < 2:
            raise ValueError("valid lane line frame requires at least two curves")
        if self.status is LaneLineStatus.UNKNOWN and self.curves:
            raise ValueError("unknown lane line frame cannot contain curves")
        if self.status is not LaneLineStatus.VALID and not self.reason:
            raise ValueError("non-valid lane line frame requires a reason")
        return self


class LaneLineRecord(BaseModel):
    """可寫入 JSONL 的逐幀白線結果。"""

    model_config = ConfigDict(frozen=True)

    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    lane_lines: LaneLineFrame

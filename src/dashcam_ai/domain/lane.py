"""可序列化的車道幾何與車道歸屬領域模型。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D


class LaneGeometryStatus(StrEnum):
    VALID = "valid"
    UNKNOWN = "unknown"


class LaneGeometryProvenance(StrEnum):
    CONFIGURED = "configured"
    UNKNOWN = "unknown"


class LaneMembership(StrEnum):
    OUTSIDE = "outside"
    BOUNDARY = "boundary"
    INSIDE = "inside"
    UNKNOWN = "unknown"


class NormalizedPoint2D(BaseModel):
    """以影像寬高比例表示、範圍為零到一的座標。"""

    model_config = ConfigDict(frozen=True)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)

    def to_original(self, width: int, height: int) -> Point2D:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Point2D(x=self.x * width, y=self.y * height)


class LaneBoundary(BaseModel):
    model_config = ConfigDict(frozen=True)
    boundary_id: str = Field(min_length=1)
    points: list[Point2D] = Field(min_length=2)


class LaneRegion(BaseModel):
    model_config = ConfigDict(frozen=True)
    region_id: str = Field(min_length=1)
    polygon: list[Point2D] = Field(min_length=3)


class LaneGeometry(BaseModel):
    """單幀可用的自車車道區域、邊界及品質資訊。"""

    model_config = ConfigDict(frozen=True)
    status: LaneGeometryStatus
    provenance: LaneGeometryProvenance
    confidence: float = Field(ge=0, le=1)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    ego_lane: LaneRegion | None = None
    boundaries: list[LaneBoundary] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> LaneGeometry:
        if self.status is LaneGeometryStatus.VALID and self.ego_lane is None:
            raise ValueError("valid lane geometry requires an ego lane region")
        if self.status is LaneGeometryStatus.UNKNOWN and self.ego_lane is not None:
            raise ValueError("unknown lane geometry cannot contain an ego lane region")
        return self


class LaneMembershipFeature(BaseModel):
    """單一錨點的車道歸屬與穩定幾何特徵。"""

    model_config = ConfigDict(frozen=True)
    membership: LaneMembership
    anchor: Point2D
    signed_boundary_distance: float | None = None
    nearest_boundary_id: str | None = None
    geometry_confidence: float = Field(ge=0, le=1)

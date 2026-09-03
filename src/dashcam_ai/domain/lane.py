"""可序列化的車道幾何與車道歸屬領域模型。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashcam_ai.domain.geometry import Point2D


class LaneGeometryStatus(StrEnum):
    VALID = "valid"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class LaneGeometryProvenance(StrEnum):
    CONFIGURED = "configured"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class LaneBoundaryEvidenceSource(StrEnum):
    """描述單幀邊界來自直接觀察、時間推定或設定。"""

    CONFIGURED = "configured"
    OBSERVED = "observed"
    INFERRED = "inferred"
    CONFIGURED_FALLBACK = "configured_fallback"


class LaneMembership(StrEnum):
    OUTSIDE_CONFIGURED_LANES = "outside_configured_lanes"
    NEAR_BOUNDARY = "near_boundary"
    INSIDE_LANE = "inside_lane"
    UNKNOWN = "unknown"
    OUTSIDE = "outside_configured_lanes"
    BOUNDARY = "near_boundary"
    INSIDE = "inside_lane"


class NormalizedPoint2D(BaseModel):
    """以影像寬高比例表示、範圍為零到一的座標。"""

    model_config = ConfigDict(frozen=True)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)

    def to_original(self, width: int, height: int) -> Point2D:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return Point2D(x=self.x * width, y=self.y * height)


class NormalizedLaneRegion(BaseModel):
    """與影片解析度無關的 configured lane polygon。"""

    model_config = ConfigDict(frozen=True, populate_by_name=True)
    lane_id: str = Field(min_length=1, validation_alias="id", serialization_alias="id")
    lateral_order: int
    polygon: tuple[NormalizedPoint2D, ...] = Field(min_length=3)


class NormalizedLaneBoundary(BaseModel):
    """分隔兩條 configured lane 的 normalized polyline。"""

    model_config = ConfigDict(frozen=True, populate_by_name=True)
    boundary_id: str = Field(
        min_length=1, validation_alias="id", serialization_alias="id"
    )
    left_lane_id: str = Field(min_length=1)
    right_lane_id: str = Field(min_length=1)
    points: tuple[NormalizedPoint2D, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_distinct_lanes(self) -> NormalizedLaneBoundary:
        if self.left_lane_id == self.right_lane_id:
            raise ValueError("lane boundary must separate two distinct lanes")
        return self


class LaneBoundary(BaseModel):
    model_config = ConfigDict(frozen=True)
    boundary_id: str = Field(min_length=1)
    left_lane_id: str = Field(min_length=1)
    right_lane_id: str = Field(min_length=1)
    points: tuple[Point2D, ...] = Field(min_length=2)
    confidence: float = Field(default=1.0, ge=0, le=1)
    evidence_source: LaneBoundaryEvidenceSource = LaneBoundaryEvidenceSource.CONFIGURED

    @model_validator(mode="after")
    def validate_distinct_lanes(self) -> LaneBoundary:
        if self.left_lane_id == self.right_lane_id:
            raise ValueError("lane boundary must separate two distinct lanes")
        return self


class LaneRegion(BaseModel):
    model_config = ConfigDict(frozen=True)
    lane_id: str = Field(min_length=1)
    lateral_order: int
    polygon: tuple[Point2D, ...] = Field(min_length=3)


class LaneGeometry(BaseModel):
    """單幀可用的一般車道區域、共享邊界及品質資訊。"""

    model_config = ConfigDict(frozen=True)
    status: LaneGeometryStatus
    provenance: LaneGeometryProvenance
    confidence: float = Field(ge=0, le=1)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    lanes: tuple[LaneRegion, ...] = ()
    boundaries: tuple[LaneBoundary, ...] = ()
    topology_id: str | None = None
    topology_version: int | None = Field(default=None, ge=1)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> LaneGeometry:
        if (self.topology_id is None) != (self.topology_version is None):
            raise ValueError("topology ID and version must be provided together")
        if self.status is LaneGeometryStatus.VALID and not self.lanes:
            raise ValueError("valid lane geometry requires at least one lane region")
        if self.status is LaneGeometryStatus.DEGRADED and not self.reason:
            raise ValueError("degraded lane geometry requires a reason")
        if self.status is LaneGeometryStatus.UNKNOWN and (self.lanes or self.boundaries):
            raise ValueError("unknown lane geometry cannot contain lanes or boundaries")
        if self.status is LaneGeometryStatus.UNKNOWN and self.confidence != 0:
            raise ValueError("unknown lane geometry must have zero confidence")
        if self.status is LaneGeometryStatus.UNKNOWN and not self.reason:
            raise ValueError("unknown lane geometry requires a reason")
        if (
            self.provenance is LaneGeometryProvenance.UNKNOWN
            and self.status is not LaneGeometryStatus.UNKNOWN
        ):
            raise ValueError("unknown provenance requires unknown geometry status")
        if (
            self.status is LaneGeometryStatus.UNKNOWN
            and self.provenance is not LaneGeometryProvenance.UNKNOWN
        ):
            raise ValueError("unknown geometry requires unknown provenance")
        lane_ids = [lane.lane_id for lane in self.lanes]
        orders = [lane.lateral_order for lane in self.lanes]
        boundary_ids = [boundary.boundary_id for boundary in self.boundaries]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("lane geometry lane IDs must be unique")
        if len(set(orders)) != len(orders):
            raise ValueError("lane geometry lateral orders must be unique")
        if len(set(boundary_ids)) != len(boundary_ids):
            raise ValueError("lane geometry boundary IDs must be unique")
        known_lanes = set(lane_ids)
        for boundary in self.boundaries:
            if {
                boundary.left_lane_id,
                boundary.right_lane_id,
            } - known_lanes:
                raise ValueError("lane boundary references an unknown lane")
        return self


class LaneMembershipFeature(BaseModel):
    """單一錨點的車道歸屬與穩定幾何特徵。"""

    model_config = ConfigDict(frozen=True)
    membership: LaneMembership
    anchor: Point2D
    lane_id: str | None = None
    lane_lateral_order: int | None = None
    boundary_lane_ids: tuple[str, str] | None = None
    boundary_lane_orders: tuple[int, int] | None = None
    signed_boundary_distance: float | None = None
    nearest_boundary_id: str | None = None
    geometry_confidence: float = Field(ge=0, le=1)

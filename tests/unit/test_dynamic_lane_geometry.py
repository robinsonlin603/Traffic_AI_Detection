import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import DynamicLaneGeometryConfig
from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneBoundaryEvidenceSource,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
)


def lane() -> LaneRegion:
    return LaneRegion(
        lane_id="lane_0",
        lateral_order=0,
        polygon=(Point2D(x=0, y=0), Point2D(x=10, y=0), Point2D(x=10, y=10)),
    )


def test_degraded_dynamic_geometry_requires_an_explicit_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        LaneGeometry(
            status=LaneGeometryStatus.DEGRADED,
            provenance=LaneGeometryProvenance.DYNAMIC,
            confidence=0.45,
            frame_width=10,
            frame_height=10,
            lanes=(lane(),),
        )


def test_degraded_hybrid_geometry_can_carry_fallback_boundary() -> None:
    boundary = LaneBoundary(
        boundary_id="boundary_0",
        left_lane_id="lane_0",
        right_lane_id="lane_1",
        points=(Point2D(x=5, y=0), Point2D(x=5, y=10)),
        confidence=0.4,
        evidence_source=LaneBoundaryEvidenceSource.CONFIGURED_FALLBACK,
    )
    right_lane = lane().model_copy(update={"lane_id": "lane_1", "lateral_order": 1})

    geometry = LaneGeometry(
        status=LaneGeometryStatus.DEGRADED,
        provenance=LaneGeometryProvenance.HYBRID,
        confidence=0.4,
        frame_width=10,
        frame_height=10,
        lanes=(lane(), right_lane),
        boundaries=(boundary,),
        reason="dynamic evidence missing; configured fallback retained",
    )

    assert geometry.status is LaneGeometryStatus.DEGRADED
    assert geometry.boundaries[0].evidence_source is (
        LaneBoundaryEvidenceSource.CONFIGURED_FALLBACK
    )


def test_unknown_geometry_requires_zero_confidence_reason_and_provenance() -> None:
    with pytest.raises(ValidationError, match="zero confidence"):
        LaneGeometry(
            status=LaneGeometryStatus.UNKNOWN,
            provenance=LaneGeometryProvenance.UNKNOWN,
            confidence=0.1,
            frame_width=10,
            frame_height=10,
            reason="no lane evidence",
        )


def test_topology_identity_requires_matching_version() -> None:
    with pytest.raises(ValidationError, match="topology ID and version"):
        LaneGeometry(
            status=LaneGeometryStatus.DEGRADED,
            provenance=LaneGeometryProvenance.DYNAMIC,
            confidence=0.5,
            frame_width=10,
            frame_height=10,
            lanes=(lane(),),
            topology_id="topology-1",
            reason="pending",
        )
    with pytest.raises(ValidationError, match="unknown provenance"):
        LaneGeometry(
            status=LaneGeometryStatus.UNKNOWN,
            provenance=LaneGeometryProvenance.DYNAMIC,
            confidence=0,
            frame_width=10,
            frame_height=10,
            reason="no lane evidence",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("smoothing_alpha", 0),
        ("maximum_boundary_jump_ratio", 1.1),
        ("topology_confirmation_frames", 0),
        ("maximum_lane_count", 1),
        ("minimum_boundary_separation_ratio", 1.0),
    ],
)
def test_dynamic_configuration_rejects_unsafe_thresholds(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        DynamicLaneGeometryConfig.model_validate({field: value})


def test_dynamic_configuration_rejects_inverted_canny_thresholds() -> None:
    with pytest.raises(ValidationError, match="Canny|canny"):
        DynamicLaneGeometryConfig(
            canny_low_threshold=180,
            canny_high_threshold=100,
        )

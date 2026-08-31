import pytest

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneMembership,
    LaneRegion,
)
from dashcam_ai.lane.membership import LaneMembershipEvaluator


def geometry(*, confidence: float = 1.0, overlap: bool = False) -> LaneGeometry:
    from dashcam_ai.domain.lane import LaneBoundary

    left = LaneRegion(
        lane_id="lane_left",
        lateral_order=0,
        polygon=(
            Point2D(x=0, y=0),
            Point2D(x=50, y=0),
            Point2D(x=50, y=100),
            Point2D(x=0, y=100),
        ),
    )
    center_left = 40 if overlap else 50
    center = LaneRegion(
        lane_id="lane_center",
        lateral_order=1,
        polygon=(
            Point2D(x=center_left, y=0),
            Point2D(x=100, y=0),
            Point2D(x=100, y=100),
            Point2D(x=center_left, y=100),
        ),
    )
    boundary = LaneBoundary(
        boundary_id="boundary_left",
        left_lane_id="lane_left",
        right_lane_id="lane_center",
        points=(Point2D(x=50, y=0), Point2D(x=50, y=100)),
    )
    return LaneGeometry(
        status=LaneGeometryStatus.VALID,
        provenance=LaneGeometryProvenance.CONFIGURED,
        confidence=confidence,
        frame_width=120,
        frame_height=100,
        lanes=(left, center),
        boundaries=(boundary,),
    )


@pytest.mark.parametrize(
    ("anchor", "lane_id"),
    [(Point2D(x=20, y=50), "lane_left"), (Point2D(x=80, y=50), "lane_center")],
)
def test_inside_lane_returns_general_lane_id(anchor: Point2D, lane_id: str) -> None:
    result = LaneMembershipEvaluator(5).evaluate(anchor, geometry())
    assert result.membership is LaneMembership.INSIDE_LANE
    assert result.lane_id == lane_id
    assert result.signed_boundary_distance is not None
    assert result.signed_boundary_distance > 0


def test_near_boundary_exposes_both_lane_candidates() -> None:
    result = LaneMembershipEvaluator(5).evaluate(Point2D(x=48, y=50), geometry())
    assert result.membership is LaneMembership.NEAR_BOUNDARY
    assert result.nearest_boundary_id == "boundary_left"
    assert result.boundary_lane_ids == ("lane_left", "lane_center")


def test_outside_configured_lanes_has_negative_distance() -> None:
    result = LaneMembershipEvaluator(5).evaluate(Point2D(x=115, y=50), geometry())
    assert result.membership is LaneMembership.OUTSIDE_CONFIGURED_LANES
    assert result.lane_id is None
    assert result.signed_boundary_distance is not None
    assert result.signed_boundary_distance < 0


def test_overlapping_lane_polygons_return_unknown() -> None:
    result = LaneMembershipEvaluator(0).evaluate(Point2D(x=45, y=50), geometry(overlap=True))
    assert result.membership is LaneMembership.UNKNOWN
    assert result.lane_id is None


def test_low_confidence_geometry_returns_unknown() -> None:
    result = LaneMembershipEvaluator(5, minimum_geometry_confidence=0.8).evaluate(
        Point2D(x=20, y=50), geometry(confidence=0.7)
    )
    assert result.membership is LaneMembership.UNKNOWN
    assert result.signed_boundary_distance is None


def test_membership_threshold_validation() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        LaneMembershipEvaluator(5, minimum_geometry_confidence=1.1)

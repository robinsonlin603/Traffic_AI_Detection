import json

import pytest
from pydantic import ValidationError

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneMembership,
    NormalizedPoint2D,
)
from dashcam_ai.lane.configured import ConfiguredLaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator


def detector() -> ConfiguredLaneDetector:
    return ConfiguredLaneDetector(
        polygon=[
            NormalizedPoint2D(x=0.4, y=0.4),
            NormalizedPoint2D(x=0.6, y=0.4),
            NormalizedPoint2D(x=0.8, y=1.0),
            NormalizedPoint2D(x=0.2, y=1.0),
        ],
        confidence=0.9,
    )


def test_normalized_point_maps_to_original_resolution() -> None:
    point = NormalizedPoint2D(x=0.25, y=0.75)
    assert point.to_original(1920, 1080) == Point2D(x=480, y=810)
    assert point.to_original(1280, 720) == Point2D(x=320, y=540)


def test_normalized_point_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        NormalizedPoint2D(x=1.1, y=0.5)


def test_configured_detector_maps_polygon_and_boundaries() -> None:
    geometry = detector().detect(frame=None, width=1000, height=500)
    assert geometry.status is LaneGeometryStatus.VALID
    assert geometry.provenance is LaneGeometryProvenance.CONFIGURED
    assert geometry.confidence == 0.9
    assert geometry.ego_lane is not None
    assert geometry.ego_lane.polygon[0] == Point2D(x=400, y=200)
    assert geometry.ego_lane.polygon[2] == Point2D(x=800, y=500)
    assert [boundary.boundary_id for boundary in geometry.boundaries] == ["left", "right"]


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (Point2D(x=500, y=400), LaneMembership.INSIDE),
        (Point2D(x=250, y=400), LaneMembership.OUTSIDE),
        (Point2D(x=263, y=400), LaneMembership.BOUNDARY),
        (Point2D(x=271, y=400), LaneMembership.BOUNDARY),
    ],
)
def test_membership_uses_boundary_margin(anchor: Point2D, expected: LaneMembership) -> None:
    geometry = detector().detect(frame=None, width=1000, height=500)
    feature = LaneMembershipEvaluator(boundary_margin=8).evaluate(anchor, geometry)
    assert feature.membership is expected


def test_signed_distance_is_positive_inside_and_negative_outside() -> None:
    geometry = detector().detect(frame=None, width=1000, height=500)
    evaluator = LaneMembershipEvaluator(boundary_margin=0)
    inside = evaluator.evaluate(Point2D(x=500, y=400), geometry)
    outside = evaluator.evaluate(Point2D(x=250, y=400), geometry)
    assert inside.signed_boundary_distance is not None
    assert outside.signed_boundary_distance is not None
    assert inside.signed_boundary_distance > 0
    assert outside.signed_boundary_distance < 0


def test_unknown_geometry_produces_unknown_membership() -> None:
    geometry = LaneGeometry(
        status=LaneGeometryStatus.UNKNOWN,
        provenance=LaneGeometryProvenance.UNKNOWN,
        confidence=0,
        frame_width=1920,
        frame_height=1080,
        reason="calibration unavailable",
    )
    feature = LaneMembershipEvaluator(boundary_margin=10).evaluate(
        Point2D(x=960, y=900), geometry
    )
    assert feature.membership is LaneMembership.UNKNOWN
    assert feature.signed_boundary_distance is None


def test_lane_geometry_serialization_is_json_compatible() -> None:
    geometry = detector().detect(frame=None, width=1920, height=1080)
    payload = json.loads(geometry.model_dump_json())
    assert payload["status"] == "valid"
    assert payload["provenance"] == "configured"
    assert payload["ego_lane"]["polygon"][0] == {"x": 768.0, "y": 432.0}

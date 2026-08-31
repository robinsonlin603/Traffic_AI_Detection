import json

import pytest
from pydantic import ValidationError

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
    NormalizedLaneBoundary,
    NormalizedLaneRegion,
    NormalizedPoint2D,
)
from dashcam_ai.lane.configured import ConfiguredLaneDetector


def configured_lanes() -> list[NormalizedLaneRegion]:
    return [
        NormalizedLaneRegion(
            lane_id="lane_left",
            lateral_order=0,
            polygon=(
                NormalizedPoint2D(x=0.2, y=1.0),
                NormalizedPoint2D(x=0.4, y=0.4),
                NormalizedPoint2D(x=0.5, y=0.4),
                NormalizedPoint2D(x=0.5, y=1.0),
            ),
        ),
        NormalizedLaneRegion(
            lane_id="lane_center",
            lateral_order=1,
            polygon=(
                NormalizedPoint2D(x=0.5, y=0.4),
                NormalizedPoint2D(x=0.6, y=0.4),
                NormalizedPoint2D(x=0.8, y=1.0),
                NormalizedPoint2D(x=0.5, y=1.0),
            ),
        ),
    ]


def configured_boundaries() -> list[NormalizedLaneBoundary]:
    return [
        NormalizedLaneBoundary(
            boundary_id="boundary_left",
            left_lane_id="lane_left",
            right_lane_id="lane_center",
            points=(
                NormalizedPoint2D(x=0.5, y=0.4),
                NormalizedPoint2D(x=0.5, y=1.0),
            ),
        )
    ]


def detector() -> ConfiguredLaneDetector:
    return ConfiguredLaneDetector(configured_lanes(), configured_boundaries(), 0.9)


def test_normalized_point_maps_to_original_resolution() -> None:
    point = NormalizedPoint2D(x=0.25, y=0.75)
    assert point.to_original(1920, 1080) == Point2D(x=480, y=810)
    assert point.to_original(1280, 720) == Point2D(x=320, y=540)


def test_normalized_point_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        NormalizedPoint2D(x=1.1, y=0.5)


def test_configured_detector_maps_all_lanes_and_boundaries() -> None:
    geometry = detector().detect(frame=None, width=1000, height=500)
    assert geometry.status is LaneGeometryStatus.VALID
    assert geometry.provenance is LaneGeometryProvenance.CONFIGURED
    assert geometry.confidence == 0.9
    assert [lane.lane_id for lane in geometry.lanes] == ["lane_left", "lane_center"]
    assert geometry.lanes[1].polygon[0] == Point2D(x=500, y=200)
    assert geometry.lanes[1].polygon[2] == Point2D(x=800, y=500)
    assert geometry.boundaries[0].left_lane_id == "lane_left"
    assert geometry.boundaries[0].right_lane_id == "lane_center"
    assert geometry.boundaries[0].points[1] == Point2D(x=500, y=500)


def test_configured_detector_maps_at_a_second_resolution() -> None:
    geometry = detector().detect(frame=None, width=1920, height=1080)
    assert geometry.lanes[1].polygon[0] == Point2D(x=960, y=432)
    assert geometry.boundaries[0].points[1] == Point2D(x=960, y=1080)


def test_configured_detector_rejects_unknown_boundary_lane() -> None:
    boundary = configured_boundaries()[0].model_copy(
        update={"right_lane_id": "missing_lane"}
    )
    with pytest.raises(ValueError, match="unknown lane"):
        ConfiguredLaneDetector(configured_lanes(), [boundary])


def test_lane_geometry_rejects_duplicate_lane_ids_and_orders() -> None:
    lane = LaneRegion(
        lane_id="lane_a",
        lateral_order=0,
        polygon=(Point2D(x=0, y=0), Point2D(x=1, y=0), Point2D(x=1, y=1)),
    )
    with pytest.raises(ValidationError, match="lane IDs"):
        LaneGeometry(
            status=LaneGeometryStatus.VALID,
            provenance=LaneGeometryProvenance.CONFIGURED,
            confidence=1,
            frame_width=10,
            frame_height=10,
            lanes=(lane, lane),
        )
    other = lane.model_copy(update={"lane_id": "lane_b"})
    with pytest.raises(ValidationError, match="lateral orders"):
        LaneGeometry(
            status=LaneGeometryStatus.VALID,
            provenance=LaneGeometryProvenance.CONFIGURED,
            confidence=1,
            frame_width=10,
            frame_height=10,
            lanes=(lane, other),
        )


def test_lane_boundary_must_separate_distinct_lanes() -> None:
    with pytest.raises(ValidationError, match="distinct lanes"):
        LaneBoundary(
            boundary_id="bad",
            left_lane_id="lane_a",
            right_lane_id="lane_a",
            points=(Point2D(x=0, y=0), Point2D(x=1, y=1)),
        )


def test_unknown_geometry_cannot_contain_lanes() -> None:
    lane = LaneRegion(
        lane_id="lane_a",
        lateral_order=0,
        polygon=(Point2D(x=0, y=0), Point2D(x=1, y=0), Point2D(x=1, y=1)),
    )
    with pytest.raises(ValidationError, match="cannot contain"):
        LaneGeometry(
            status=LaneGeometryStatus.UNKNOWN,
            provenance=LaneGeometryProvenance.UNKNOWN,
            confidence=0,
            frame_width=10,
            frame_height=10,
            lanes=(lane,),
        )


def test_lane_geometry_serialization_is_json_compatible_without_ego_lane() -> None:
    payload = json.loads(detector().detect(None, 1920, 1080).model_dump_json())
    assert "ego_lane" not in payload
    assert payload["lanes"][1]["lane_id"] == "lane_center"
    assert payload["lanes"][1]["polygon"][0] == {"x": 960.0, "y": 432.0}
    assert payload["boundaries"][0]["boundary_id"] == "boundary_left"

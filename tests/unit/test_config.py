from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import (
    DynamicLaneGeometryConfig,
    LaneEvidenceBackendName,
    LaneGeometryConfig,
    LaneGeometryMode,
    load_config,
)
from dashcam_ai.domain.lane import (
    NormalizedLaneBoundary,
    NormalizedLaneRegion,
    NormalizedPoint2D,
)


def test_default_configuration_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.tracking.tracker == "botsort.yaml"
    assert config.tracking.minimum_track_length == 2
    assert config.detection.imgsz == 1280
    assert "car" in config.detection.classes
    assert [lane.lane_id for lane in config.lane_geometry.lanes] == [
        "lane_left",
        "lane_center",
        "lane_right",
    ]
    assert config.lane_geometry.mode is LaneGeometryMode.CONFIGURED
    assert config.lane_geometry.dynamic.minimum_confidence == 0.7
    assert config.lane_geometry.dynamic.topology_confirmation_frames == 6
    assert config.lane_geometry.dynamic.maximum_lane_count == 3
    assert config.lane_geometry.dynamic.minimum_boundary_separation_ratio == 0.08
    assert config.lane_geometry.dynamic.canny_low_threshold == 50
    assert config.lane_geometry.dynamic.canny_high_threshold == 150
    assert [lane.lateral_order for lane in config.lane_geometry.lanes] == [0, 1, 2]
    assert config.lane_geometry.lanes[1].polygon[0].x == 0.44
    assert len(config.lane_geometry.boundaries) == 2
    assert config.lane_membership.boundary_margin_pixels == 12.0
    assert config.lane_membership.minimum_geometry_confidence == 0.5
    assert config.ego_motion.minimum_inliers == 8
    assert config.ego_motion.optical_flow_window_size == 21
    assert config.relative_motion.enabled is True
    assert config.relative_motion.minimum_valid_observations == 2
    assert config.temporal_lane.smoothing_window_frames == 3
    assert config.temporal_lane.maximum_missing_frames == 2
    assert not hasattr(config, "forward_corridor")
    assert not hasattr(config, "cut_in")


@pytest.mark.parametrize(
    ("path", "device"),
    [
        (Path("configs/default.yaml"), "auto"),
        (Path("configs/mac.yaml"), "mps"),
        (Path("configs/nvidia.yaml"), "cuda:0"),
    ],
)
def test_platform_configurations_include_milestone_2_sections(path: Path, device: str) -> None:
    config = load_config(path)

    assert config.detection.device == device
    assert config.lane_geometry.enabled is True
    assert config.lane_geometry.mode is LaneGeometryMode.CONFIGURED
    assert len(config.lane_geometry.lanes) == 3
    assert len(config.lane_geometry.boundaries) == 2
    assert config.ego_motion.minimum_tracked_features >= 4
    assert config.relative_motion.minimum_cumulative_lateral_ratio > 0
    assert config.temporal_lane.minimum_confirmation_frames > 0


def test_nvidia_lane_calibration_stays_on_test3_road_surface() -> None:
    config = load_config(Path("configs/nvidia.yaml"))
    center_lane = next(
        lane for lane in config.lane_geometry.lanes if lane.lane_id == "lane_center"
    ).polygon

    assert min(point.y for point in center_lane) >= 0.66


def test_lane_geometry_configuration_rejects_duplicate_ids_and_unknown_refs() -> None:
    lane = NormalizedLaneRegion(
        lane_id="lane_a",
        lateral_order=0,
        polygon=(
            NormalizedPoint2D(x=0, y=0),
            NormalizedPoint2D(x=1, y=0),
            NormalizedPoint2D(x=1, y=1),
        ),
    )
    with pytest.raises(ValidationError, match="lane IDs"):
        LaneGeometryConfig(lanes=[lane, lane])

    boundary = NormalizedLaneBoundary(
        boundary_id="boundary_a",
        left_lane_id="lane_a",
        right_lane_id="missing",
        points=(NormalizedPoint2D(x=0, y=0), NormalizedPoint2D(x=0, y=1)),
    )
    with pytest.raises(ValidationError, match="unknown lane"):
        LaneGeometryConfig(lanes=[lane], boundaries=[boundary])


def test_dynamic_mode_allows_geometry_without_configured_lanes() -> None:
    config = LaneGeometryConfig(mode=LaneGeometryMode.DYNAMIC, lanes=[], boundaries=[])

    assert config.lanes == []
    assert len(config.dynamic.road_roi) == 4


def test_yolop_backend_requires_explicit_weight_path() -> None:
    with pytest.raises(ValidationError, match="weights path"):
        DynamicLaneGeometryConfig(backend=LaneEvidenceBackendName.YOLOP_ONNX)


@pytest.mark.parametrize("mode", [LaneGeometryMode.CONFIGURED, LaneGeometryMode.HYBRID])
def test_configured_and_hybrid_modes_require_configured_lanes(
    mode: LaneGeometryMode,
) -> None:
    with pytest.raises(ValidationError, match="requires configured lanes"):
        LaneGeometryConfig(mode=mode, lanes=[], boundaries=[])

from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import CutInConfig, LaneGeometryConfig, load_config
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
    assert len(config.forward_corridor.polygon) == 4
    assert config.cut_in.minimum_confirmed_confidence == 0.65


@pytest.mark.parametrize(
    ("path", "device"),
    [
        (Path("configs/default.yaml"), "auto"),
        (Path("configs/mac.yaml"), "mps"),
        (Path("configs/nvidia.yaml"), "cuda:0"),
    ],
)
def test_platform_configurations_include_milestone_2_sections(
    path: Path, device: str
) -> None:
    config = load_config(path)

    assert config.detection.device == device
    assert config.lane_geometry.enabled is True
    assert len(config.lane_geometry.lanes) == 3
    assert len(config.lane_geometry.boundaries) == 2
    assert config.ego_motion.minimum_tracked_features >= 4
    assert config.relative_motion.minimum_cumulative_lateral_ratio > 0
    assert config.temporal_lane.minimum_confirmation_frames > 0
    assert len(config.forward_corridor.polygon) >= 3
    assert config.cut_in.minimum_confirmed_confidence > 0


def test_nvidia_lane_calibration_stays_on_test3_road_surface() -> None:
    config = load_config(Path("configs/nvidia.yaml"))
    center_lane = next(
        lane for lane in config.lane_geometry.lanes if lane.lane_id == "lane_center"
    ).polygon
    corridor = config.forward_corridor.polygon

    assert min(point.y for point in center_lane) >= 0.66
    assert min(point.y for point in corridor) >= 0.70
    assert corridor[0].x > center_lane[0].x
    assert corridor[1].x < center_lane[1].x
    assert corridor[2].x < center_lane[2].x
    assert corridor[3].x > center_lane[3].x


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


def test_cutin_configuration_requires_positive_weight_sum() -> None:
    with pytest.raises(ValidationError, match="positive sum"):
        CutInConfig(
            lane_change_weight=0,
            corridor_weight=0,
            bbox_expansion_weight=0,
            motion_quality_weight=0,
            relative_motion_weight=0,
            lateral_progress_weight=0,
            direction_compatibility_weight=0,
            scene_consistency_weight=0,
        )

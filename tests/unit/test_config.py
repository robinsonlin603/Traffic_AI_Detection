from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import CutInConfig, load_config


def test_default_configuration_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.tracking.tracker == "botsort.yaml"
    assert config.tracking.minimum_track_length == 2
    assert config.detection.imgsz == 1280
    assert "car" in config.detection.classes
    assert len(config.lane_geometry.ego_lane_polygon) == 4
    assert config.lane_geometry.ego_lane_polygon[0].x == 0.44
    assert config.lane_membership.boundary_margin_pixels == 12.0
    assert config.ego_motion.minimum_inliers == 8
    assert config.ego_motion.optical_flow_window_size == 21
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
    assert len(config.lane_geometry.ego_lane_polygon) == 4
    assert config.ego_motion.minimum_tracked_features >= 4
    assert config.temporal_lane.minimum_confirmation_frames > 0
    assert len(config.forward_corridor.polygon) >= 3
    assert config.cut_in.minimum_confirmed_confidence > 0


def test_cutin_configuration_requires_positive_weight_sum() -> None:
    with pytest.raises(ValidationError, match="positive sum"):
        CutInConfig(
            lane_change_weight=0,
            corridor_weight=0,
            bbox_expansion_weight=0,
            motion_quality_weight=0,
        )

from pathlib import Path

from dashcam_ai.config.models import load_config


def test_default_configuration_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.tracking.tracker == "botsort.yaml"
    assert config.detection.imgsz == 1280
    assert "car" in config.detection.classes
    assert len(config.lane_geometry.ego_lane_polygon) == 4
    assert config.lane_geometry.ego_lane_polygon[0].x == 0.44
    assert config.lane_membership.boundary_margin_pixels == 12.0
    assert config.ego_motion.minimum_inliers == 8
    assert config.ego_motion.optical_flow_window_size == 21
    assert config.temporal_lane.smoothing_window_frames == 3
    assert config.temporal_lane.maximum_missing_frames == 2

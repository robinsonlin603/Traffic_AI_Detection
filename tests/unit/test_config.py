from pathlib import Path

import pytest

from dashcam_ai.config.models import load_config


@pytest.mark.parametrize(
    ("path", "device"),
    [("default", "auto"), ("mac", "mps"), ("nvidia", "cuda:0")],
)
def test_platform_perception_configuration(path: str, device: str) -> None:
    config = load_config(Path(f"configs/{path}.yaml"))
    assert config.detection.device == device
    assert config.detection.imgsz == 1280
    assert "car" in config.detection.classes
    assert set(config.detection.classes) == {"car", "motorcycle", "bus", "truck"}
    assert config.detection.minimum_vehicle_area_ratio == 0
    assert config.detection.duplicate_vehicle_iou_threshold == 0.85
    assert config.detection.duplicate_vehicle_containment_threshold == 0.9
    assert config.detection.duplicate_vehicle_center_distance_ratio == 0.2
    assert config.tracking.tracker == "botsort.yaml"
    assert config.tracking.minimum_track_length == 2
    assert set(config.model_dump()) == {"detection", "tracking", "output", "logging"}

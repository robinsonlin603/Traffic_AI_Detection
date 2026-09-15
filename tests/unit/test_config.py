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
    assert "person" not in config.detection.classes
    assert config.tracking.tracker == "botsort.yaml"
    assert config.tracking.minimum_track_length == 2
    assert set(config.model_dump()) == {"detection", "tracking", "output", "logging"}

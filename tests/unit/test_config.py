from pathlib import Path

from dashcam_ai.config.models import load_config


def test_default_configuration_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.tracking.tracker == "botsort.yaml"
    assert config.detection.imgsz == 1280
    assert "car" in config.detection.classes


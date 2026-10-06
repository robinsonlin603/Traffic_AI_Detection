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
    assert config.detection.duplicate_vehicle_center_distance_ratio == 0.22
    assert config.tracking.tracker == "botsort.yaml"
    assert config.tracking.minimum_track_length == 2
    assert config.lane_detection.backend == "yolop_onnx"
    assert config.lane_detection.model == "models/yolop-640-640.onnx"
    assert config.lane_detection.model_sha256 == (
        "cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696"
    )
    assert config.lane_detection.input_size == 640
    assert config.lane_detection.probability_threshold == 0.5
    assert config.lane_detection.maximum_horizontal_to_vertical_ratio == 5.0
    assert config.lane_detection.maximum_missing_frames == 1
    assert config.lane_detection.minimum_fragment_vertical_span_ratio == 0.05
    assert config.lane_detection.maximum_arrow_fit_error_ratio == 0.008
    assert config.lane_detection.maximum_row_width_variation_ratio == 3.5
    assert set(config.model_dump()) == {
        "detection",
        "tracking",
        "lane_detection",
        "output",
        "logging",
    }

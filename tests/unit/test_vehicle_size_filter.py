from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import DetectionConfig
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker


@pytest.mark.parametrize("ratio", [-0.001, 1.001, float("nan")])
def test_invalid_vehicle_area_ratio(ratio: float) -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(minimum_vehicle_area_ratio=ratio)


@pytest.mark.parametrize("scale", [1, 2])
def test_filter_boundary_resolution_and_non_vehicle_classes(scale: int) -> None:
    torch = pytest.importorskip("torch")
    Results = pytest.importorskip("ultralytics.engine.results").Results
    backend = object.__new__(UltralyticsDetectorTracker)
    backend._minimum_vehicle_area_ratio = 0.001
    backend._vehicle_class_ids = {2, 3, 5, 7}
    # A 1000x1000 image has a 1000-pixel area threshold. Exact boundary survives.
    rows = [[0, 0, 20, 49, 0.9, cls] for cls in (2, 3, 5, 7)]
    rows += [[0, 0, 20, 50, 0.9, 2], [0, 0, 2, 2, 0.9, 0], [0, 0, 2, 2, 0.9, 1]]
    data = torch.tensor(rows)
    data[:, :4] *= scale
    result = Results(
        orig_img=np.zeros((1000 * scale, 1000 * scale, 3), dtype=np.uint8),
        path="test", names={}, boxes=data,
    )
    predictor = SimpleNamespace(results=[result])
    backend._filter_small_vehicles(predictor)
    assert predictor.results[0].boxes.cls.tolist() == [2, 0, 1]
    assert predictor.results[0].boxes.id is None  # Still raw detections, before tracking.
    backend._minimum_vehicle_area_ratio = 0
    predictor.results = [result]
    backend._filter_small_vehicles(predictor)
    assert len(predictor.results[0].boxes) == 7


def test_all_small_and_empty_results() -> None:
    torch = pytest.importorskip("torch")
    Results = pytest.importorskip("ultralytics.engine.results").Results
    backend = object.__new__(UltralyticsDetectorTracker)
    backend._minimum_vehicle_area_ratio = 0.001
    backend._vehicle_class_ids = {2, 3, 5, 7}
    result = Results(
        orig_img=np.zeros((1000, 1000, 3), dtype=np.uint8),
        path="test", names={}, boxes=torch.tensor([[0, 0, 2, 2, 0.9, 2]]),
    )
    predictor = SimpleNamespace(results=[result])
    backend._filter_small_vehicles(predictor)
    assert len(predictor.results[0].boxes) == 0
    backend._filter_small_vehicles(predictor)
    assert len(predictor.results[0].boxes) == 0

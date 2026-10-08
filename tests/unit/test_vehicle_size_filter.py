from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from dashcam_ai.config.models import DetectionConfig
from dashcam_ai.detection.ego_mask import EgoVehicleMask
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker


def backend(
    *,
    area_ratio: float = 0,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.9,
    center_distance_ratio: float = 0.22,
):
    value = object.__new__(UltralyticsDetectorTracker)
    value._minimum_vehicle_area_ratio = area_ratio
    value._duplicate_vehicle_iou_threshold = iou_threshold
    value._duplicate_vehicle_containment_threshold = containment_threshold
    value._duplicate_vehicle_center_distance_ratio = center_distance_ratio
    value._vehicle_class_ids = {2, 3, 5, 7}
    value._vehicle_class_id = 80
    value._ego_vehicle_mask = EgoVehicleMask([])
    value._context_model = None
    value._context_boxes = None
    return value


def result(rows):
    torch = pytest.importorskip("torch")
    Results = pytest.importorskip("ultralytics.engine.results").Results
    return Results(
        orig_img=np.zeros((1000, 1000, 3), dtype=np.uint8),
        path="test",
        names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        boxes=torch.tensor(rows, dtype=torch.float32).reshape((-1, 6)),
    )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_vehicle_area_ratio",
        "duplicate_vehicle_iou_threshold",
        "duplicate_vehicle_containment_threshold",
        "duplicate_vehicle_center_distance_ratio",
    ],
)
@pytest.mark.parametrize("value", [-0.001, 1.001, float("nan")])
def test_invalid_detection_thresholds(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(**{field: value})


def test_cross_class_duplicates_keep_highest_confidence_and_map_to_vehicle() -> None:
    prediction = result(
        [
            [100, 100, 300, 300, 0.70, 2],
            [102, 102, 298, 298, 0.90, 7],
            [500, 500, 600, 600, 0.80, 3],
        ]
    )
    predictor = SimpleNamespace(results=[prediction])

    backend()._prepare_vehicle_detections(predictor)

    boxes = predictor.results[0].boxes
    assert boxes is not None
    assert boxes.conf.tolist() == pytest.approx([0.9, 0.8])
    assert boxes.cls.tolist() == [80, 80]
    assert predictor.results[0].names[80] == "vehicle"
    assert boxes.id is None  # Preparation occurs before BoT-SORT assigns IDs.



def test_nested_cross_class_duplicate_is_removed() -> None:
    prediction = result(
        [
            [100, 100, 300, 300, 0.80, 3],
            [110, 120, 290, 280, 0.70, 2],
        ]
    )
    predictor = SimpleNamespace(results=[prediction])

    backend()._prepare_vehicle_detections(predictor)

    boxes = predictor.results[0].boxes
    assert boxes is not None
    assert len(boxes) == 1
    assert boxes.conf.tolist() == pytest.approx([0.8])


def test_contained_vehicle_with_distant_center_is_not_removed() -> None:
    prediction = result(
        [
            [100, 100, 300, 300, 0.80, 7],
            [100, 100, 160, 160, 0.70, 3],
        ]
    )
    predictor = SimpleNamespace(results=[prediction])

    backend()._prepare_vehicle_detections(predictor)

    assert len(predictor.results[0].boxes) == 2

def test_non_overlapping_vehicles_are_not_merged() -> None:
    prediction = result(
        [[0, 0, 100, 100, 0.9, 2], [50, 0, 150, 100, 0.8, 3]]
    )
    predictor = SimpleNamespace(results=[prediction])

    backend()._prepare_vehicle_detections(predictor)

    assert len(predictor.results[0].boxes) == 2


def test_area_filter_is_disabled_by_default_but_remains_configurable() -> None:
    prediction = result([[0, 0, 2, 2, 0.9, 3]])
    predictor = SimpleNamespace(results=[prediction])
    backend()._prepare_vehicle_detections(predictor)
    assert len(predictor.results[0].boxes) == 1

    predictor.results = [prediction]
    backend(area_ratio=0.001)._prepare_vehicle_detections(predictor)
    assert len(predictor.results[0].boxes) == 0


def test_empty_results_are_unchanged() -> None:
    prediction = result([])
    predictor = SimpleNamespace(results=[prediction])
    backend()._prepare_vehicle_detections(predictor)
    assert len(predictor.results[0].boxes) == 0

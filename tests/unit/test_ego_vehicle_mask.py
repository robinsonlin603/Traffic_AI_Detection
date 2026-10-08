from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from test_vehicle_size_filter import backend, result

from dashcam_ai.config.models import DetectionConfig
from dashcam_ai.detection.ego_mask import EgoVehicleMask
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker

POLYGON = [(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)]


def test_mask_copy_preserves_original_and_pixels_outside_body() -> None:
    original = np.full((100, 200, 3), 37, dtype=np.uint8)
    mask = EgoVehicleMask(POLYGON)
    prepared = mask.prepare(original)
    assert np.all(original == 37)
    assert np.all(prepared[mask.mask((100, 200))] == 114)
    assert np.all(prepared[~mask.mask((100, 200))] == 37)


def test_mask_scales_and_rebuilds_when_resolution_changes() -> None:
    mask = EgoVehicleMask(POLYGON)
    assert mask.excludes([0, 60, 80, 99], (100, 200))
    assert mask.excludes([0, 120, 160, 198], (200, 400))
    assert not mask.excludes([100, 0, 200, 100], (100, 200))


def test_only_majority_overlap_is_excluded_and_clipped_boxes_are_safe() -> None:
    mask = EgoVehicleMask(POLYGON)
    assert mask.excludes([-20, 60, 80, 120], (100, 200))
    assert not mask.excludes([80, 40, 180, 90], (100, 200))
    assert not mask.excludes([200, 50, 200, 60], (100, 200))
    assert not mask.excludes([300, 300, 400, 400], (100, 200))


def test_empty_polygon_disables_preprocessing_and_filtering() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = EgoVehicleMask([])
    assert mask.prepare(frame) is frame
    assert not mask.excludes([0, 0, 100, 100], (100, 100))
    assert DetectionConfig().ego_vehicle_polygon == []


@pytest.mark.parametrize(
    "polygon",
    [
        [(0, 0), (1, 1)],
        [(0, 0), (0, 0), (1, 1)],
        [(0, 0), (0.5, 0.5), (1, 1)],
        [(0, 0), (1.01, 0), (0, 1)],
        [(0, 0), (float("nan"), 0), (0, 1)],
        [(0, 0), (float("inf"), 0), (0, 1)],
    ],
)
def test_invalid_polygons_are_rejected(polygon) -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(ego_vehicle_polygon=polygon)
    with pytest.raises(ValueError):
        EgoVehicleMask(polygon)


@pytest.mark.parametrize("threshold", [0, -1, 1.01, float("nan")])
def test_invalid_overlap_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(ego_vehicle_overlap_threshold=threshold)
    with pytest.raises(ValueError):
        EgoVehicleMask(POLYGON, threshold)


def test_self_detection_is_removed_before_id_allocation_but_neighbor_survives() -> None:
    prediction = result(
        [
            [0, 600, 400, 950, 0.95, 2],
            [400, 400, 800, 900, 0.80, 3],
        ]
    )
    predictor = SimpleNamespace(results=[prediction])
    tracker = backend()
    tracker._ego_vehicle_mask = EgoVehicleMask(POLYGON)
    tracker._prepare_vehicle_detections(predictor)
    boxes = predictor.results[0].boxes
    assert boxes is not None and boxes.id is None
    assert len(boxes) == 1
    assert boxes.xyxy.tolist() == [[400, 400, 800, 900]]


def test_process_passes_masked_copy_to_tracker_and_preserves_input() -> None:
    tracker = backend()
    tracker._ego_vehicle_mask = EgoVehicleMask(POLYGON)
    received = []
    tracker._model = SimpleNamespace(track=lambda **kw: received.append(kw["source"]) or [])
    tracker._identity_resolver = SimpleNamespace(update=lambda rows: rows)
    tracker._tracker = "botsort.yaml"
    tracker._confidence = 0.35
    tracker._imgsz = 1280
    tracker._allowed_class_ids = [2, 3]
    tracker._device = "cpu"
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert UltralyticsDetectorTracker.process(tracker, frame) == []
    assert np.all(frame == 0)
    assert received[0][80, 20, 0] == 114
    assert received[0][20, 160, 0] == 0


def test_outside_body_context_is_recovered_before_ids_without_restoring_self() -> None:
    # Gray input can lower an unrelated vehicle's score below the detection threshold.
    # Recover its original prediction, but never an original box touching the ego body.
    predictor = SimpleNamespace(results=[result([])])
    tracker = backend()
    tracker._ego_vehicle_mask = EgoVehicleMask(POLYGON)
    tracker._context_boxes = result(
        [
            [700, 400, 900, 700, 0.40, 3],
            [0, 550, 600, 1000, 0.90, 2],
        ]
    ).boxes.data
    tracker._prepare_vehicle_detections(predictor)
    boxes = predictor.results[0].boxes
    assert boxes.id is None
    assert boxes.xyxy.tolist() == [[700, 400, 900, 700]]


def test_context_and_masked_duplicates_keep_original_higher_confidence() -> None:
    predictor = SimpleNamespace(results=[result([[700, 400, 900, 700, 0.60, 3]])])
    tracker = backend()
    tracker._ego_vehicle_mask = EgoVehicleMask(POLYGON)
    tracker._context_boxes = result([[702, 402, 898, 698, 0.90, 3]]).boxes.data
    tracker._prepare_vehicle_detections(predictor)
    assert len(predictor.results[0].boxes) == 1
    assert predictor.results[0].boxes.conf.tolist() == pytest.approx([0.90])


def test_original_context_predictor_never_tracks_and_is_cleared_after_processing() -> None:
    tracker = backend()
    tracker._ego_vehicle_mask = EgoVehicleMask(POLYGON)
    received = []
    context = result([[700, 400, 900, 700, 0.40, 3]])
    tracker._context_model = SimpleNamespace(
        predict=lambda **kw: received.append(kw["source"]) or [context]
    )
    tracker._model = SimpleNamespace(track=lambda **kw: [])
    tracker._identity_resolver = SimpleNamespace(update=lambda rows: rows)
    tracker._tracker = "botsort.yaml"
    tracker._confidence = 0.35
    tracker._imgsz = 1280
    tracker._allowed_class_ids = [2, 3]
    tracker._device = "cpu"
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert tracker.process(frame) == []
    assert received[0] is frame
    assert tracker._context_boxes is None

"""Validate safe experiment behavior and the independence of quality measurements."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane_lines import LaneCurve
from dashcam_ai.lane.prototype import (
    ExperimentalLaneDetector,
    MarkingClassifier,
    VehicleInstance,
    curve_features,
    pixel_occlusion,
)
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector

_scripts = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "lane_prototype_review", _scripts / "evaluate_lane_prototype.py"
)
assert _spec is not None and _spec.loader is not None
review = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(_scripts))
try:
    _spec.loader.exec_module(review)
finally:
    sys.path.pop(0)


def _curve() -> LaneCurve:
    return LaneCurve(
        boundary_id="test", confidence=0.9, points=(Point2D(x=30, y=20), Point2D(x=30, y=70))
    )


def test_matched_pixel_mask_preserves_visible_road_inside_vehicle_box() -> None:
    box = BBox(x1=10, y1=10, x2=80, y2=90)
    rectangular = np.zeros((100, 100), np.uint8)
    rectangular[10:91, 10:81] = 255
    silhouette = np.zeros_like(rectangular)
    silhouette[15:80, 30:60] = 255
    result, replaced = pixel_occlusion(
        [box],
        [VehicleInstance(box, silhouette, "car")],
        rectangular,
        [rectangular],
    )
    assert replaced == 1
    assert result[50, 45] > 0
    assert result[50, 15] == 0


def test_missing_or_tiny_instance_keeps_unmatched_box() -> None:
    box = BBox(x1=0, y1=0, x2=90, y2=90)
    fallback = np.zeros((100, 100), np.uint8)
    fallback[:91, :91] = 255
    tiny = np.zeros_like(fallback)
    tiny[30:35, 30:35] = 255
    for instances in ([], [VehicleInstance(BBox(x1=30, y1=30, x2=35, y2=35), tiny, "car")]):
        result, replaced = pixel_occlusion([box], instances, fallback, [fallback])
        assert replaced == 0
        assert result[80, 80] == 255


def test_extra_vehicle_is_masked_when_recorded_detector_misses_it() -> None:
    empty = np.zeros((100, 100), np.uint8)
    instance = empty.copy()
    instance[30:60, 30:60] = 255
    result, _ = pixel_occlusion(
        [],
        [
            VehicleInstance(
                BBox(x1=30, y1=30, x2=60, y2=60),
                instance,
                "motorcycle",
            )
        ],
        empty,
        [],
    )
    assert result[40, 40] == 255


def test_rider_alone_cannot_replace_motorcycle_box() -> None:
    box = BBox(x1=10, y1=10, x2=80, y2=90)
    fallback = np.zeros((100, 100), np.uint8)
    fallback[10:91, 10:81] = 255
    rider = np.zeros_like(fallback)
    rider[10:50, 20:70] = 255
    result, replaced = pixel_occlusion(
        [box],
        [VehicleInstance(BBox(x1=20, y1=10, x2=70, y2=50), rider, "person")],
        fallback,
        [fallback],
    )
    assert replaced == 0
    assert result[80, 40] > 0


def test_no_segmentation_preserves_exact_fallback_without_growing_it() -> None:
    box = BBox(x1=10, y1=10, x2=80, y2=90)
    fallback = np.zeros((100, 100), np.uint8)
    fallback[10:91, 10:81] = 255
    result, replaced = pixel_occlusion([box], [], fallback, [fallback])
    assert replaced == 0
    np.testing.assert_array_equal(result, fallback)


def test_classifier_rejection_prevents_confirmed_marking_from_being_carried() -> None:
    detector = object.__new__(ExperimentalLaneDetector)
    detector.classifier = SimpleNamespace()
    detector.marking_mask = np.zeros((100, 100), np.uint8)
    detector.predictions = [{"label": "parking", "curve": _curve().model_dump()}]
    detector._previous = {"test": _curve()}
    detector._missing = {}
    detector._ages = {"test": 3}
    detector._minimum_confirmation = 2
    detector._maximum_missing = 1
    detector._minimum_confidence = 0.5
    detector._association_distance = 0.08
    result = detector._stabilize([], 100)
    assert not result.curves
    assert result.diagnostics.candidate_decisions[0].reason == "current_marking_evidence"


def test_wrong_mask_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        pixel_occlusion(
            [],
            [
                VehicleInstance(
                    BBox(x1=0, y1=0, x2=9, y2=9),
                    np.zeros((9, 9), np.uint8),
                    "car",
                )
            ],
            np.zeros((10, 10), np.uint8),
            [],
        )


@pytest.mark.parametrize(
    "label,confidence,reason,expected",
    [
        ("lane", 0.95, "colored_curb_context", None),
        ("lane", 0.70, "colored_curb_context", "colored_curb_context"),
        ("lane", 0.95, "arrow_paint", "arrow_paint"),
        ("parking", 0.95, None, "classifier_parking"),
        ("unknown", 0.40, "colored_curb_context", "colored_curb_context"),
    ],
)
def test_classification_exceptions_are_limited_and_unknown_keeps_original_rules(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    confidence: float,
    reason: str | None,
    expected: str | None,
) -> None:
    monkeypatch.setattr(YoloPLaneLineDetector, "_context_rejection", lambda *a, **kw: reason)
    detector = object.__new__(ExperimentalLaneDetector)
    detector.classifier = SimpleNamespace(predict=lambda *a: (label, confidence))
    detector.predictions = []
    image = np.zeros((100, 100), np.uint8)
    assert (
        detector._context_rejection(
            _curve(),
            image,
            image,
            image,
            np.zeros((100, 100, 3), np.uint8),
        )
        == expected
    )


def test_classifier_serialization_roundtrip_and_version_guard(tmp_path: Path) -> None:
    frames = [np.zeros((100, 100, 3), np.uint8), np.full((100, 100, 3), 200, np.uint8)]
    rows = np.asarray([curve_features(frame, _curve()) for frame in frames] * 4)
    classifier = MarkingClassifier.train(rows, ["lane", "parking"] * 4)
    path = tmp_path / "classifier.xml"
    classifier.save(path)
    loaded = MarkingClassifier.load(path)
    assert loaded.predict(frames[0], _curve()) == classifier.predict(frames[0], _curve())
    with pytest.raises(ValueError, match="already exists"):
        classifier.save(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["feature_version"] = 999
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incompatible"):
        MarkingClassifier.load(path)


def test_training_temporal_buffer_cannot_include_extra_window() -> None:
    fixture = {
        "videos": [
            {
                "name": "one",
                "width": 100,
                "height": 100,
                "cases": [{"frame_id": 20, "split": "extra", "polyline": [[1, 1], [2, 2]]}],
            }
        ],
        "training_regions": [
            {"video": "one", "frame_id": 10, "split": "train", "polyline": [[1, 1], [2, 2]]}
        ],
    }
    with pytest.raises(ValueError, match="overlaps"):
        review.validate_fixture(fixture)
    fixture["training_regions"][0]["frame_id"] = 0
    review.validate_fixture(fixture)


def test_variant_scoring_uses_each_variants_own_occlusion_mask() -> None:
    video = {
        "width": 100,
        "height": 100,
        "fps": 30,
        "tolerance_pixels": 1,
        "cases": [
            {
                "frame_id": 1,
                "group": "line",
                "category": "lane",
                "split": "extra",
                "expected_line": True,
                "polyline": [[30, 20], [30, 70]],
            }
        ],
    }
    curves = {"curves": [_curve().model_dump()]}
    masked = {1: {"lane_lines": curves, "occlusion_runs": [[0, 10000]]}}
    visible = {1: {"lane_lines": curves, "occlusion_runs": []}}
    assert review.score_variant(video, masked, "extra")["summary"]["visible_line_coverage"] == 0
    assert review.score_variant(video, visible, "extra")["summary"]["visible_line_coverage"] == 1


def test_unlabelled_candidate_is_not_implicitly_background() -> None:
    assert review.annotated_class(_curve(), []) is None
    regions = [
        {"label": "lane", "polyline": [[30, 20], [30, 70]]},
        {"label": "parking", "polyline": [[30, 20], [30, 70]]},
    ]
    assert review.annotated_class(_curve(), regions) is None


def test_empty_body_mask_cannot_get_perfect_scores() -> None:
    mask = np.zeros((100, 100), np.uint8)
    annotation: dict[str, Any] = {
        "audit_box": [0, 0, 100, 100],
        "polygon": [[20, 20], [40, 20], [40, 40], [20, 40]],
    }
    assert review.body_scores(mask, annotation) == {"precision": 0, "recall": 0, "iou": 0}


def test_windows_have_warmup_and_do_not_duplicate_overlapping_frames() -> None:
    assert review.windows([40, 41, 80], 100) == [(25, 42), (65, 81)]
    with pytest.raises(ValueError, match="outside"):
        review.windows([100], 100)


def test_fixture_keeps_published_holdouts_as_regressions() -> None:
    fixture = json.loads(
        (_scripts.parent / "tests/fixtures/lane_prototype_review.json").read_text()
    )
    review.validate_fixture(fixture)
    assert sum(c["split"] == "regression" for v in fixture["videos"] for c in v["cases"]) == 55
    assert sum(c["split"] == "extra" for v in fixture["videos"] for c in v["cases"]) == 50


def _body_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "body_audit",
        _scripts / "check_lane_prototype_masks.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_two_visible_bodies_in_audit_region_are_not_background() -> None:
    audit = _body_audit_module()
    mask = np.zeros((100, 100), np.uint8)
    mask[10:21, 10:21] = 255
    mask[30:41, 30:41] = 255
    polygons = [[[10, 10], [20, 10], [20, 20], [10, 20]], [[30, 30], [40, 30], [40, 40], [30, 40]]]
    assert audit.score_body(mask, polygons, [0, 0, 100, 100]) == {
        "precision": 1,
        "recall": 1,
        "iou": 1,
    }


@pytest.mark.parametrize("region", [[-1, 0, 100, 100], [50, 50, 60, 60]])
def test_invalid_or_unreviewed_body_region_cannot_pass(region: list[int]) -> None:
    audit = _body_audit_module()
    with pytest.raises(ValueError, match="outside|no reviewed"):
        audit.score_body(
            np.zeros((100, 100), np.uint8), [[[10, 10], [20, 10], [20, 20], [10, 20]]], region
        )

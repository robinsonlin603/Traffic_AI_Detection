"""Opt-in candidate classification and pixel occlusion experiments; no production wiring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.lane_lines import LaneCurve, LaneLineDiagnostics, LaneLineFrame
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector
from dashcam_ai.video.reader import _cv2

CLASSES = ("lane", "arrow", "parking", "crosswalk", "curb", "text", "background")
FEATURE_VERSION = 1


@dataclass(frozen=True)
class VehicleInstance:
    bbox: BBox
    mask: np.ndarray[Any, Any]
    category: str


def mask_runs(mask: np.ndarray[Any, Any]) -> list[list[int]]:
    transitions = np.flatnonzero(np.diff(np.r_[False, mask.ravel() > 0, False]))
    return transitions.reshape(-1, 2).tolist()


def pixel_occlusion(
    boxes: list[BBox],
    instances: list[VehicleInstance],
    fallback: np.ndarray[Any, Any],
    box_fallbacks: list[np.ndarray[Any, Any]],
) -> tuple[np.ndarray[Any, Any], int]:
    """Only replace a recorded box when a substantial matching instance exists."""
    if len(boxes) != len(box_fallbacks):
        raise ValueError("one fallback mask is required per box")
    if fallback.ndim != 2 or fallback.dtype != np.uint8:
        raise ValueError("fallback must be a two-dimensional uint8 mask")
    if any(mask.shape != fallback.shape or mask.dtype != np.uint8 for mask in box_fallbacks):
        raise ValueError("box fallback dimensions or dtype differ")
    valid = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
    if any(i.mask.shape != fallback.shape or i.mask.dtype != np.uint8 for i in instances):
        raise ValueError("instance dimensions or dtype differ")
    instances = [i for i in instances if i.category in valid and np.any(i.mask)]
    if not instances:
        return fallback.copy(), 0
    result = np.zeros_like(fallback)
    # Extra detected vehicles/riders must remain hidden even when the recorded
    # detector missed them. A missed segment never erases an unmatched box.
    for instance in instances:
        result |= instance.mask
    result = _cv2().dilate(result, np.ones((3, 3), np.uint8))
    replaced = 0
    for box, box_mask in zip(boxes, box_fallbacks, strict=True):
        matched = False
        for instance in instances:
            if instance.category == "person":
                # A rider prediction cannot prove that the motorcycle itself
                # was segmented. Keep its box if the vehicle instance is absent.
                continue
            other = instance.bbox
            intersection = max(0.0, min(box.x2, other.x2) - max(box.x1, other.x1)) * max(
                0.0, min(box.y2, other.y2) - max(box.y1, other.y1)
            )
            union = box.area + other.area - intersection
            if union > 0 and (
                intersection / union >= 0.2
                or (
                    intersection / max(other.area, 1) >= 0.7
                    and other.area / max(box.area, 1) >= 0.25
                )
            ):
                matched = True
                break
        if matched:
            replaced += 1
        else:
            result |= box_mask
    return result, replaced


def curve_features(frame: np.ndarray[Any, Any], curve: LaneCurve) -> np.ndarray[Any, Any]:
    """Local appearance and relative geometry, without frame IDs or absolute position."""
    cv2 = _cv2()
    height, width = frame.shape[:2]
    points = np.asarray([[p.x, p.y] for p in curve.points], np.float32)
    if not np.isfinite(points).all():
        raise ValueError("non-finite candidate coordinates")
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    span = max(x1 - x0, y1 - y0, 1)
    padding = max(24, span * 0.3)
    left, top = max(0, int(x0 - padding)), max(0, int(y0 - padding))
    right, bottom = min(width, int(x1 + padding + 1)), min(height, int(y1 + padding + 1))
    if left >= right or top >= bottom:
        raise ValueError("candidate outside frame")
    crop = frame[top:bottom, left:right, :3]
    patch = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(patch, cv2.COLOR_BGR2HLS)
    contrast = gray.astype(np.int16) - cv2.GaussianBlur(gray, (0, 0), 3).astype(np.int16)
    paint = ((contrast >= 5) & (gray >= 50) & (hls[:, :, 2] < 160)).astype(np.float32)
    appearance = cv2.resize(paint, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray).ravel()
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    colors = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HLS)[ys, xs].astype(float) / 255
    direction = points[-1] - points[0]
    direction /= max(float(np.linalg.norm(direction)), 1)
    geometry = np.r_[
        (x1 - x0) / span,
        (y1 - y0) / span,
        direction,
        colors.mean(axis=0),
        colors.std(axis=0),
        curve.lane_probability or 0,
        curve.drivable_probability or 0,
    ]
    return np.asarray(np.r_[appearance, hog, geometry], np.float32)


class MarkingClassifier:
    """Small supervised RTrees candidate prototype, with explicit unknown predictions."""

    def __init__(self, forest: Any) -> None:
        self.forest = forest

    @classmethod
    def train(cls, samples: np.ndarray[Any, Any], labels: list[str]) -> MarkingClassifier:
        if len(samples) != len(labels) or len(set(labels)) < 2 or len(labels) < 4:
            raise ValueError("training requires at least four candidates and two classes")
        if samples.ndim != 2 or not np.isfinite(samples).all():
            raise ValueError("training features must be finite rows")
        cv2 = _cv2()
        cv2.setRNGSeed(20261004)
        forest = cv2.ml.RTrees_create()
        forest.setMaxDepth(8)
        forest.setMinSampleCount(2)
        forest.setMaxCategories(len(CLASSES))
        forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 128, 0))
        forest.setCalculateVarImportance(True)
        targets = np.asarray([CLASSES.index(label) for label in labels], np.int32)
        if not forest.train(np.asarray(samples, np.float32), cv2.ml.ROW_SAMPLE, targets):
            raise RuntimeError("candidate classifier training failed")
        return cls(forest)

    def predict(self, frame: np.ndarray[Any, Any], curve: LaneCurve) -> tuple[str, float]:
        features = curve_features(frame, curve)[None]
        if features.shape[1] != self.forest.getVarCount():
            raise ValueError("classifier feature dimensions differ")
        votes = np.asarray(self.forest.getVotes(features, 0))
        index = int(np.argmax(votes[1]))
        confidence = float(votes[1, index] / max(int(votes[1].sum()), 1))
        label = CLASSES[int(votes[0, index])]
        return (label if confidence >= 0.65 else "unknown"), confidence

    def save(self, path: Path) -> None:
        if path.exists() or path.with_suffix(".json").exists():
            raise ValueError("classifier output already exists")
        self.forest.save(str(path))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "feature_version": FEATURE_VERSION,
                    "classes": CLASSES,
                    "feature_count": self.forest.getVarCount(),
                    "seed": 20261004,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> MarkingClassifier:
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["feature_version"] != FEATURE_VERSION or tuple(metadata["classes"]) != CLASSES:
            raise ValueError("classifier metadata incompatible")
        forest = _cv2().ml.RTrees_load(str(path))
        if forest.getVarCount() != metadata["feature_count"]:
            raise ValueError("classifier feature count incompatible")
        return cls(forest)


class ExperimentalLaneDetector(YoloPLaneLineDetector):
    """Explicit opt-in experiment; the default analyzer continues using its existing detector."""

    def __init__(self, *, classifier: MarkingClassifier | None = None, **settings: Any) -> None:
        super().__init__(**settings)
        self.classifier = classifier
        self.pixel_mask: np.ndarray[Any, Any] | None = None
        self.predictions: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.pixel_mask = None
        self.predictions = []

    def detect(self, frame: Any, excluded_boxes: list[BBox] | None = None) -> LaneLineFrame:
        self.predictions = []
        return super().detect(frame, excluded_boxes)

    def _stabilize(
        self,
        candidates: list[LaneCurve],
        width: int,
        diagnostics: LaneLineDiagnostics | None = None,
        frame: np.ndarray[Any, Any] | None = None,
    ) -> LaneLineFrame:
        if self.marking_mask is not None and self.classifier is not None:
            # A newly classified marking must also suppress yesterday's lane
            # track, rather than reappear through the missing-frame carry rule.
            self.marking_mask = self.marking_mask.copy()
            for prediction in self.predictions:
                if prediction["label"] in ("lane", "unknown"):
                    continue
                points = np.rint([[p["x"], p["y"]] for p in prediction["curve"]["points"]]).astype(
                    np.int32
                )
                _cv2().polylines(self.marking_mask, [points], False, 255, 5)
        return super()._stabilize(candidates, width, diagnostics, frame)

    def _candidate_mask(
        self,
        frame: np.ndarray[Any, Any],
        probability: np.ndarray[Any, Any],
        excluded_boxes: list[BBox],
        drivable_probability: np.ndarray[Any, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, int]]:
        if self.pixel_mask is None:
            return super()._candidate_mask(frame, probability, excluded_boxes, drivable_probability)
        if self.pixel_mask.shape != probability.shape or self.pixel_mask.dtype != np.uint8:
            raise ValueError("pixel mask must match the frame dimensions and have uint8 dtype")
        mask, diagnostics = super()._candidate_mask(frame, probability, [], drivable_probability)
        self.occlusion_mask = self.pixel_mask.copy()
        removed = int(np.count_nonzero((mask > 0) & (self.pixel_mask > 0)))
        mask[self.pixel_mask > 0] = 0
        diagnostics["vehicle_excluded_pixel_count"] = removed
        return mask, diagnostics

    def _context_rejection(
        self,
        curve: LaneCurve,
        paint: np.ndarray[Any, Any],
        rejected_paint: np.ndarray[Any, Any],
        drivable: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any] | None = None,
        *, observed_paint: bool = False,
    ) -> str | None:
        reason = super()._context_rejection(
            curve, paint, rejected_paint, drivable, frame, observed_paint=observed_paint,
        )
        if self.classifier is None or frame is None:
            return reason
        label, confidence = self.classifier.predict(frame, curve)
        self.predictions.append(
            {
                "label": label,
                "confidence": confidence,
                "original_reason": reason,
                "curve": curve.model_dump(mode="json"),
            }
        )
        if label not in ("lane", "unknown"):
            return "classifier_" + label
        if label == "lane" and confidence >= 0.8 and reason == "colored_curb_context":
            return None
        return reason

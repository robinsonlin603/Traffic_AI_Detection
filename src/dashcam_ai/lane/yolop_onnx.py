"""以官方 YOLOP ONNX 產生 lane segmentation 曲線候選。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneGeometryStatus
from dashcam_ai.domain.lane_evidence import LaneCurveEvidence, LaneEvidenceFrame
from dashcam_ai.video.reader import _cv2


class YOLOPOnnxLaneEvidenceBackend:
    """只使用 lane-line segmentation output，不執行 YOLOP detection head。"""

    def __init__(
        self,
        weights: Path,
        *,
        input_size: int,
        minimum_confidence: float,
        segmentation_threshold: float,
        minimum_component_area_ratio: float,
        curve_sample_count: int,
    ) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOP ONNX weights do not exist: {weights}")
        if input_size <= 0 or curve_sample_count < 2:
            raise ValueError("YOLOP input and curve sample sizes must be positive")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between zero and one")
        if not 0 < segmentation_threshold < 1:
            raise ValueError("segmentation threshold must be between zero and one")
        if not 0 < minimum_component_area_ratio < 1:
            raise ValueError("component area ratio must be between zero and one")
        self._cv2 = _cv2()
        self._net = self._cv2.dnn.readNetFromONNX(str(weights))
        self._input_size = input_size
        self._minimum_confidence = minimum_confidence
        self._segmentation_threshold = segmentation_threshold
        self._minimum_area_ratio = minimum_component_area_ratio
        self._sample_count = curve_sample_count
        self._weight_sha256 = self._sha256(weights)

    def detect(
        self, frame: Any, road_roi: tuple[Point2D, ...]
    ) -> LaneEvidenceFrame:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return self._unknown("YOLOP requires a BGR color numpy frame")
        height, width = frame.shape[:2]
        tensor, ratio, pad_x, pad_y, resized_width, resized_height = self._preprocess(frame)
        self._net.setInput(tensor)
        _, _, lane_output = self._net.forward(
            ["det_out", "drive_area_seg", "lane_line_seg"]
        )
        logits = lane_output[0]
        probabilities = self._lane_probability(logits)
        cropped = probabilities[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ]
        if ratio <= 0 or cropped.size == 0:
            return self._unknown("YOLOP preprocessing produced an empty lane mask")
        probability = self._cv2.resize(cropped, (width, height))
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_array = np.asarray(
            [[round(point.x), round(point.y)] for point in road_roi], dtype=np.int32
        )
        self._cv2.fillPoly(roi_mask, [roi_array], 1)
        binary = (
            (probability >= self._segmentation_threshold) & (roi_mask.astype(bool))
        ).astype(np.uint8)
        count, labels, stats, _ = self._cv2.connectedComponentsWithStats(binary, 8)
        minimum_area = max(1, round(width * height * self._minimum_area_ratio))
        curves: list[LaneCurveEvidence] = []
        for label in range(1, count):
            area = int(stats[label, self._cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            curve = self._component_curve(labels == label, probability, len(curves), area)
            if curve is not None:
                curves.append(curve)
        curves.sort(key=lambda item: item.points[-1].x)
        if not curves:
            return self._unknown("YOLOP found no lane components inside the road ROI")
        confidence = float(sum(item.confidence for item in curves) / len(curves))
        status = (
            LaneGeometryStatus.VALID
            if len(curves) >= 2 and confidence >= self._minimum_confidence
            else LaneGeometryStatus.DEGRADED
        )
        return LaneEvidenceFrame(
            status=status,
            backend="yolop_onnx",
            weight_sha256=self._weight_sha256,
            confidence=confidence,
            curves=tuple(curves),
            estimated_lane_count=None,
            reason=(
                None
                if status is LaneGeometryStatus.VALID
                else "insufficient confident YOLOP lane components"
            ),
        )

    def _preprocess(
        self, frame: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], float, int, int, int, int]:
        height, width = frame.shape[:2]
        ratio = min(self._input_size / height, self._input_size / width)
        resized_width = round(width * ratio)
        resized_height = round(height * ratio)
        pad_x = (self._input_size - resized_width) // 2
        pad_y = (self._input_size - resized_height) // 2
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        resized = self._cv2.resize(rgb, (resized_width, resized_height))
        canvas = np.full(
            (self._input_size, self._input_size, 3), 114, dtype=np.uint8
        )
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        scaled = canvas.astype(np.float32) / np.float32(255.0)
        normalized = (
            scaled - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return (
            np.asarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32),
            ratio,
            pad_x,
            pad_y,
            resized_width,
            resized_height,
        )

    @staticmethod
    def _lane_probability(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        shifted = logits - np.max(logits, axis=0, keepdims=True)
        exponent = np.exp(shifted)
        return np.asarray(exponent[1] / np.sum(exponent, axis=0), dtype=np.float32)

    def _component_curve(
        self,
        component: np.ndarray[Any, Any],
        probability: np.ndarray[Any, Any],
        index: int,
        area: int,
    ) -> LaneCurveEvidence | None:
        ys, xs = np.nonzero(component)
        if len(ys) < 2:
            return None
        sample_edges = np.linspace(
            float(np.min(ys)), float(np.max(ys)) + 1, self._sample_count + 1
        )
        points: list[Point2D] = []
        for start, end in zip(sample_edges[:-1], sample_edges[1:], strict=True):
            selected = (ys >= start) & (ys < end)
            if not np.any(selected):
                continue
            points.append(
                Point2D(x=float(np.median(xs[selected])), y=float(np.median(ys[selected])))
            )
        if len(points) < 2:
            return None
        confidence = float(np.mean(probability[component]))
        return LaneCurveEvidence(
            evidence_id=f"yolop_curve_{index}",
            points=tuple(points),
            confidence=max(0.0, min(1.0, confidence)),
            supporting_segments=area,
        )

    def _unknown(self, reason: str) -> LaneEvidenceFrame:
        return LaneEvidenceFrame(
            status=LaneGeometryStatus.UNKNOWN,
            backend="yolop_onnx",
            weight_sha256=self._weight_sha256,
            confidence=0,
            reason=reason,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

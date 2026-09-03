"""OpenCV 單幀車道曲線候選 prototype。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneGeometryStatus
from dashcam_ai.domain.lane_evidence import LaneCurveEvidence, LaneEvidenceFrame
from dashcam_ai.video.reader import _cv2


@dataclass(frozen=True, slots=True)
class _Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    def x_at(self, y: float) -> float:
        if self.y2 == self.y1:
            return (self.x1 + self.x2) / 2
        ratio = (y - self.y1) / (self.y2 - self.y1)
        return self.x1 + ratio * (self.x2 - self.x1)

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))


class OpenCVLaneEvidenceBackend:
    """以 Canny、probabilistic Hough 與線性擬合產生候選曲線。"""

    def __init__(
        self,
        *,
        minimum_confidence: float,
        minimum_curve_fit_confidence: float,
        canny_low_threshold: int,
        canny_high_threshold: int,
        hough_threshold: int,
        minimum_line_length_pixels: int,
        maximum_line_gap_pixels: int,
        minimum_absolute_slope: float,
        boundary_cluster_distance_ratio: float,
        curve_sample_count: int,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between zero and one")
        if not 0 <= minimum_curve_fit_confidence <= 1:
            raise ValueError("curve confidence must be between zero and one")
        if not 0 <= canny_low_threshold < canny_high_threshold <= 255:
            raise ValueError("invalid Canny thresholds")
        if hough_threshold <= 0 or minimum_line_length_pixels <= 0:
            raise ValueError("Hough thresholds must be positive")
        if maximum_line_gap_pixels < 0 or minimum_absolute_slope <= 0:
            raise ValueError("line filters are invalid")
        if not 0 < boundary_cluster_distance_ratio <= 1 or curve_sample_count < 2:
            raise ValueError("curve clustering parameters are invalid")
        self._minimum_confidence = minimum_confidence
        self._minimum_curve_confidence = minimum_curve_fit_confidence
        self._canny_low = canny_low_threshold
        self._canny_high = canny_high_threshold
        self._hough_threshold = hough_threshold
        self._minimum_length = minimum_line_length_pixels
        self._maximum_gap = maximum_line_gap_pixels
        self._minimum_slope = minimum_absolute_slope
        self._cluster_ratio = boundary_cluster_distance_ratio
        self._sample_count = curve_sample_count

    def detect(
        self, frame: Any, road_roi: tuple[Point2D, ...]
    ) -> LaneEvidenceFrame:
        cv2 = _cv2()
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            return self._unknown("frame must be a grayscale or color numpy array")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0 or len(road_roi) < 3:
            return self._unknown("frame and road ROI must be non-empty")
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)
        mask = np.zeros_like(edges)
        roi_array = np.asarray(
            [[round(point.x), round(point.y)] for point in road_roi], dtype=np.int32
        )
        cv2.fillPoly(mask, [roi_array], 255)
        masked = cv2.bitwise_and(edges, mask)
        raw_lines = cv2.HoughLinesP(
            masked,
            1,
            np.pi / 180,
            self._hough_threshold,
            minLineLength=self._minimum_length,
            maxLineGap=self._maximum_gap,
        )
        if raw_lines is None:
            return self._unknown("no lane-like line evidence")
        segments = self._segments(raw_lines)
        if not segments:
            return self._unknown("no line evidence passed the slope gate")
        bottom_y = float(max(point.y for point in road_roi))
        top_y = float(min(point.y for point in road_roi))
        clusters = self._cluster(segments, bottom_y, width * self._cluster_ratio)
        curves = tuple(
            curve
            for index, cluster in enumerate(clusters)
            if (
                curve := self._fit_curve(cluster, index, top_y, bottom_y, height)
            ).confidence
            >= self._minimum_curve_confidence
        )
        if not curves:
            return self._unknown("line fits are below the confidence gate")
        confidence = float(sum(curve.confidence for curve in curves) / len(curves))
        if confidence < self._minimum_confidence or len(curves) < 2:
            return LaneEvidenceFrame(
                status=LaneGeometryStatus.DEGRADED,
                backend="opencv_canny_hough",
                confidence=confidence,
                curves=curves,
                reason="insufficient confident boundary candidates",
            )
        return LaneEvidenceFrame(
            status=LaneGeometryStatus.VALID,
            backend="opencv_canny_hough",
            confidence=confidence,
            curves=curves,
            estimated_lane_count=None,
        )

    def _segments(self, raw_lines: Any) -> list[_Segment]:
        output: list[_Segment] = []
        for values in np.asarray(raw_lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in values)
            dx = x2 - x1
            slope = float("inf") if abs(dx) < 1e-9 else (y2 - y1) / dx
            if abs(slope) < self._minimum_slope:
                continue
            output.append(_Segment(x1=x1, y1=y1, x2=x2, y2=y2))
        return output

    @staticmethod
    def _cluster(
        segments: list[_Segment], bottom_y: float, maximum_distance: float
    ) -> list[list[_Segment]]:
        ordered = sorted(segments, key=lambda item: item.x_at(bottom_y))
        clusters: list[list[_Segment]] = []
        centers: list[float] = []
        for segment in ordered:
            position = segment.x_at(bottom_y)
            if not clusters or abs(position - centers[-1]) > maximum_distance:
                clusters.append([segment])
                centers.append(position)
                continue
            clusters[-1].append(segment)
            centers[-1] = sum(item.x_at(bottom_y) for item in clusters[-1]) / len(
                clusters[-1]
            )
        return clusters

    def _fit_curve(
        self,
        segments: list[_Segment],
        index: int,
        top_y: float,
        bottom_y: float,
        frame_height: int,
    ) -> LaneCurveEvidence:
        ys = np.asarray([value for item in segments for value in (item.y1, item.y2)])
        xs = np.asarray([value for item in segments for value in (item.x1, item.x2)])
        slope, intercept = np.polyfit(ys, xs, 1)
        predicted = slope * ys + intercept
        mean_error = float(np.mean(np.abs(predicted - xs)))
        support_score = min(1.0, sum(item.length for item in segments) / frame_height)
        error_score = max(0.0, 1.0 - mean_error / max(frame_height * 0.05, 1.0))
        confidence = max(0.0, min(1.0, support_score * error_score))
        sample_ys = np.linspace(max(top_y, float(np.min(ys))), bottom_y, self._sample_count)
        points = tuple(
            Point2D(x=float(slope * y + intercept), y=float(y)) for y in sample_ys
        )
        return LaneCurveEvidence(
            evidence_id=f"curve_{index}",
            points=points,
            confidence=confidence,
            supporting_segments=len(segments),
        )

    @staticmethod
    def _unknown(reason: str) -> LaneEvidenceFrame:
        return LaneEvidenceFrame(
            status=LaneGeometryStatus.UNKNOWN,
            backend="opencv_canny_hough",
            confidence=0,
            reason=reason,
        )

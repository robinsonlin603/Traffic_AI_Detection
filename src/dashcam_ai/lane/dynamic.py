"""將單幀 lane evidence 穩定成可安全使用的動態車道拓撲。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneBoundaryEvidenceSource,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
    NormalizedPoint2D,
)
from dashcam_ai.domain.lane_evidence import LaneCurveEvidence, LaneEvidenceFrame
from dashcam_ai.lane.evidence import LaneEvidenceBackend


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    points: tuple[Point2D, ...]
    confidence: float


class TemporalLaneGeometryTracker:
    """跨幀關聯曲線，並在足夠證據後建立穩定 lane topology。"""

    def __init__(
        self,
        *,
        smoothing_alpha: float,
        maximum_boundary_jump_ratio: float,
        missing_frame_tolerance: int,
        topology_confirmation_frames: int,
        maximum_lane_count: int,
        minimum_boundary_separation_ratio: float,
        curve_sample_count: int,
        minimum_confidence: float,
    ) -> None:
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing alpha must be between zero and one")
        if not 0 < maximum_boundary_jump_ratio <= 1:
            raise ValueError("maximum boundary jump ratio must be between zero and one")
        if missing_frame_tolerance < 0:
            raise ValueError("missing frame tolerance cannot be negative")
        if topology_confirmation_frames <= 0:
            raise ValueError("topology confirmation frames must be positive")
        if maximum_lane_count < 2:
            raise ValueError("maximum lane count must be at least two")
        if not 0 < minimum_boundary_separation_ratio < 1:
            raise ValueError("minimum boundary separation ratio must be between zero and one")
        if curve_sample_count < 2:
            raise ValueError("curve sample count must be at least two")
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between zero and one")
        self._alpha = smoothing_alpha
        self._maximum_jump_ratio = maximum_boundary_jump_ratio
        self._missing_tolerance = missing_frame_tolerance
        self._confirmation_frames = topology_confirmation_frames
        self._maximum_boundaries = maximum_lane_count - 1
        self._minimum_separation_ratio = minimum_boundary_separation_ratio
        self._sample_count = curve_sample_count
        self._minimum_confidence = minimum_confidence
        self._stable: tuple[_BoundaryCandidate, ...] = ()
        self._pending: tuple[_BoundaryCandidate, ...] = ()
        self._pending_frames = 0
        self._missing_frames = 0
        self._topology_version = 0

    def update(
        self,
        evidence: LaneEvidenceFrame,
        *,
        width: int,
        height: int,
        road_roi: tuple[Point2D, ...],
    ) -> LaneGeometry:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if len(road_roi) < 3:
            raise ValueError("road ROI requires at least three points")
        candidates = self._candidates(evidence.curves, width, road_roi)
        if evidence.status is LaneGeometryStatus.UNKNOWN or not candidates:
            return self._bridge_or_unknown(width, height, road_roi, evidence.reason)
        if self._stable and len(candidates) == len(self._stable):
            smoothed = self._associate_and_smooth(self._stable, candidates, width)
            if smoothed is not None:
                self._stable = smoothed
                self._pending = ()
                self._pending_frames = 0
                self._missing_frames = 0
                return self._geometry(
                    smoothed,
                    width,
                    height,
                    road_roi,
                    status=(
                        LaneGeometryStatus.VALID
                        if evidence.status is LaneGeometryStatus.VALID
                        and self._mean_confidence(smoothed) >= self._minimum_confidence
                        else LaneGeometryStatus.DEGRADED
                    ),
                    source=LaneBoundaryEvidenceSource.OBSERVED,
                    reason=(
                        None
                        if evidence.status is LaneGeometryStatus.VALID
                        and self._mean_confidence(smoothed) >= self._minimum_confidence
                        else "observed topology has degraded lane evidence"
                    ),
                )
            return self._bridge_or_unknown(width, height, road_roi, "abrupt boundary jump rejected")
        self._advance_pending(candidates, width)
        if self._pending_frames >= self._confirmation_frames:
            self._stable = self._pending
            self._pending = ()
            self._pending_frames = 0
            self._missing_frames = 0
            self._topology_version += 1
            return self._geometry(
                self._stable,
                width,
                height,
                road_roi,
                status=(
                    LaneGeometryStatus.VALID
                    if evidence.status is LaneGeometryStatus.VALID
                    and self._mean_confidence(self._stable) >= self._minimum_confidence
                    else LaneGeometryStatus.DEGRADED
                ),
                source=LaneBoundaryEvidenceSource.OBSERVED,
                reason=(
                    None
                    if evidence.status is LaneGeometryStatus.VALID
                    and self._mean_confidence(self._stable) >= self._minimum_confidence
                    else "confirmed topology has degraded lane evidence"
                ),
            )
        if self._stable:
            return self._geometry(
                self._stable,
                width,
                height,
                road_roi,
                status=LaneGeometryStatus.DEGRADED,
                source=LaneBoundaryEvidenceSource.INFERRED,
                reason="lane topology change awaiting confirmation",
            )
        return self._geometry(
            candidates,
            width,
            height,
            road_roi,
            status=LaneGeometryStatus.DEGRADED,
            source=LaneBoundaryEvidenceSource.OBSERVED,
            reason="initial lane topology awaiting confirmation",
            topology_id=None,
        )

    def _advance_pending(self, candidates: tuple[_BoundaryCandidate, ...], width: int) -> None:
        if len(candidates) == len(self._pending):
            smoothed = self._associate_and_smooth(self._pending, candidates, width)
            if smoothed is not None:
                self._pending = smoothed
                self._pending_frames += 1
                return
        self._pending = candidates
        self._pending_frames = 1

    def _bridge_or_unknown(
        self,
        width: int,
        height: int,
        road_roi: tuple[Point2D, ...],
        reason: str | None,
    ) -> LaneGeometry:
        self._pending = ()
        self._pending_frames = 0
        if self._stable and self._missing_frames < self._missing_tolerance:
            self._missing_frames += 1
            return self._geometry(
                self._stable,
                width,
                height,
                road_roi,
                status=LaneGeometryStatus.DEGRADED,
                source=LaneBoundaryEvidenceSource.INFERRED,
                reason=reason or "lane evidence temporarily missing",
            )
        self._missing_frames += 1
        if self._stable:
            self._stable = ()
        return LaneGeometry(
            status=LaneGeometryStatus.UNKNOWN,
            provenance=LaneGeometryProvenance.UNKNOWN,
            confidence=0,
            frame_width=width,
            frame_height=height,
            reason=reason or "lane evidence unavailable",
        )

    def _candidates(
        self,
        curves: tuple[LaneCurveEvidence, ...],
        width: int,
        road_roi: tuple[Point2D, ...],
    ) -> tuple[_BoundaryCandidate, ...]:
        minimum_y = min(point.y for point in road_roi)
        maximum_y = max(point.y for point in road_roi)
        sample_ys = np.linspace(minimum_y, maximum_y, self._sample_count)
        fitted: list[_BoundaryCandidate] = []
        for curve in curves:
            ys = np.asarray([point.y for point in curve.points], dtype=np.float64)
            xs = np.asarray([point.x for point in curve.points], dtype=np.float64)
            if len(np.unique(ys)) < 2:
                continue
            degree = min(2, len(np.unique(ys)) - 1)
            coefficients = np.polyfit(ys, xs, degree)
            predicted = np.polyval(coefficients, sample_ys)
            if not np.all(np.isfinite(predicted)):
                continue
            fitted.append(
                _BoundaryCandidate(
                    points=tuple(
                        Point2D(x=float(x), y=float(y))
                        for x, y in zip(predicted, sample_ys, strict=True)
                    ),
                    confidence=curve.confidence,
                )
            )
        fitted.sort(key=self._bottom_x)
        merged: list[_BoundaryCandidate] = []
        for candidate in fitted:
            if merged and self._distance(merged[-1], candidate) < (
                self._minimum_separation_ratio * width
            ):
                merged[-1] = self._merge(merged[-1], candidate)
            else:
                merged.append(candidate)
        if len(merged) > self._maximum_boundaries:
            merged = sorted(merged, key=lambda item: item.confidence, reverse=True)[
                : self._maximum_boundaries
            ]
            merged.sort(key=self._bottom_x)
        return tuple(merged)

    def _associate_and_smooth(
        self,
        previous: tuple[_BoundaryCandidate, ...],
        current: tuple[_BoundaryCandidate, ...],
        width: int,
    ) -> tuple[_BoundaryCandidate, ...] | None:
        if any(
            self._distance(old, new) > self._maximum_jump_ratio * width
            for old, new in zip(previous, current, strict=True)
        ):
            return None
        return tuple(
            _BoundaryCandidate(
                points=tuple(
                    Point2D(
                        x=(1 - self._alpha) * old_point.x + self._alpha * new_point.x,
                        y=old_point.y,
                    )
                    for old_point, new_point in zip(old.points, new.points, strict=True)
                ),
                confidence=(1 - self._alpha) * old.confidence + self._alpha * new.confidence,
            )
            for old, new in zip(previous, current, strict=True)
        )

    def _geometry(
        self,
        boundaries: tuple[_BoundaryCandidate, ...],
        width: int,
        height: int,
        road_roi: tuple[Point2D, ...],
        *,
        status: LaneGeometryStatus,
        source: LaneBoundaryEvidenceSource,
        reason: str | None,
        topology_id: str | None = "stable",
    ) -> LaneGeometry:
        sample_ys = tuple(point.y for point in boundaries[0].points)
        left_edge = tuple(Point2D(x=self._roi_edges_at_y(road_roi, y)[0], y=y) for y in sample_ys)
        right_edge = tuple(Point2D(x=self._roi_edges_at_y(road_roi, y)[1], y=y) for y in sample_ys)
        ordered_edges = (left_edge,) + tuple(item.points for item in boundaries) + (right_edge,)
        lanes = tuple(
            LaneRegion(
                lane_id=f"lane_{index}",
                lateral_order=index,
                polygon=left + tuple(reversed(right)),
            )
            for index, (left, right) in enumerate(
                zip(ordered_edges[:-1], ordered_edges[1:], strict=True)
            )
        )
        lane_boundaries = tuple(
            LaneBoundary(
                boundary_id=f"boundary_{index}_{index + 1}",
                left_lane_id=f"lane_{index}",
                right_lane_id=f"lane_{index + 1}",
                points=item.points,
                confidence=item.confidence,
                evidence_source=source,
            )
            for index, item in enumerate(boundaries)
        )
        stable_id = f"topology-{self._topology_version}" if topology_id is not None else None
        return LaneGeometry(
            status=status,
            provenance=LaneGeometryProvenance.DYNAMIC,
            confidence=self._mean_confidence(boundaries),
            frame_width=width,
            frame_height=height,
            lanes=lanes,
            boundaries=lane_boundaries,
            topology_id=stable_id,
            topology_version=self._topology_version if stable_id is not None else None,
            reason=reason,
        )

    @staticmethod
    def _roi_edges_at_y(road_roi: tuple[Point2D, ...], y: float) -> tuple[float, float]:
        intersections: list[float] = []
        for start, end in zip(road_roi, road_roi[1:] + road_roi[:1], strict=True):
            lower, upper = sorted((start.y, end.y))
            if y < lower or y > upper or start.y == end.y:
                continue
            ratio = (y - start.y) / (end.y - start.y)
            intersections.append(start.x + ratio * (end.x - start.x))
        if len(intersections) < 2:
            xs = [point.x for point in road_roi]
            return min(xs), max(xs)
        return min(intersections), max(intersections)

    @staticmethod
    def _distance(left: _BoundaryCandidate, right: _BoundaryCandidate) -> float:
        return float(
            np.mean(
                [
                    abs(left_point.x - right_point.x)
                    for left_point, right_point in zip(left.points, right.points, strict=True)
                ]
            )
        )

    @staticmethod
    def _bottom_x(candidate: _BoundaryCandidate) -> float:
        return max(candidate.points, key=lambda point: point.y).x

    @staticmethod
    def _merge(left: _BoundaryCandidate, right: _BoundaryCandidate) -> _BoundaryCandidate:
        total = left.confidence + right.confidence
        left_weight = left.confidence / total if total else 0.5
        return _BoundaryCandidate(
            points=tuple(
                Point2D(
                    x=left_weight * left_point.x + (1 - left_weight) * right_point.x,
                    y=left_point.y,
                )
                for left_point, right_point in zip(left.points, right.points, strict=True)
            ),
            confidence=max(left.confidence, right.confidence),
        )

    @staticmethod
    def _mean_confidence(boundaries: tuple[_BoundaryCandidate, ...]) -> float:
        return float(sum(item.confidence for item in boundaries) / len(boundaries))


class DynamicLaneDetector:
    """將單幀 backend 與 temporal topology 組成 LaneDetector。"""

    def __init__(
        self,
        backend: LaneEvidenceBackend,
        tracker: TemporalLaneGeometryTracker,
        road_roi: tuple[NormalizedPoint2D, ...],
    ) -> None:
        if len(road_roi) < 3:
            raise ValueError("road ROI requires at least three points")
        self._backend = backend
        self._tracker = tracker
        self._road_roi = road_roi

    def detect(self, frame: Any, width: int, height: int) -> LaneGeometry:
        roi = tuple(point.to_original(width, height) for point in self._road_roi)
        evidence = self._backend.detect(frame, roi)
        return self._tracker.update(
            evidence,
            width=width,
            height=height,
            road_roi=roi,
        )

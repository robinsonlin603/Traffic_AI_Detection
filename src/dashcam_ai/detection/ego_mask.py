"""Exclude the fixed camera's own vehicle before tracking assigns identities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def validate_ego_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("ego_vehicle_polygon requires at least three distinct vertices")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for point in points for v in point):
        raise ValueError("ego_vehicle_polygon coordinates must be finite ratios in [0, 1]")
    area = sum(
        a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=True)
    )
    if abs(area) < 1e-8:
        raise ValueError("ego_vehicle_polygon must have nonzero area")
    return points


class EgoVehicleMask:
    """Cache a normalized polygon; never mutate the caller's original image."""

    def __init__(self, polygon: list[tuple[float, float]], overlap_threshold: float = 0.8):
        self.polygon = list(validate_ego_polygon(polygon))
        if not math.isfinite(overlap_threshold) or not 0 < overlap_threshold <= 1:
            raise ValueError("ego_vehicle_overlap_threshold must be in (0, 1]")
        self.overlap_threshold = overlap_threshold
        self._shape: tuple[int, int] | None = None
        self._mask: Any = None
        self._integral: Any = None

    def mask(self, shape: tuple[int, int]) -> Any:
        import cv2

        if shape != self._shape:
            height, width = shape
            if height <= 0 or width <= 0:
                raise ValueError("image dimensions must be positive")
            mask = np.zeros(shape, dtype=np.uint8)
            if self.polygon:
                points = np.rint(np.asarray(self.polygon) * [width - 1, height - 1]).astype(
                    np.int32
                )
                cv2.fillPoly(mask, [points], 1)
            self._mask = mask.astype(bool)
            self._integral = cv2.integral(mask, sdepth=cv2.CV_64F)
            self._shape = shape
        return self._mask

    def prepare(self, frame: Any) -> Any:
        if not self.polygon:
            return frame
        image = frame.copy()
        image[self.mask(frame.shape[:2])] = 114
        return image

    def overlap(self, box: Any, shape: tuple[int, int]) -> float:
        if not self.polygon:
            return 0.0
        self.mask(shape)
        height, width = shape
        x1, y1, x2, y2 = (float(value) for value in box)
        left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
        right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        area = (right - left) * (bottom - top)
        if right <= left or bottom <= top:
            return 0.0
        integral = self._integral
        overlap = (
            integral[bottom, right]
            - integral[top, right]
            - integral[bottom, left]
            + integral[top, left]
        )
        return float(overlap / area)

    def excludes(self, box: Any, shape: tuple[int, int]) -> bool:
        return self.overlap(box, shape) >= self.overlap_threshold

"""以 OpenCV 背景特徵追蹤估算相鄰影格間的相機運動。"""

from __future__ import annotations

from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.motion import (
    EgoMotionEstimate,
    EgoMotionQuality,
    EgoMotionStatus,
    HomographyTransform,
)
from dashcam_ai.video.reader import _cv2


class OpenCVEgoMotionEstimator:
    """使用 Shi–Tomasi、LK optical flow 與 RANSAC homography。"""

    def __init__(
        self,
        *,
        max_features: int = 500,
        feature_quality_level: float = 0.01,
        feature_min_distance: float = 8.0,
        optical_flow_window_size: int = 21,
        optical_flow_max_level: int = 3,
        ransac_reprojection_threshold: float = 3.0,
        minimum_tracked_features: int = 12,
        minimum_inliers: int = 8,
        minimum_inlier_ratio: float = 0.5,
        maximum_mean_reprojection_error: float = 2.5,
        mask_padding_pixels: int = 8,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if not 0 < feature_quality_level <= 1:
            raise ValueError("feature_quality_level must be between zero and one")
        if feature_min_distance < 0:
            raise ValueError("feature_min_distance must not be negative")
        if optical_flow_window_size <= 0 or optical_flow_window_size % 2 == 0:
            raise ValueError("optical_flow_window_size must be a positive odd integer")
        if optical_flow_max_level < 0:
            raise ValueError("optical_flow_max_level must not be negative")
        if ransac_reprojection_threshold <= 0:
            raise ValueError("ransac_reprojection_threshold must be positive")
        if minimum_tracked_features < 4 or minimum_inliers < 4:
            raise ValueError("homography quality gates require at least four points")
        if not 0 <= minimum_inlier_ratio <= 1:
            raise ValueError("minimum_inlier_ratio must be between zero and one")
        if maximum_mean_reprojection_error < 0:
            raise ValueError("maximum_mean_reprojection_error must not be negative")
        if mask_padding_pixels < 0:
            raise ValueError("mask_padding_pixels must not be negative")
        self._max_features = max_features
        self._feature_quality_level = feature_quality_level
        self._feature_min_distance = feature_min_distance
        self._window_size = optical_flow_window_size
        self._max_level = optical_flow_max_level
        self._ransac_threshold = ransac_reprojection_threshold
        self._minimum_tracked = minimum_tracked_features
        self._minimum_inliers = minimum_inliers
        self._minimum_inlier_ratio = minimum_inlier_ratio
        self._maximum_error = maximum_mean_reprojection_error
        self._mask_padding = mask_padding_pixels

    def estimate(
        self, previous_frame: Any, current_frame: Any, excluded_boxes: list[BBox]
    ) -> EgoMotionEstimate:
        """估算 previous 到 current 的 homography；品質不足時回傳 unknown。"""
        cv2 = _cv2()
        if not isinstance(previous_frame, np.ndarray) or not isinstance(current_frame, np.ndarray):
            return self._unknown("frames must be numpy arrays")
        if previous_frame.shape[:2] != current_frame.shape[:2]:
            return self._unknown("frame dimensions do not match")
        if previous_frame.ndim not in (2, 3) or current_frame.ndim not in (2, 3):
            return self._unknown("frames must be grayscale or color images")

        previous_gray = self._to_gray(previous_frame, cv2)
        current_gray = self._to_gray(current_frame, cv2)
        mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
        height, width = previous_gray.shape
        for box in excluded_boxes:
            clipped = box.clip(width, height)
            x1 = max(int(np.floor(clipped.x1)) - self._mask_padding, 0)
            y1 = max(int(np.floor(clipped.y1)) - self._mask_padding, 0)
            x2 = min(int(np.ceil(clipped.x2)) + self._mask_padding, width)
            y2 = min(int(np.ceil(clipped.y2)) + self._mask_padding, height)
            mask[y1:y2, x1:x2] = 0

        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self._max_features,
            qualityLevel=self._feature_quality_level,
            minDistance=self._feature_min_distance,
            mask=mask,
        )
        detected = 0 if previous_points is None else len(previous_points)
        if previous_points is None or detected < self._minimum_tracked:
            return self._unknown("insufficient background features", detected=detected)

        current_points, tracking_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(self._window_size, self._window_size),
            maxLevel=self._max_level,
        )
        if current_points is None or tracking_status is None:
            return self._unknown("optical flow failed", detected=detected)
        tracked_mask = tracking_status.reshape(-1).astype(bool)
        source = previous_points.reshape(-1, 2)[tracked_mask]
        destination = current_points.reshape(-1, 2)[tracked_mask]
        finite = np.isfinite(source).all(axis=1) & np.isfinite(destination).all(axis=1)
        source = source[finite]
        destination = destination[finite]
        tracked = len(source)
        if tracked < self._minimum_tracked:
            return self._unknown(
                "insufficient tracked background features", detected=detected, tracked=tracked
            )

        homography, inlier_mask = cv2.findHomography(
            source, destination, cv2.RANSAC, self._ransac_threshold
        )
        if homography is None or inlier_mask is None or not np.isfinite(homography).all():
            return self._unknown("homography estimation failed", detected=detected, tracked=tracked)
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / tracked
        if inlier_count:
            projected = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), homography)
            errors = np.linalg.norm(projected.reshape(-1, 2) - destination[inliers], axis=1)
            mean_error = float(np.mean(errors))
        else:
            mean_error = None
        quality = self._quality(detected, tracked, inlier_count, inlier_ratio, mean_error)
        if inlier_count < self._minimum_inliers:
            return self._unknown("insufficient homography inliers", quality=quality)
        if inlier_ratio < self._minimum_inlier_ratio:
            return self._unknown("homography inlier ratio below threshold", quality=quality)
        if mean_error is None or mean_error > self._maximum_error:
            return self._unknown("homography reprojection error above threshold", quality=quality)
        if abs(float(homography[2, 2])) < 1e-12:
            return self._unknown("homography normalization is unstable", quality=quality)
        normalized = homography / homography[2, 2]
        values = [float(value) for value in normalized.flat]
        return EgoMotionEstimate(
            status=EgoMotionStatus.VALID,
            transform=HomographyTransform(
                values=(
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                )
            ),
            quality=quality,
        )

    @staticmethod
    def _to_gray(frame: np.ndarray[Any, Any], cv2: Any) -> np.ndarray[Any, Any]:
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _quality(
        self,
        detected: int,
        tracked: int,
        inlier_count: int,
        inlier_ratio: float,
        mean_error: float | None,
    ) -> EgoMotionQuality:
        error_score = (
            0.0
            if mean_error is None
            else max(0.0, 1.0 - mean_error / max(self._maximum_error, 1e-12))
        )
        confidence = max(0.0, min(1.0, inlier_ratio * error_score))
        return EgoMotionQuality(
            detected_features=detected,
            tracked_features=tracked,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            mean_reprojection_error=mean_error,
            confidence=confidence,
        )

    def _unknown(
        self,
        reason: str,
        *,
        detected: int = 0,
        tracked: int = 0,
        quality: EgoMotionQuality | None = None,
    ) -> EgoMotionEstimate:
        return EgoMotionEstimate(
            status=EgoMotionStatus.UNKNOWN,
            quality=quality or self._quality(detected, tracked, 0, 0.0, None),
            reason=reason,
        )

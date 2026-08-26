from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.motion.opencv import OpenCVEgoMotionEstimator


def textured_frame(seed: int = 7) -> np.ndarray:
    random = np.random.default_rng(seed)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for x, y in random.integers([10, 10], [310, 230], size=(120, 2)):
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
    return frame


def translate(frame: np.ndarray, x: float, y: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, x], [0.0, 1.0, y]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def test_estimator_recovers_synthetic_camera_translation() -> None:
    previous = textured_frame()
    current = translate(previous, 5, -3)

    result = OpenCVEgoMotionEstimator().estimate(previous, current, [])

    assert result.status is EgoMotionStatus.VALID
    assert result.transform is not None
    assert abs(result.transform.values[2] - 5) < 0.5
    assert abs(result.transform.values[5] + 3) < 0.5
    assert result.quality.inlier_count >= 8
    assert result.quality.inlier_ratio >= 0.5


def test_vehicle_boxes_are_excluded_from_feature_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_mask: np.ndarray | None = None

    class FakeCV2:
        def goodFeaturesToTrack(self, image: np.ndarray, **kwargs: object) -> None:
            nonlocal captured_mask
            mask = kwargs["mask"]
            assert isinstance(mask, np.ndarray)
            captured_mask = mask
            return None

    monkeypatch.setattr("dashcam_ai.motion.opencv._cv2", lambda: FakeCV2())
    frame = np.zeros((100, 120), dtype=np.uint8)
    box = BBox(x1=20, y1=30, x2=60, y2=70)

    result = OpenCVEgoMotionEstimator(mask_padding_pixels=5).estimate(frame, frame, [box])

    assert result.status is EgoMotionStatus.UNKNOWN
    assert captured_mask is not None
    assert np.all(captured_mask[25:75, 15:65] == 0)
    assert captured_mask[0, 0] == 255


def test_low_texture_returns_unknown_without_transform() -> None:
    blank = np.zeros((120, 160, 3), dtype=np.uint8)

    result = OpenCVEgoMotionEstimator().estimate(blank, blank.copy(), [])

    assert result.status is EgoMotionStatus.UNKNOWN
    assert result.transform is None
    assert result.reason == "insufficient background features"


def test_mismatched_frame_dimensions_return_unknown() -> None:
    result = OpenCVEgoMotionEstimator().estimate(
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.zeros((120, 100, 3), dtype=np.uint8),
        [],
    )

    assert result.status is EgoMotionStatus.UNKNOWN
    assert result.reason == "frame dimensions do not match"


def test_quality_gate_rejects_otherwise_valid_transform() -> None:
    previous = textured_frame()
    current = translate(previous, 2, 1)

    result = OpenCVEgoMotionEstimator(minimum_inliers=500).estimate(previous, current, [])

    assert result.status is EgoMotionStatus.UNKNOWN
    assert result.transform is None
    assert result.reason == "insufficient homography inliers"
    assert result.quality.inlier_count > 0


def test_estimate_serializes_to_json() -> None:
    previous = textured_frame()
    result = OpenCVEgoMotionEstimator().estimate(previous, translate(previous, 3, 2), [])

    payload = json.loads(result.model_dump_json())
    assert payload["status"] == "valid"
    assert len(payload["transform"]["values"]) == 9
    assert payload["quality"]["tracked_features"] > 0

import numpy as np
import pytest
from pydantic import ValidationError

from dashcam_ai.application.lane_overlay import annotate_lane_evidence
from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneGeometryStatus
from dashcam_ai.domain.lane_evidence import LaneCurveEvidence, LaneEvidenceFrame
from dashcam_ai.lane.opencv_evidence import OpenCVLaneEvidenceBackend
from dashcam_ai.lane.yolop_onnx import YOLOPOnnxLaneEvidenceBackend


def backend() -> OpenCVLaneEvidenceBackend:
    return OpenCVLaneEvidenceBackend(
        minimum_confidence=0.1,
        minimum_curve_fit_confidence=0.1,
        canny_low_threshold=40,
        canny_high_threshold=120,
        hough_threshold=15,
        minimum_line_length_pixels=20,
        maximum_line_gap_pixels=10,
        minimum_absolute_slope=0.2,
        boundary_cluster_distance_ratio=0.12,
        curve_sample_count=6,
    )


def roi() -> tuple[Point2D, ...]:
    return (
        Point2D(x=0, y=239),
        Point2D(x=100, y=80),
        Point2D(x=220, y=80),
        Point2D(x=319, y=239),
    )


@pytest.mark.cv
def test_opencv_evidence_detects_synthetic_converging_boundaries() -> None:
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.line(frame, (40, 239), (135, 80), (255, 255, 255), 5)
    cv2.line(frame, (280, 239), (185, 80), (255, 255, 255), 5)

    evidence = backend().detect(frame, roi())

    assert evidence.status is LaneGeometryStatus.VALID
    assert len(evidence.curves) >= 2
    assert evidence.estimated_lane_count is None


@pytest.mark.cv
def test_opencv_evidence_returns_unknown_for_blank_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    evidence = backend().detect(frame, roi())

    assert evidence.status is LaneGeometryStatus.UNKNOWN
    assert evidence.confidence == 0
    assert evidence.reason == "no lane-like line evidence"


def test_lane_evidence_models_reject_unsafe_unknown_payload() -> None:
    curve = LaneCurveEvidence(
        evidence_id="curve_0",
        points=(Point2D(x=1, y=1), Point2D(x=2, y=2)),
        confidence=0.8,
        supporting_segments=1,
    )
    with pytest.raises(ValidationError, match="unknown lane evidence"):
        LaneEvidenceFrame(
            status=LaneGeometryStatus.UNKNOWN,
            backend="fake",
            confidence=0,
            curves=(curve,),
            reason="no evidence",
        )


def test_yolop_lane_probability_uses_lane_class_softmax() -> None:
    logits = np.asarray(
        [
            [[2.0, 0.0], [1.0, -1.0]],
            [[0.0, 2.0], [1.0, 3.0]],
        ],
        dtype=np.float32,
    )

    probability = YOLOPOnnxLaneEvidenceBackend._lane_probability(logits)

    assert probability.shape == (2, 2)
    assert probability[0, 0] == pytest.approx(0.1192029)
    assert probability[0, 1] == pytest.approx(0.8807971)
    assert probability[1, 0] == pytest.approx(0.5)
    assert probability[1, 1] == pytest.approx(0.9820138)


@pytest.mark.cv
def test_overlay_draws_status_without_estimating_lane_count() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    evidence = LaneEvidenceFrame(
        status=LaneGeometryStatus.DEGRADED,
        backend="fake",
        confidence=0.5,
        curves=(
            LaneCurveEvidence(
                evidence_id="curve_0",
                points=(Point2D(x=40, y=100), Point2D(x=70, y=40)),
                confidence=0.5,
                supporting_segments=1,
            ),
        ),
        reason="one boundary only",
    )

    annotated = annotate_lane_evidence(
        frame,
        (
            Point2D(x=0, y=119),
            Point2D(x=50, y=30),
            Point2D(x=110, y=30),
            Point2D(x=159, y=119),
        ),
        evidence,
    )

    assert annotated.shape == frame.shape
    assert np.any(annotated != frame)

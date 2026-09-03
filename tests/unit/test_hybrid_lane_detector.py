import numpy as np

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneBoundaryEvidenceSource,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
)
from dashcam_ai.lane.hybrid import HybridLaneDetector


class FakeDetector:
    def __init__(self, geometry: LaneGeometry) -> None:
        self.geometry = geometry

    def detect(self, frame: object, width: int, height: int) -> LaneGeometry:
        return self.geometry


def configured_geometry() -> LaneGeometry:
    lanes = (
        LaneRegion(
            lane_id="lane_0",
            lateral_order=0,
            polygon=(Point2D(x=0, y=0), Point2D(x=50, y=0), Point2D(x=0, y=50)),
        ),
        LaneRegion(
            lane_id="lane_1",
            lateral_order=1,
            polygon=(Point2D(x=50, y=0), Point2D(x=100, y=0), Point2D(x=100, y=50)),
        ),
    )
    return LaneGeometry(
        status=LaneGeometryStatus.VALID,
        provenance=LaneGeometryProvenance.CONFIGURED,
        confidence=1,
        frame_width=100,
        frame_height=50,
        lanes=lanes,
        boundaries=(
            LaneBoundary(
                boundary_id="boundary_0_1",
                left_lane_id="lane_0",
                right_lane_id="lane_1",
                points=(Point2D(x=50, y=0), Point2D(x=50, y=50)),
            ),
        ),
    )


def test_hybrid_fallback_is_explicitly_degraded() -> None:
    unknown = LaneGeometry(
        status=LaneGeometryStatus.UNKNOWN,
        provenance=LaneGeometryProvenance.UNKNOWN,
        confidence=0,
        frame_width=100,
        frame_height=50,
        reason="no dynamic evidence",
    )
    detector = HybridLaneDetector(FakeDetector(unknown), FakeDetector(configured_geometry()))

    geometry = detector.detect(np.zeros((50, 100, 3)), 100, 50)

    assert geometry.status is LaneGeometryStatus.DEGRADED
    assert geometry.provenance is LaneGeometryProvenance.HYBRID
    assert geometry.boundaries[0].evidence_source is (
        LaneBoundaryEvidenceSource.CONFIGURED_FALLBACK
    )


def test_hybrid_preserves_available_dynamic_geometry() -> None:
    dynamic = configured_geometry().model_copy(
        update={"provenance": LaneGeometryProvenance.DYNAMIC}
    )
    detector = HybridLaneDetector(FakeDetector(dynamic), FakeDetector(configured_geometry()))

    assert detector.detect(np.zeros((50, 100, 3)), 100, 50) is dynamic

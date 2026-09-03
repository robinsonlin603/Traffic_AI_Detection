import pytest

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneBoundaryEvidenceSource,
    LaneGeometryStatus,
)
from dashcam_ai.domain.lane_evidence import LaneCurveEvidence, LaneEvidenceFrame
from dashcam_ai.lane.dynamic import TemporalLaneGeometryTracker

WIDTH = 1000
HEIGHT = 600
ROAD_ROI = (
    Point2D(x=100, y=600),
    Point2D(x=350, y=200),
    Point2D(x=650, y=200),
    Point2D(x=900, y=600),
)


def tracker(
    *, confirmation_frames: int = 3, missing_tolerance: int = 2
) -> TemporalLaneGeometryTracker:
    return TemporalLaneGeometryTracker(
        smoothing_alpha=0.5,
        maximum_boundary_jump_ratio=0.08,
        missing_frame_tolerance=missing_tolerance,
        topology_confirmation_frames=confirmation_frames,
        maximum_lane_count=3,
        minimum_boundary_separation_ratio=0.08,
        curve_sample_count=5,
        minimum_confidence=0.7,
    )


def curve(
    identifier: str, bottom_x: float, top_x: float, confidence: float = 0.9
) -> LaneCurveEvidence:
    return LaneCurveEvidence(
        evidence_id=identifier,
        points=(
            Point2D(x=top_x, y=200),
            Point2D(x=(top_x + bottom_x) / 2, y=400),
            Point2D(x=bottom_x, y=600),
        ),
        confidence=confidence,
        supporting_segments=20,
    )


def evidence(
    *curves: LaneCurveEvidence,
    status: LaneGeometryStatus = LaneGeometryStatus.VALID,
) -> LaneEvidenceFrame:
    return LaneEvidenceFrame(
        status=status,
        backend="fake",
        confidence=sum(item.confidence for item in curves) / len(curves),
        curves=curves,
        reason="low confidence" if status is LaneGeometryStatus.DEGRADED else None,
    )


def unknown() -> LaneEvidenceFrame:
    return LaneEvidenceFrame(
        status=LaneGeometryStatus.UNKNOWN,
        backend="fake",
        confidence=0,
        reason="no markings",
    )


def update(subject: TemporalLaneGeometryTracker, frame: LaneEvidenceFrame):
    return subject.update(frame, width=WIDTH, height=HEIGHT, road_roi=ROAD_ROI)


def test_initial_topology_requires_repeated_evidence() -> None:
    subject = tracker(confirmation_frames=3)
    frame = evidence(curve("center", 500, 500))

    first = update(subject, frame)
    second = update(subject, frame)
    third = update(subject, frame)

    assert first.status is LaneGeometryStatus.DEGRADED
    assert first.topology_id is None
    assert second.status is LaneGeometryStatus.DEGRADED
    assert third.status is LaneGeometryStatus.VALID
    assert third.topology_id == "topology-1"
    assert len(third.lanes) == 2
    assert [lane.lane_id for lane in third.lanes] == ["lane_0", "lane_1"]
    assert third.boundaries[0].evidence_source is LaneBoundaryEvidenceSource.OBSERVED


def test_two_internal_boundaries_form_three_lane_topology() -> None:
    subject = tracker(confirmation_frames=2)
    frame = evidence(
        curve("left", 380, 450),
        curve("right", 650, 550),
    )

    update(subject, frame)
    geometry = update(subject, frame)

    assert geometry.status is LaneGeometryStatus.VALID
    assert len(geometry.lanes) == 3
    assert len(geometry.boundaries) == 2
    assert geometry.boundaries[0].left_lane_id == "lane_0"
    assert geometry.boundaries[1].right_lane_id == "lane_2"


def test_fragmented_components_are_merged_before_topology_count() -> None:
    subject = tracker(confirmation_frames=1)
    frame = evidence(
        curve("fragment-a", 500, 500),
        curve("fragment-b", 530, 520),
    )

    geometry = update(subject, frame)

    assert geometry.status is LaneGeometryStatus.VALID
    assert len(geometry.boundaries) == 1
    assert len(geometry.lanes) == 2


def test_short_missing_gap_is_inferred_then_expires_to_unknown() -> None:
    subject = tracker(confirmation_frames=1, missing_tolerance=2)
    stable = update(subject, evidence(curve("center", 500, 500)))

    first_gap = update(subject, unknown())
    second_gap = update(subject, unknown())
    expired = update(subject, unknown())

    assert stable.status is LaneGeometryStatus.VALID
    assert first_gap.status is LaneGeometryStatus.DEGRADED
    assert second_gap.status is LaneGeometryStatus.DEGRADED
    assert all(
        boundary.evidence_source is LaneBoundaryEvidenceSource.INFERRED
        for boundary in second_gap.boundaries
    )
    assert expired.status is LaneGeometryStatus.UNKNOWN
    assert not expired.lanes


def test_abrupt_boundary_jump_does_not_replace_stable_topology() -> None:
    subject = tracker(confirmation_frames=1, missing_tolerance=1)
    stable = update(subject, evidence(curve("center", 500, 500)))

    jumped = update(subject, evidence(curve("jumped", 750, 750)))
    expired = update(subject, evidence(curve("jumped", 750, 750)))

    assert stable.status is LaneGeometryStatus.VALID
    assert jumped.status is LaneGeometryStatus.DEGRADED
    assert jumped.reason == "abrupt boundary jump rejected"
    assert jumped.boundaries[0].points[-1].x == pytest.approx(500)
    assert expired.status is LaneGeometryStatus.UNKNOWN

    pending_recovery = update(subject, evidence(curve("jumped", 750, 750)))

    assert pending_recovery.status is LaneGeometryStatus.VALID
    assert pending_recovery.topology_version == 2


@pytest.mark.parametrize("loss_kind", ["missing", "jump"])
def test_reacquisition_waits_without_exposing_old_topology_version(loss_kind: str) -> None:
    subject = tracker(confirmation_frames=6, missing_tolerance=1)
    original = evidence(curve("center", 500, 500))
    recovered = evidence(curve("new", 750, 750))
    for _ in range(6):
        stable = update(subject, original)
    assert stable.topology_version == 1

    loss = unknown() if loss_kind == "missing" else recovered
    assert update(subject, loss).status is LaneGeometryStatus.DEGRADED
    expired = update(subject, loss)
    assert expired.status is LaneGeometryStatus.UNKNOWN
    assert expired.topology_id is None
    assert expired.topology_version is None

    for _ in range(5):
        pending = update(subject, recovered)
        assert pending.status is LaneGeometryStatus.DEGRADED
        assert pending.topology_id is None
        assert pending.topology_version is None
        payload = pending.model_dump(mode="json")
        assert payload["topology_id"] is None
        assert payload["topology_version"] is None

    confirmed = update(subject, recovered)
    assert confirmed.status is LaneGeometryStatus.VALID
    assert confirmed.topology_id == "topology-2"
    assert confirmed.topology_version == 2


def test_lane_count_change_waits_before_new_topology_version() -> None:
    subject = tracker(confirmation_frames=2)
    two_lanes = evidence(curve("center", 500, 500))
    three_lanes = evidence(
        curve("left", 400, 460),
        curve("right", 650, 540),
    )
    update(subject, two_lanes)
    original = update(subject, two_lanes)

    pending = update(subject, three_lanes)
    changed = update(subject, three_lanes)

    assert original.topology_version == 1
    assert len(pending.lanes) == 2
    assert pending.status is LaneGeometryStatus.DEGRADED
    assert pending.reason == "lane topology change awaiting confirmation"
    assert changed.status is LaneGeometryStatus.VALID
    assert changed.topology_version == 2
    assert len(changed.lanes) == 3


def test_degraded_observation_cannot_produce_valid_geometry() -> None:
    subject = tracker(confirmation_frames=1)

    geometry = update(
        subject,
        evidence(
            curve("center", 500, 500, confidence=0.6),
            status=LaneGeometryStatus.DEGRADED,
        ),
    )

    assert geometry.status is LaneGeometryStatus.DEGRADED
    assert geometry.confidence == pytest.approx(0.6)

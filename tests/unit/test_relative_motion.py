from __future__ import annotations

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import (
    EgoMotionEstimate,
    EgoMotionQuality,
    EgoMotionStatus,
    HomographyTransform,
    RelativeMotionEvidence,
    RelativeMotionStatus,
)
from dashcam_ai.domain.temporal import LaneChangeStatus
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.motion.relative import RelativeMotionEvaluator, summarize_relative_motion


def ego_motion(values: tuple[float, ...]) -> EgoMotionEstimate:
    return EgoMotionEstimate(
        status=EgoMotionStatus.VALID,
        transform=HomographyTransform(values=values),  # type: ignore[arg-type]
        quality=EgoMotionQuality(
            detected_features=50,
            tracked_features=45,
            inlier_count=40,
            inlier_ratio=40 / 45,
            mean_reprojection_error=0.2,
            confidence=0.9,
        ),
    )


def identity_motion() -> EgoMotionEstimate:
    return ego_motion((1, 0, 0, 0, 1, 0, 0, 0, 1))


def evidence(lateral: float, *, stationary: bool = False) -> RelativeMotionEvidence:
    previous = Point2D(x=100, y=100)
    current = Point2D(x=100 + lateral * 1000, y=100)
    displacement = Point2D(x=lateral * 1000, y=0)
    return RelativeMotionEvidence(
        status=RelativeMotionStatus.VALID,
        previous_anchor=previous,
        current_anchor=current,
        predicted_background_anchor=previous,
        observed_displacement=displacement,
        predicted_background_displacement=Point2D(x=0, y=0),
        compensated_displacement=displacement,
        normalized_lateral_displacement=lateral,
        normalized_longitudinal_displacement=0,
        stationary=stationary,
        scene_consistent=True,
        confidence=0.9,
    )


def lane_feature(distance: float) -> LaneMembershipFeature:
    return LaneMembershipFeature(
        membership=LaneMembership.INSIDE if distance > 0 else LaneMembership.OUTSIDE,
        anchor=Point2D(x=100, y=200),
        signed_boundary_distance=distance,
        nearest_boundary_id="left",
        geometry_confidence=1,
    )


def test_camera_translation_is_removed_from_stationary_track() -> None:
    evaluator = RelativeMotionEvaluator()
    motion = ego_motion((1, 0, 10, 0, 1, 5, 0, 0, 1))

    result = evaluator.evaluate(
        Point2D(x=100, y=100), Point2D(x=110, y=105), motion, 1000, 500
    )

    assert result.status is RelativeMotionStatus.VALID
    assert result.compensated_displacement == Point2D(x=0, y=0)
    assert result.stationary is True


def test_real_lateral_motion_remains_after_camera_compensation() -> None:
    evaluator = RelativeMotionEvaluator()
    motion = ego_motion((1, 0, 10, 0, 1, 5, 0, 0, 1))

    result = evaluator.evaluate(
        Point2D(x=100, y=100), Point2D(x=130, y=105), motion, 1000, 500
    )

    assert result.compensated_displacement == Point2D(x=20, y=0)
    assert result.normalized_lateral_displacement == 0.02
    assert result.stationary is False


def test_invalid_or_unstable_homography_returns_unknown() -> None:
    evaluator = RelativeMotionEvaluator()
    unknown = EgoMotionEstimate(
        status=EgoMotionStatus.UNKNOWN,
        quality=EgoMotionQuality(
            detected_features=0,
            tracked_features=0,
            inlier_count=0,
            inlier_ratio=0,
            confidence=0,
        ),
        reason="unavailable",
    )
    missing = evaluator.evaluate(None, Point2D(x=100, y=100), unknown, 1000, 500)
    unstable = evaluator.evaluate(
        Point2D(x=100, y=100),
        Point2D(x=100, y=100),
        ego_motion((1, 0, 0, 0, 1, 0, 0, 0, 0)),
        1000,
        500,
    )

    assert missing.status is RelativeMotionStatus.UNKNOWN
    assert unstable.status is RelativeMotionStatus.UNKNOWN
    assert unstable.reason == "homography projection is unstable"


def test_scene_consistency_rejects_consensus_mass_motion() -> None:
    evaluator = RelativeMotionEvaluator(
        scene_minimum_tracks=3,
        scene_lateral_motion_ratio=0.001,
        scene_consensus_ratio=0.75,
    )

    result = evaluator.apply_scene_consistency(
        {1: evidence(0.01), 2: evidence(0.02), 3: evidence(0.015)}
    )

    assert all(item.scene_consistent is False for item in result.values())


def test_relative_motion_summary_requires_progress_and_non_stationary_evidence() -> None:
    from dashcam_ai.domain.temporal import LanePosition, ManeuverRelation

    supported = summarize_relative_motion(
        [evidence(0.002), evidence(0.002)],
        ManeuverRelation.ENTERING_EGO,
        LanePosition.LEFT_ADJACENT,
        LanePosition.EGO,
        minimum_valid_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
        minimum_directional_consistency=0.6,
        minimum_scene_consistency=0.8,
        maximum_stationary_ratio=0.5,
    )
    stationary = summarize_relative_motion(
        [evidence(0, stationary=True), evidence(0, stationary=True)],
        ManeuverRelation.ENTERING_EGO,
        LanePosition.LEFT_ADJACENT,
        LanePosition.EGO,
        minimum_valid_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
        minimum_directional_consistency=0.6,
        minimum_scene_consistency=0.8,
        maximum_stationary_ratio=0.5,
    )

    assert supported.supported is True
    assert stationary.supported is False
    assert stationary.reason == "vehicle is stationary relative to the background"


def test_temporal_confirmation_requires_supported_relative_motion() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
        require_relative_motion=True,
        minimum_relative_motion_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
    )
    distances = [-80, -20, -5, 25, 30]
    states = [
        tracker.update(
            1,
            frame_id,
            frame_id * 0.1,
            lane_feature(distance),
            EgoMotionStatus.VALID,
            evidence(0.002),
        )
        for frame_id, distance in enumerate(distances)
    ]

    assert states[-1].status is LaneChangeStatus.CONFIRMED
    assert states[-1].relative_motion is not None
    assert states[-1].relative_motion.supported is True


def test_stationary_relative_motion_cannot_confirm_lane_change() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
        require_relative_motion=True,
        minimum_relative_motion_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
    )
    distances = [-80, -20, -5, 25, 30]
    states = [
        tracker.update(
            1,
            frame_id,
            frame_id * 0.1,
            lane_feature(distance),
            EgoMotionStatus.VALID,
            evidence(0, stationary=True),
        )
        for frame_id, distance in enumerate(distances)
    ]

    assert states[-1].status is LaneChangeStatus.CANDIDATE
    assert states[-1].relative_motion is not None
    assert states[-1].relative_motion.reason == (
        "vehicle is stationary relative to the background"
    )

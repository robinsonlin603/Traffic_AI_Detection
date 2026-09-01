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
from dashcam_ai.motion.relative import (
    RelativeMotionEvaluator,
    summarize_lane_relative_motion,
)


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


def general_lane_feature(
    membership: LaneMembership, lane_id: str, distance: float = 30
) -> LaneMembershipFeature:
    return LaneMembershipFeature(
        membership=membership,
        anchor=Point2D(x=100, y=200),
        lane_id=lane_id,
        lane_lateral_order={"lane_center": 1, "lane_right": 2}[lane_id],
        boundary_lane_ids=("lane_center", "lane_right"),
        boundary_lane_orders=(1, 2),
        signed_boundary_distance=distance,
        nearest_boundary_id="boundary_center_right",
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


def test_general_lane_relative_motion_uses_source_and_target_order() -> None:
    moving_right = summarize_lane_relative_motion(
        [evidence(0.002), evidence(0.002)],
        1,
        2,
        minimum_valid_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
        minimum_directional_consistency=0.6,
        minimum_scene_consistency=0.8,
        maximum_stationary_ratio=0.5,
    )
    wrong_direction = summarize_lane_relative_motion(
        [evidence(0.002), evidence(0.002)],
        1,
        0,
        minimum_valid_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
        minimum_directional_consistency=0.6,
        minimum_scene_consistency=0.8,
        maximum_stationary_ratio=0.5,
    )

    assert moving_right.supported is True
    assert wrong_direction.supported is False
    assert wrong_direction.reason == (
        "relative lateral direction is incompatible with the maneuver"
    )


def test_general_lane_change_requires_compensated_motion_support() -> None:
    tracker = TemporalLaneTracker(
        smoothing_window_frames=1,
        debounce_frames=1,
        minimum_confirmation_frames=2,
        minimum_confirmation_duration_seconds=0.1,
        require_relative_motion=True,
        minimum_relative_motion_observations=2,
        minimum_cumulative_lateral_ratio=0.003,
    )
    observations = [
        general_lane_feature(LaneMembership.INSIDE_LANE, "lane_center"),
        general_lane_feature(LaneMembership.NEAR_BOUNDARY, "lane_center", 2),
        general_lane_feature(LaneMembership.NEAR_BOUNDARY, "lane_right", 1),
        general_lane_feature(LaneMembership.INSIDE_LANE, "lane_right"),
        general_lane_feature(LaneMembership.INSIDE_LANE, "lane_right"),
    ]
    states = [
        tracker.update(
            8,
            frame_id,
            frame_id * 0.1,
            item,
            EgoMotionStatus.VALID,
            evidence(0, stationary=True),
        )
        for frame_id, item in enumerate(observations)
    ]

    assert states[-1].lane_change_status is LaneChangeStatus.CANDIDATE
    assert states[-1].relative_motion is not None
    assert states[-1].relative_motion.supported is False
    assert states[-1].completed_frame is None

"""以背景 homography 計算 Track 的 image-space 相對運動。"""

from __future__ import annotations

from math import hypot, isfinite

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.motion import (
    EgoMotionEstimate,
    EgoMotionStatus,
    RelativeMotionEvidence,
    RelativeMotionStatus,
    RelativeMotionSummary,
)
from dashcam_ai.domain.temporal import LanePosition, ManeuverRelation


class RelativeMotionEvaluator:
    """投影背景錨點並保留扣除相機運動後的位移。"""

    def __init__(
        self,
        *,
        stationary_residual_ratio: float = 0.001,
        maximum_projection_margin_ratio: float = 0.25,
        scene_minimum_tracks: int = 3,
        scene_lateral_motion_ratio: float = 0.003,
        scene_consensus_ratio: float = 0.75,
    ) -> None:
        if stationary_residual_ratio < 0:
            raise ValueError("stationary_residual_ratio must not be negative")
        if maximum_projection_margin_ratio < 0:
            raise ValueError("maximum_projection_margin_ratio must not be negative")
        if scene_minimum_tracks < 2:
            raise ValueError("scene_minimum_tracks must be at least two")
        if scene_lateral_motion_ratio <= 0:
            raise ValueError("scene_lateral_motion_ratio must be positive")
        if not 0.5 <= scene_consensus_ratio <= 1:
            raise ValueError("scene_consensus_ratio must be between 0.5 and one")
        self._stationary_ratio = stationary_residual_ratio
        self._projection_margin = maximum_projection_margin_ratio
        self._scene_minimum_tracks = scene_minimum_tracks
        self._scene_lateral_ratio = scene_lateral_motion_ratio
        self._scene_consensus_ratio = scene_consensus_ratio

    def evaluate(
        self,
        previous_anchor: Point2D | None,
        current_anchor: Point2D,
        ego_motion: EgoMotionEstimate,
        frame_width: int,
        frame_height: int,
    ) -> RelativeMotionEvidence:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if previous_anchor is None:
            return self._unknown(current_anchor, "previous track anchor unavailable")
        if ego_motion.status is not EgoMotionStatus.VALID or ego_motion.transform is None:
            return self._unknown(current_anchor, "ego-motion transform unavailable")
        values = ego_motion.transform.values
        denominator = values[6] * previous_anchor.x + values[7] * previous_anchor.y + values[8]
        if not isfinite(denominator) or abs(denominator) < 1e-12:
            return self._unknown(current_anchor, "homography projection is unstable")
        predicted_x = (
            values[0] * previous_anchor.x
            + values[1] * previous_anchor.y
            + values[2]
        ) / denominator
        predicted_y = (
            values[3] * previous_anchor.x
            + values[4] * previous_anchor.y
            + values[5]
        ) / denominator
        coordinates = (
            predicted_x,
            predicted_y,
            previous_anchor.x,
            previous_anchor.y,
            current_anchor.x,
            current_anchor.y,
        )
        if not all(isfinite(value) for value in coordinates):
            return self._unknown(current_anchor, "homography projection is not finite")
        margin_x = frame_width * self._projection_margin
        margin_y = frame_height * self._projection_margin
        if not (
            -margin_x <= predicted_x <= frame_width + margin_x
            and -margin_y <= predicted_y <= frame_height + margin_y
        ):
            return self._unknown(current_anchor, "homography projection is outside safe bounds")

        predicted = Point2D(x=predicted_x, y=predicted_y)
        observed_displacement = Point2D(
            x=current_anchor.x - previous_anchor.x,
            y=current_anchor.y - previous_anchor.y,
        )
        background_displacement = Point2D(
            x=predicted.x - previous_anchor.x,
            y=predicted.y - previous_anchor.y,
        )
        compensated = Point2D(
            x=current_anchor.x - predicted.x,
            y=current_anchor.y - predicted.y,
        )
        normalized_x = compensated.x / frame_width
        normalized_y = compensated.y / frame_height
        residual_ratio = hypot(normalized_x, normalized_y)
        return RelativeMotionEvidence(
            status=RelativeMotionStatus.VALID,
            previous_anchor=previous_anchor,
            current_anchor=current_anchor,
            predicted_background_anchor=predicted,
            observed_displacement=observed_displacement,
            predicted_background_displacement=background_displacement,
            compensated_displacement=compensated,
            normalized_lateral_displacement=normalized_x,
            normalized_longitudinal_displacement=normalized_y,
            stationary=residual_ratio <= self._stationary_ratio,
            scene_consistent=True,
            confidence=ego_motion.quality.confidence,
        )

    def apply_scene_consistency(
        self, evidences: dict[int, RelativeMotionEvidence]
    ) -> dict[int, RelativeMotionEvidence]:
        significant = [
            item
            for item in evidences.values()
            if item.status is RelativeMotionStatus.VALID
            and item.normalized_lateral_displacement is not None
            and abs(item.normalized_lateral_displacement) >= self._scene_lateral_ratio
        ]
        consistent = True
        if len(significant) >= self._scene_minimum_tracks:
            positive = sum(
                item.normalized_lateral_displacement is not None
                and item.normalized_lateral_displacement > 0
                for item in significant
            )
            consensus = max(positive, len(significant) - positive) / len(significant)
            consistent = consensus < self._scene_consensus_ratio
        return {
            track_id: (
                item.model_copy(update={"scene_consistent": consistent})
                if item.status is RelativeMotionStatus.VALID
                else item
            )
            for track_id, item in evidences.items()
        }

    @staticmethod
    def _unknown(current_anchor: Point2D, reason: str) -> RelativeMotionEvidence:
        return RelativeMotionEvidence(
            status=RelativeMotionStatus.UNKNOWN,
            current_anchor=current_anchor,
            confidence=0,
            reason=reason,
        )


def summarize_relative_motion(
    evidences: list[RelativeMotionEvidence],
    relation: ManeuverRelation,
    from_lane: LanePosition,
    to_lane: LanePosition,
    *,
    minimum_valid_observations: int,
    minimum_cumulative_lateral_ratio: float,
    minimum_directional_consistency: float,
    minimum_scene_consistency: float,
    maximum_stationary_ratio: float,
) -> RelativeMotionSummary:
    """將候選期間位移彙整成 deterministic confirmation gate。"""
    valid = [item for item in evidences if item.status is RelativeMotionStatus.VALID]
    lateral = [
        item.normalized_lateral_displacement
        for item in valid
        if item.normalized_lateral_displacement is not None
    ]
    expected_sign = _expected_lateral_sign(relation, from_lane, to_lane)
    cumulative = float(sum(lateral))
    expected_progress = cumulative * expected_sign if expected_sign is not None else 0.0
    directional = (
        sum(value * expected_sign > 0 for value in lateral) / len(lateral)
        if lateral and expected_sign is not None
        else 0.0
    )
    motion_quality = (
        sum(item.confidence for item in valid) / len(valid) if valid else 0.0
    )
    scene_consistency = (
        sum(item.scene_consistent is True for item in valid) / len(valid) if valid else 0.0
    )
    stationary_ratio = (
        sum(item.stationary is True for item in valid) / len(valid) if valid else 1.0
    )
    progress_score = min(
        1.0,
        max(0.0, expected_progress) / max(minimum_cumulative_lateral_ratio, 1e-12),
    )
    confidence = max(
        0.0,
        min(
            1.0,
            (
                progress_score
                + directional
                + motion_quality
                + scene_consistency
                + (1.0 - stationary_ratio)
            )
            / 5.0,
        ),
    )
    reason: str | None = None
    if expected_sign is None:
        reason = "maneuver direction is unknown"
    elif len(valid) < minimum_valid_observations:
        reason = "relative motion evidence is incomplete"
    elif scene_consistency < minimum_scene_consistency:
        reason = "scene-wide motion is inconsistent with independent lane changes"
    elif stationary_ratio > maximum_stationary_ratio:
        reason = "vehicle is stationary relative to the background"
    elif expected_progress < minimum_cumulative_lateral_ratio:
        reason = "relative lateral progress is below confirmation threshold"
    elif directional < minimum_directional_consistency:
        reason = "relative lateral direction is incompatible with the maneuver"
    return RelativeMotionSummary(
        valid_observations=len(valid),
        cumulative_lateral_displacement=cumulative,
        expected_lateral_progress=expected_progress,
        directional_consistency=directional,
        motion_quality=motion_quality,
        scene_consistency=scene_consistency,
        stationary_ratio=stationary_ratio,
        supported=reason is None,
        confidence=confidence,
        reason=reason,
    )


def _expected_lateral_sign(
    relation: ManeuverRelation,
    from_lane: LanePosition,
    to_lane: LanePosition,
) -> float | None:
    if relation is ManeuverRelation.ENTERING_EGO:
        if from_lane is LanePosition.LEFT_ADJACENT:
            return 1.0
        if from_lane is LanePosition.RIGHT_ADJACENT:
            return -1.0
    if relation is ManeuverRelation.LEAVING_EGO:
        if to_lane is LanePosition.LEFT_ADJACENT:
            return -1.0
        if to_lane is LanePosition.RIGHT_ADJACENT:
            return 1.0
    return None

"""以 forward corridor 與 image-space motion 建立可解釋切入事件。"""

from __future__ import annotations

from dashcam_ai.domain.events import (
    ConfidenceBreakdown,
    CutInEvent,
    EventEvidence,
    EventEvidenceFrame,
    EventStatus,
    LaneChangeEvent,
)
from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.lane import LaneMembership
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.events.corridor import ForwardCorridor


class CutInDetector:
    """不使用真實距離或 TTC 的 deterministic image-space heuristic。"""

    def __init__(
        self,
        *,
        minimum_bbox_expansion_ratio: float = 0.05,
        minimum_confirmed_confidence: float = 0.65,
        minimum_motion_quality_ratio: float = 0.8,
        lane_change_weight: float = 0.4,
        corridor_weight: float = 0.3,
        bbox_expansion_weight: float = 0.15,
        motion_quality_weight: float = 0.15,
    ) -> None:
        if minimum_bbox_expansion_ratio <= 0:
            raise ValueError("minimum_bbox_expansion_ratio must be positive")
        if not 0 <= minimum_confirmed_confidence <= 1:
            raise ValueError("minimum_confirmed_confidence must be between zero and one")
        if not 0 <= minimum_motion_quality_ratio <= 1:
            raise ValueError("minimum_motion_quality_ratio must be between zero and one")
        weights = (
            lane_change_weight,
            corridor_weight,
            bbox_expansion_weight,
            motion_quality_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("confidence weights must be non-negative with a positive sum")
        total = sum(weights)
        self._minimum_expansion = minimum_bbox_expansion_ratio
        self._minimum_confidence = minimum_confirmed_confidence
        self._minimum_motion_quality = minimum_motion_quality_ratio
        self._weights = tuple(weight / total for weight in weights)

    def detect(
        self,
        lane_change: LaneChangeEvent,
        current_bbox: BBox,
        previous_bbox: BBox | None,
        corridor: ForwardCorridor,
    ) -> CutInEvent:
        corridor_score = corridor.interaction_score(current_bbox)
        expansion_ratio = self._expansion_ratio(previous_bbox, current_bbox)
        expansion_score = (
            0.0
            if expansion_ratio is None
            else min(1.0, max(0.0, expansion_ratio) / self._minimum_expansion)
        )
        frames = lane_change.evidence.frames
        valid_motion = sum(
            item.ego_motion_status is EgoMotionStatus.VALID for item in frames
        )
        motion_score = valid_motion / len(frames) if frames else 0.0
        lane_score = {
            EventStatus.CONFIRMED: 1.0,
            EventStatus.CANDIDATE: 0.5,
            EventStatus.REJECTED: 0.0,
        }[lane_change.status]
        overall = sum(
            score * weight
            for score, weight in zip(
                (lane_score, corridor_score, expansion_score, motion_score),
                self._weights,
                strict=True,
            )
        )
        status, reason = self._status(
            lane_change.status, corridor_score, expansion_score, motion_score, overall
        )
        latest = frames[-1] if frames else None
        current_evidence = EventEvidenceFrame(
            frame_id=lane_change.end_frame,
            timestamp=lane_change.end_timestamp,
            membership=latest.membership if latest is not None else LaneMembership.UNKNOWN,
            signed_boundary_distance=(
                latest.signed_boundary_distance if latest is not None else None
            ),
            smoothed_signed_boundary_distance=(
                latest.smoothed_signed_boundary_distance if latest is not None else None
            ),
            ego_motion_status=(
                latest.ego_motion_status if latest is not None else EgoMotionStatus.UNKNOWN
            ),
            bbox=current_bbox,
        )
        return CutInEvent(
            event_id=f"cut-in:{lane_change.track_id}:{lane_change.start_frame}",
            lane_change_event_id=lane_change.event_id,
            status=status,
            track_id=lane_change.track_id,
            frame_id=lane_change.end_frame,
            timestamp=lane_change.end_timestamp,
            anchor=current_bbox.bottom_center,
            corridor_interaction=corridor_score == 1.0,
            bbox_expansion_ratio=expansion_ratio,
            confidence=ConfidenceBreakdown(
                lane_change=lane_score,
                corridor_interaction=corridor_score,
                bbox_expansion=expansion_score,
                motion_quality=motion_score,
                overall=overall,
            ),
            evidence=EventEvidence(
                boundary_id=lane_change.evidence.boundary_id,
                frames=(*frames, current_evidence),
            ),
            reason=reason,
        )

    @staticmethod
    def _expansion_ratio(previous: BBox | None, current: BBox) -> float | None:
        if previous is None or previous.area <= 0:
            return None
        return current.area / previous.area - 1.0

    def _status(
        self,
        lane_status: EventStatus,
        corridor_score: float,
        expansion_score: float,
        motion_score: float,
        overall: float,
    ) -> tuple[EventStatus, str | None]:
        if lane_status is EventStatus.REJECTED:
            return EventStatus.REJECTED, "lane change was rejected"
        if corridor_score < 1.0:
            return EventStatus.REJECTED, "vehicle bottom-center is outside forward corridor"
        if expansion_score <= 0:
            return EventStatus.REJECTED, "bbox does not show forward image-space interaction"
        if expansion_score < 1.0:
            return EventStatus.CANDIDATE, "bbox expansion is below confirmation threshold"
        if motion_score < self._minimum_motion_quality:
            return EventStatus.CANDIDATE, "ego-motion evidence is incomplete"
        if lane_status is EventStatus.CONFIRMED and overall >= self._minimum_confidence:
            return EventStatus.CONFIRMED, None
        return EventStatus.CANDIDATE, "cut-in confidence is below confirmation threshold"

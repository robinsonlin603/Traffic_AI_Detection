"""將一般 temporal lane state 轉成結構化換道事件。"""

from __future__ import annotations

from dashcam_ai.domain.events import (
    ConfidenceBreakdown,
    EventEvidence,
    EventEvidenceFrame,
    EventStatus,
    LaneChangeEvent,
)
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import LaneChangeStatus, TemporalLaneState


class LaneChangeEventBuilder:
    def __init__(self, evidence_history_size: int = 12) -> None:
        if evidence_history_size <= 0:
            raise ValueError("evidence_history_size must be positive")
        self._history_size = evidence_history_size

    def build(self, state: TemporalLaneState) -> LaneChangeEvent | None:
        if (
            state.lane_change_status is LaneChangeStatus.IDLE
            or state.candidate_started_frame is None
            or state.candidate_started_timestamp is None
            or state.source_lane is None
            or state.target_lane is None
        ):
            return None
        history = tuple(
            item for item in state.history if item.frame_id >= state.candidate_started_frame
        )[-self._history_size :]
        frames = tuple(
            EventEvidenceFrame(
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                membership=item.membership,
                lane_id=item.lane_id,
                stable_lane_id=item.stable_lane_id,
                anchor=item.anchor,
                signed_boundary_distance=item.signed_boundary_distance,
                smoothed_signed_boundary_distance=item.smoothed_signed_boundary_distance,
                ego_motion_status=item.ego_motion_status,
                relative_motion=item.relative_motion,
            )
            for item in history
        )
        motion_ratio = (
            sum(item.ego_motion_status is EgoMotionStatus.VALID for item in frames) / len(frames)
            if frames
            else 0.0
        )
        target_ratio = (
            sum(item.stable_lane_id == state.target_lane for item in frames) / len(frames)
            if frames
            else 0.0
        )
        timeline_score = (
            1.0
            if state.completed_frame is not None
            else 0.7
            if state.boundary_crossed_frame is not None
            else 0.4
        )
        relative_score = state.relative_motion.confidence if state.relative_motion else 0.0
        overall = min(
            1.0,
            0.3 * timeline_score + 0.25 * target_ratio + 0.25 * motion_ratio + 0.2 * relative_score,
        )
        breakdown = ConfidenceBreakdown(
            timeline=timeline_score,
            membership_stability=target_ratio,
            ego_motion=motion_ratio,
            relative_motion=relative_score,
            overall=overall,
        )
        return LaneChangeEvent(
            event_id=f"lane-change:{state.track_id}:{state.candidate_started_frame}",
            status=EventStatus(state.lane_change_status.value),
            track_id=state.track_id,
            source_lane=state.source_lane,
            target_lane=state.target_lane,
            direction=state.direction,
            started_frame=state.candidate_started_frame,
            started_at=state.candidate_started_timestamp,
            lane_crossed_frame=state.boundary_crossed_frame,
            lane_crossed_at=state.boundary_crossed_timestamp,
            completed_frame=state.completed_frame,
            completed_at=state.completed_timestamp,
            confidence=overall,
            confidence_breakdown=breakdown,
            relative_motion=state.relative_motion,
            evidence=EventEvidence(
                boundary_id=state.boundary_id,
                frame_ids=tuple(item.frame_id for item in frames),
                trajectory=tuple(item.anchor for item in frames),
                frames=frames,
            ),
            reason=state.reason,
        )

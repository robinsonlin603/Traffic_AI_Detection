"""將 temporal lane state 轉成結構化換道事件。"""

from __future__ import annotations

from dashcam_ai.domain.events import (
    EventEvidence,
    EventEvidenceFrame,
    EventStatus,
    LaneChangeEvent,
)
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import LaneChangeStatus, TemporalLaneState


class LaneChangeEventBuilder:
    """保留候選、確認與拒絕狀態，不替 idle state 建立事件。"""

    def __init__(self, evidence_history_size: int = 12) -> None:
        if evidence_history_size <= 0:
            raise ValueError("evidence_history_size must be positive")
        self._history_size = evidence_history_size

    def build(self, state: TemporalLaneState) -> LaneChangeEvent | None:
        if state.status is LaneChangeStatus.IDLE or state.candidate_started_frame is None:
            return None
        status = EventStatus(state.status.value)
        candidate_history = tuple(
            item for item in state.history if item.frame_id >= state.candidate_started_frame
        )
        frames = tuple(
            EventEvidenceFrame(
                frame_id=item.frame_id,
                timestamp=item.timestamp,
                membership=item.membership,
                signed_boundary_distance=item.signed_boundary_distance,
                smoothed_signed_boundary_distance=item.smoothed_signed_boundary_distance,
                ego_motion_status=item.ego_motion_status,
                relative_motion=item.relative_motion,
            )
            for item in candidate_history[-self._history_size :]
        )
        valid_motion = sum(
            item.ego_motion_status is EgoMotionStatus.VALID for item in frames
        )
        motion_ratio = valid_motion / len(frames) if frames else 0.0
        relative_score = (
            state.relative_motion.confidence if state.relative_motion is not None else motion_ratio
        )
        latest_motion_valid = bool(frames) and (
            frames[-1].ego_motion_status is EgoMotionStatus.VALID
        )
        if status is EventStatus.CONFIRMED and not latest_motion_valid:
            status = EventStatus.CANDIDATE
        status_score = {
            EventStatus.CANDIDATE: 0.5,
            EventStatus.CONFIRMED: 1.0,
            EventStatus.REJECTED: 0.0,
        }[status]
        confidence = min(
            1.0, 0.5 * status_score + 0.25 * motion_ratio + 0.25 * relative_score
        )
        return LaneChangeEvent(
            event_id=f"lane-change:{state.track_id}:{state.candidate_started_frame}",
            status=status,
            track_id=state.track_id,
            start_frame=state.candidate_started_frame,
            end_frame=state.frame_id,
            start_timestamp=(
                state.candidate_started_timestamp
                if state.candidate_started_timestamp is not None
                else 0.0
            ),
            end_timestamp=state.timestamp,
            confidence=confidence,
            maneuver_relation=state.maneuver_relation,
            from_lane=state.from_lane,
            to_lane=state.to_lane,
            relative_motion=state.relative_motion,
            evidence=EventEvidence(boundary_id=state.boundary_id, frames=frames),
            reason=(
                "current ego-motion evidence is invalid"
                if state.status is LaneChangeStatus.CONFIRMED and status is EventStatus.CANDIDATE
                else state.reason
            ),
        )

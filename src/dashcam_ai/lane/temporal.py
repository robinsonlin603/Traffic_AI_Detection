"""以平滑、遲滯與 debounce 維護每個 track 的換道狀態。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import (
    EgoMotionStatus,
    RelativeMotionEvidence,
    RelativeMotionSummary,
)
from dashcam_ai.domain.temporal import (
    LaneChangeDirection,
    LaneChangePhase,
    LaneChangeStatus,
    TemporalLaneObservation,
    TemporalLaneState,
)
from dashcam_ai.motion.relative import summarize_lane_relative_motion


@dataclass(slots=True)
class _TrackState:
    stable_lane_id: str | None = None
    pending_lane_id: str | None = None
    pending_lane_count: int = 0
    lane_change_phase: LaneChangePhase = LaneChangePhase.UNKNOWN
    lane_change_status: LaneChangeStatus = LaneChangeStatus.IDLE
    source_lane: str | None = None
    target_lane: str | None = None
    source_lane_order: int | None = None
    target_lane_order: int | None = None
    direction: LaneChangeDirection = LaneChangeDirection.UNKNOWN
    general_candidate_frame: int | None = None
    general_candidate_timestamp: float | None = None
    crossed_frame: int | None = None
    crossed_timestamp: float | None = None
    general_entered_frame: int | None = None
    general_entered_timestamp: float | None = None
    completed_frame: int | None = None
    completed_timestamp: float | None = None
    general_entered_count: int = 0
    general_valid_motion_count: int = 0
    general_relative_motion: RelativeMotionSummary | None = None
    missing_count: int = 0
    boundary_id: str | None = None
    reason: str | None = None
    last_frame: int | None = None
    last_timestamp: float | None = None
    distances: deque[float] = field(default_factory=deque)
    history: deque[TemporalLaneObservation] = field(default_factory=deque)


class TemporalLaneTracker:
    """將單幀 lane membership 轉成有界、可解釋的 per-track 狀態。"""

    def __init__(
        self,
        *,
        smoothing_window_frames: int = 3,
        approaching_distance_pixels: float = 40.0,
        entered_distance_pixels: float = 20.0,
        debounce_frames: int = 2,
        minimum_confirmation_frames: int = 3,
        minimum_confirmation_duration_seconds: float = 0.1,
        maximum_missing_frames: int = 2,
        candidate_timeout_seconds: float = 2.0,
        history_size: int = 30,
        require_relative_motion: bool = False,
        minimum_relative_motion_observations: int = 2,
        minimum_cumulative_lateral_ratio: float = 0.003,
        minimum_directional_consistency: float = 0.6,
        minimum_scene_consistency: float = 0.8,
        maximum_stationary_ratio: float = 0.5,
    ) -> None:
        if smoothing_window_frames <= 0:
            raise ValueError("smoothing_window_frames must be positive")
        if approaching_distance_pixels <= 0:
            raise ValueError("approaching_distance_pixels must be positive")
        if entered_distance_pixels <= 0:
            raise ValueError("entered_distance_pixels must be positive")
        if debounce_frames <= 0 or minimum_confirmation_frames <= 0:
            raise ValueError("frame thresholds must be positive")
        if minimum_confirmation_duration_seconds < 0:
            raise ValueError("minimum_confirmation_duration_seconds must not be negative")
        if maximum_missing_frames < 0:
            raise ValueError("maximum_missing_frames must not be negative")
        if candidate_timeout_seconds <= 0:
            raise ValueError("candidate_timeout_seconds must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if minimum_relative_motion_observations <= 0:
            raise ValueError("minimum_relative_motion_observations must be positive")
        if minimum_cumulative_lateral_ratio <= 0:
            raise ValueError("minimum_cumulative_lateral_ratio must be positive")
        ratios = (
            minimum_directional_consistency,
            minimum_scene_consistency,
            maximum_stationary_ratio,
        )
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("relative motion ratios must be between zero and one")
        self._smoothing_window = smoothing_window_frames
        self._approaching_distance = approaching_distance_pixels
        self._entered_distance = entered_distance_pixels
        self._debounce_frames = debounce_frames
        self._minimum_confirmation_frames = minimum_confirmation_frames
        self._minimum_confirmation_duration = minimum_confirmation_duration_seconds
        self._maximum_missing = maximum_missing_frames
        self._candidate_timeout = candidate_timeout_seconds
        self._history_size = history_size
        self._require_relative_motion = require_relative_motion
        self._minimum_relative_observations = minimum_relative_motion_observations
        self._minimum_lateral_ratio = minimum_cumulative_lateral_ratio
        self._minimum_directional_consistency = minimum_directional_consistency
        self._minimum_scene_consistency = minimum_scene_consistency
        self._maximum_stationary_ratio = maximum_stationary_ratio
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        feature: LaneMembershipFeature,
        ego_motion_status: EgoMotionStatus,
        relative_motion: RelativeMotionEvidence | None = None,
    ) -> TemporalLaneState:
        """加入一筆觀察；frame 與 timestamp 對同一 track 必須嚴格遞增。"""
        if track_id < 0 or frame_id < 0 or timestamp < 0:
            raise ValueError("track, frame, and timestamp values must not be negative")
        state = self._tracks.setdefault(track_id, self._new_track())
        if state.last_frame is not None and frame_id <= state.last_frame:
            raise ValueError("frame_id must increase for each track")
        if state.last_timestamp is not None and timestamp <= state.last_timestamp:
            raise ValueError("timestamp must increase for each track")
        state.last_frame = frame_id
        state.last_timestamp = timestamp

        usable = (
            ego_motion_status is EgoMotionStatus.VALID
            and feature.membership is not LaneMembership.UNKNOWN
            and feature.signed_boundary_distance is not None
        )
        if not usable:
            observation = TemporalLaneObservation(
                frame_id=frame_id,
                timestamp=timestamp,
                membership=feature.membership,
                anchor=feature.anchor,
                lane_id=feature.lane_id,
                stable_lane_id=state.stable_lane_id,
                signed_boundary_distance=feature.signed_boundary_distance,
                nearest_boundary_id=feature.nearest_boundary_id,
                ego_motion_status=ego_motion_status,
                relative_motion=relative_motion,
            )
            self._append_history(state, observation)
            self._handle_missing(state, timestamp)
            return self._snapshot(track_id, frame_id, timestamp, state)

        distance = feature.signed_boundary_distance
        assert distance is not None
        state.missing_count = 0
        state.distances.append(distance)
        while len(state.distances) > self._smoothing_window:
            state.distances.popleft()
        smoothed = float(median(state.distances))
        previous_stable_lane = state.stable_lane_id
        self._stabilize_lane(state, feature)
        if feature.nearest_boundary_id is not None:
            state.boundary_id = feature.nearest_boundary_id
        observation = TemporalLaneObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            membership=feature.membership,
            anchor=feature.anchor,
            lane_id=feature.lane_id,
            stable_lane_id=state.stable_lane_id,
            signed_boundary_distance=distance,
            smoothed_signed_boundary_distance=smoothed,
            nearest_boundary_id=feature.nearest_boundary_id,
            ego_motion_status=ego_motion_status,
            relative_motion=relative_motion,
        )
        self._append_history(state, observation)
        self._advance_general(
            state, frame_id, timestamp, feature, previous_stable_lane
        )
        return self._snapshot(track_id, frame_id, timestamp, state)

    def forget(self, track_id: int) -> None:
        """追蹤 ID 永久消失後釋放其 bounded temporal state。"""
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        """拓撲身分改變時清除所有不可跨拓撲沿用的 track state。"""
        self._tracks.clear()

    def _new_track(self) -> _TrackState:
        return _TrackState(
            distances=deque(maxlen=self._smoothing_window),
            history=deque(maxlen=self._history_size),
        )

    def _stabilize_lane(
        self, state: _TrackState, feature: LaneMembershipFeature
    ) -> None:
        if feature.membership is LaneMembership.INSIDE_LANE and feature.lane_id:
            candidate = feature.lane_id
        elif feature.membership is LaneMembership.OUTSIDE_CONFIGURED_LANES:
            candidate = ""
        else:
            return
        if candidate == state.pending_lane_id:
            state.pending_lane_count += 1
        else:
            state.pending_lane_id = candidate
            state.pending_lane_count = 1
        if state.pending_lane_count >= self._debounce_frames:
            state.stable_lane_id = candidate or None

    def _advance_general(
        self,
        state: _TrackState,
        frame_id: int,
        timestamp: float,
        feature: LaneMembershipFeature,
        previous_stable_lane: str | None,
    ) -> None:
        if state.lane_change_status is LaneChangeStatus.CONFIRMED:
            self._rearm_general(state)
        if state.lane_change_status is LaneChangeStatus.IDLE:
            if (
                previous_stable_lane is not None
                and state.stable_lane_id is not None
                and state.stable_lane_id != previous_stable_lane
            ):
                state.lane_change_phase = LaneChangePhase.UNKNOWN
                state.lane_change_status = LaneChangeStatus.UNKNOWN
                state.reason = "lane changed without shared-boundary evidence"
                return
            if state.stable_lane_id is not None:
                state.lane_change_phase = LaneChangePhase.STABLE_IN_LANE
            if (
                feature.membership is LaneMembership.NEAR_BOUNDARY
                and previous_stable_lane is not None
                and feature.boundary_lane_ids is not None
                and feature.boundary_lane_orders is not None
                and previous_stable_lane in feature.boundary_lane_ids
            ):
                index = feature.boundary_lane_ids.index(previous_stable_lane)
                target_index = 1 - index
                state.source_lane = previous_stable_lane
                state.target_lane = feature.boundary_lane_ids[target_index]
                state.source_lane_order = feature.boundary_lane_orders[index]
                state.target_lane_order = feature.boundary_lane_orders[target_index]
                state.direction = (
                    LaneChangeDirection.RIGHT
                    if state.target_lane_order > state.source_lane_order
                    else LaneChangeDirection.LEFT
                )
                state.boundary_id = feature.nearest_boundary_id
                state.general_candidate_frame = frame_id
                state.general_candidate_timestamp = timestamp
                state.general_valid_motion_count = 1
                state.lane_change_phase = LaneChangePhase.APPROACHING_BOUNDARY
                state.lane_change_status = LaneChangeStatus.CANDIDATE
            return
        if state.lane_change_status is LaneChangeStatus.UNKNOWN:
            if feature.membership is LaneMembership.INSIDE_LANE:
                self._rearm_general(state)
            return
        if state.lane_change_status is not LaneChangeStatus.CANDIDATE:
            return
        state.general_valid_motion_count += 1
        self._refresh_general_relative_motion(state)
        assert state.general_candidate_timestamp is not None
        if timestamp - state.general_candidate_timestamp > self._candidate_timeout:
            state.lane_change_status = LaneChangeStatus.REJECTED
            state.reason = "candidate timed out"
            return
        if (
            feature.membership is LaneMembership.NEAR_BOUNDARY
            and feature.lane_id == state.target_lane
        ):
            state.lane_change_phase = LaneChangePhase.CROSSING_BOUNDARY
            if state.crossed_frame is None:
                state.crossed_frame = frame_id
                state.crossed_timestamp = timestamp
            return
        if state.stable_lane_id == state.source_lane:
            if feature.membership is LaneMembership.INSIDE_LANE:
                state.lane_change_status = LaneChangeStatus.REJECTED
                state.reason = "vehicle returned to source lane"
            return
        if feature.membership is LaneMembership.NEAR_BOUNDARY:
            state.lane_change_phase = LaneChangePhase.CROSSING_BOUNDARY
            return
        if state.stable_lane_id != state.target_lane:
            return
        if state.general_entered_frame is None:
            if state.crossed_frame is None:
                state.crossed_frame = frame_id
                state.crossed_timestamp = timestamp
            state.general_entered_frame = frame_id
            state.general_entered_timestamp = timestamp
            state.general_entered_count = 1
        else:
            state.general_entered_count += 1
        state.lane_change_phase = LaneChangePhase.ENTERED_NEW_LANE
        assert state.general_entered_timestamp is not None
        enough_dwell = (
            state.general_entered_count >= self._minimum_confirmation_frames
            and timestamp - state.general_entered_timestamp + 1e-9
            >= self._minimum_confirmation_duration
        )
        motion_supported = (
            not self._require_relative_motion
            or (
                state.general_relative_motion is not None
                and state.general_relative_motion.supported
            )
        )
        if enough_dwell and motion_supported:
            state.lane_change_status = LaneChangeStatus.CONFIRMED
            state.completed_frame = frame_id
            state.completed_timestamp = timestamp
            state.reason = None

    def _refresh_general_relative_motion(self, state: _TrackState) -> None:
        if not self._require_relative_motion:
            state.general_relative_motion = None
            return
        if state.source_lane_order is None or state.target_lane_order is None:
            return
        evidences = [
            item.relative_motion
            for item in state.history
            if item.relative_motion is not None
            and (
                state.general_candidate_frame is None
                or item.frame_id >= state.general_candidate_frame
            )
        ]
        state.general_relative_motion = summarize_lane_relative_motion(
            evidences,
            state.source_lane_order,
            state.target_lane_order,
            minimum_valid_observations=self._minimum_relative_observations,
            minimum_cumulative_lateral_ratio=self._minimum_lateral_ratio,
            minimum_directional_consistency=self._minimum_directional_consistency,
            minimum_scene_consistency=self._minimum_scene_consistency,
            maximum_stationary_ratio=self._maximum_stationary_ratio,
        )

    @staticmethod
    def _rearm_general(state: _TrackState) -> None:
        state.lane_change_status = LaneChangeStatus.IDLE
        state.lane_change_phase = LaneChangePhase.STABLE_IN_LANE
        state.source_lane = None
        state.target_lane = None
        state.source_lane_order = None
        state.target_lane_order = None
        state.direction = LaneChangeDirection.UNKNOWN
        state.general_candidate_frame = None
        state.general_candidate_timestamp = None
        state.crossed_frame = None
        state.crossed_timestamp = None
        state.general_entered_frame = None
        state.general_entered_timestamp = None
        state.completed_frame = None
        state.completed_timestamp = None
        state.general_entered_count = 0
        state.general_valid_motion_count = 0
        state.general_relative_motion = None
        state.reason = None

    def _handle_missing(self, state: _TrackState, timestamp: float) -> None:
        state.missing_count += 1
        if (
            state.lane_change_status is LaneChangeStatus.CANDIDATE
            and state.general_candidate_timestamp is not None
            and timestamp - state.general_candidate_timestamp > self._candidate_timeout
        ):
            state.lane_change_status = LaneChangeStatus.REJECTED
            state.reason = "candidate timed out"
            return
        if state.missing_count <= self._maximum_missing:
            return
        state.stable_lane_id = None
        state.lane_change_phase = LaneChangePhase.UNKNOWN
        if state.lane_change_status is LaneChangeStatus.CANDIDATE:
            state.lane_change_status = LaneChangeStatus.UNKNOWN
            state.reason = "temporal evidence missing beyond tolerance"

    @staticmethod
    def _append_history(state: _TrackState, observation: TemporalLaneObservation) -> None:
        state.history.append(observation)

    @staticmethod
    def _snapshot(
        track_id: int, frame_id: int, timestamp: float, state: _TrackState
    ) -> TemporalLaneState:
        return TemporalLaneState(
            track_id=track_id,
            frame_id=frame_id,
            timestamp=timestamp,
            observed_lane_id=(state.history[-1].lane_id if state.history else None),
            stable_lane_id=state.stable_lane_id,
            lane_change_phase=state.lane_change_phase,
            lane_change_status=state.lane_change_status,
            source_lane=state.source_lane,
            target_lane=state.target_lane,
            direction=state.direction,
            candidate_started_frame=state.general_candidate_frame,
            candidate_started_timestamp=state.general_candidate_timestamp,
            boundary_crossed_frame=state.crossed_frame,
            boundary_crossed_timestamp=state.crossed_timestamp,
            entered_started_frame=state.general_entered_frame,
            entered_started_timestamp=state.general_entered_timestamp,
            completed_frame=state.completed_frame,
            completed_timestamp=state.completed_timestamp,
            missing_observations=state.missing_count,
            valid_motion_observations=state.general_valid_motion_count,
            boundary_id=state.boundary_id,
            relative_motion=state.general_relative_motion,
            reason=state.reason,
            history=tuple(state.history),
        )

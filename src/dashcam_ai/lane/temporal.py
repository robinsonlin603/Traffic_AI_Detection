"""以平滑、遲滯與 debounce 維護每個 track 的換道狀態。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionStatus
from dashcam_ai.domain.temporal import (
    LaneChangeStatus,
    LanePosition,
    LaneRelationPhase,
    ManeuverRelation,
    TemporalLaneObservation,
    TemporalLaneState,
)


@dataclass(slots=True)
class _TrackState:
    phase: LaneRelationPhase = LaneRelationPhase.UNKNOWN
    status: LaneChangeStatus = LaneChangeStatus.IDLE
    pending_phase: LaneRelationPhase = LaneRelationPhase.UNKNOWN
    pending_count: int = 0
    missing_count: int = 0
    valid_motion_count: int = 0
    saw_adjacent: bool = False
    saw_inside: bool = False
    candidate_frame: int | None = None
    candidate_timestamp: float | None = None
    entered_frame: int | None = None
    entered_timestamp: float | None = None
    entered_count: int = 0
    boundary_id: str | None = None
    maneuver_relation: ManeuverRelation = ManeuverRelation.UNKNOWN
    from_lane: LanePosition = LanePosition.UNKNOWN
    to_lane: LanePosition = LanePosition.UNKNOWN
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
        self._smoothing_window = smoothing_window_frames
        self._approaching_distance = approaching_distance_pixels
        self._entered_distance = entered_distance_pixels
        self._debounce_frames = debounce_frames
        self._minimum_confirmation_frames = minimum_confirmation_frames
        self._minimum_confirmation_duration = minimum_confirmation_duration_seconds
        self._maximum_missing = maximum_missing_frames
        self._candidate_timeout = candidate_timeout_seconds
        self._history_size = history_size
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        feature: LaneMembershipFeature,
        ego_motion_status: EgoMotionStatus,
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
                signed_boundary_distance=feature.signed_boundary_distance,
                nearest_boundary_id=feature.nearest_boundary_id,
                ego_motion_status=ego_motion_status,
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
        raw_phase = self._classify(feature.membership, smoothed)
        stable_phase = self._debounce(state, raw_phase)
        if stable_phase is not None:
            state.phase = stable_phase
        if feature.nearest_boundary_id is not None:
            state.boundary_id = feature.nearest_boundary_id
        observation = TemporalLaneObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            membership=feature.membership,
            signed_boundary_distance=distance,
            smoothed_signed_boundary_distance=smoothed,
            nearest_boundary_id=feature.nearest_boundary_id,
            ego_motion_status=ego_motion_status,
        )
        self._append_history(state, observation)
        self._advance(state, frame_id, timestamp, stable_phase)
        return self._snapshot(track_id, frame_id, timestamp, state)

    def forget(self, track_id: int) -> None:
        """追蹤 ID 永久消失後釋放其 bounded temporal state。"""
        self._tracks.pop(track_id, None)

    def _new_track(self) -> _TrackState:
        return _TrackState(
            distances=deque(maxlen=self._smoothing_window),
            history=deque(maxlen=self._history_size),
        )

    def _classify(self, membership: LaneMembership, distance: float) -> LaneRelationPhase:
        if membership is LaneMembership.BOUNDARY:
            return LaneRelationPhase.CROSSING
        if membership is LaneMembership.INSIDE:
            return (
                LaneRelationPhase.ENTERED
                if distance >= self._entered_distance
                else LaneRelationPhase.CROSSING
            )
        if distance > -self._approaching_distance:
            return LaneRelationPhase.APPROACHING
        return LaneRelationPhase.ADJACENT

    def _debounce(
        self, state: _TrackState, raw_phase: LaneRelationPhase
    ) -> LaneRelationPhase | None:
        if raw_phase is state.pending_phase:
            state.pending_count += 1
        else:
            state.pending_phase = raw_phase
            state.pending_count = 1
        return raw_phase if state.pending_count >= self._debounce_frames else None

    def _advance(
        self,
        state: _TrackState,
        frame_id: int,
        timestamp: float,
        stable_phase: LaneRelationPhase | None,
    ) -> None:
        if state.status is LaneChangeStatus.CONFIRMED:
            completed_relation = state.maneuver_relation
            self._rearm(state)
            if completed_relation is ManeuverRelation.ENTERING_EGO:
                state.saw_inside = True
            elif completed_relation is ManeuverRelation.LEAVING_EGO:
                state.saw_adjacent = True
        if stable_phase is LaneRelationPhase.ADJACENT:
            state.saw_adjacent = True
        if stable_phase is LaneRelationPhase.ENTERED:
            state.saw_inside = True
        if state.status is LaneChangeStatus.IDLE:
            if state.saw_adjacent and stable_phase in {
                LaneRelationPhase.APPROACHING,
                LaneRelationPhase.CROSSING,
            }:
                self._start_candidate(
                    state,
                    frame_id,
                    timestamp,
                    ManeuverRelation.ENTERING_EGO,
                )
            elif state.saw_inside and stable_phase in {
                LaneRelationPhase.APPROACHING,
                LaneRelationPhase.CROSSING,
            }:
                self._start_candidate(
                    state,
                    frame_id,
                    timestamp,
                    ManeuverRelation.LEAVING_EGO,
                )
            return
        if state.status is LaneChangeStatus.REJECTED:
            returned_to_origin = (
                state.maneuver_relation is ManeuverRelation.ENTERING_EGO
                and stable_phase is LaneRelationPhase.ADJACENT
            ) or (
                state.maneuver_relation is ManeuverRelation.LEAVING_EGO
                and stable_phase is LaneRelationPhase.ENTERED
            )
            if returned_to_origin:
                self._rearm(state)
            return
        if state.status is not LaneChangeStatus.CANDIDATE:
            return
        state.valid_motion_count += 1
        assert state.candidate_timestamp is not None
        if timestamp - state.candidate_timestamp > self._candidate_timeout:
            self._reject(state, "candidate timed out")
            return
        if (
            state.maneuver_relation is ManeuverRelation.ENTERING_EGO
            and stable_phase is LaneRelationPhase.ADJACENT
        ) or (
            state.maneuver_relation is ManeuverRelation.LEAVING_EGO
            and stable_phase is LaneRelationPhase.ENTERED
        ):
            self._reject(state, "vehicle returned to origin lane")
            return
        target_phase = (
            LaneRelationPhase.ENTERED
            if state.maneuver_relation is ManeuverRelation.ENTERING_EGO
            else LaneRelationPhase.ADJACENT
        )
        if stable_phase is target_phase:
            if state.entered_frame is None:
                state.entered_frame = frame_id
                state.entered_timestamp = timestamp
                state.entered_count = 1
            else:
                state.entered_count += 1
            assert state.entered_timestamp is not None
            entered_duration = timestamp - state.entered_timestamp
            if (
                state.entered_count >= self._minimum_confirmation_frames
                and entered_duration + 1e-9 >= self._minimum_confirmation_duration
                and state.valid_motion_count >= self._minimum_confirmation_frames
            ):
                state.status = LaneChangeStatus.CONFIRMED
                state.reason = None
        elif stable_phase is not None:
            state.entered_frame = None
            state.entered_timestamp = None
            state.entered_count = 0

    def _start_candidate(
        self,
        state: _TrackState,
        frame_id: int,
        timestamp: float,
        relation: ManeuverRelation,
    ) -> None:
        state.status = LaneChangeStatus.CANDIDATE
        state.candidate_frame = frame_id
        state.candidate_timestamp = timestamp
        state.valid_motion_count = 1
        state.maneuver_relation = relation
        adjacent = self._adjacent_lane(state.boundary_id)
        if relation is ManeuverRelation.ENTERING_EGO:
            state.from_lane = adjacent
            state.to_lane = LanePosition.EGO
        else:
            state.from_lane = LanePosition.EGO
            state.to_lane = adjacent
        state.reason = None

    @staticmethod
    def _adjacent_lane(boundary_id: str | None) -> LanePosition:
        if boundary_id == "left":
            return LanePosition.LEFT_ADJACENT
        if boundary_id == "right":
            return LanePosition.RIGHT_ADJACENT
        return LanePosition.UNKNOWN

    def _handle_missing(self, state: _TrackState, timestamp: float) -> None:
        state.missing_count += 1
        if (
            state.status is LaneChangeStatus.CANDIDATE
            and state.candidate_timestamp is not None
            and timestamp - state.candidate_timestamp > self._candidate_timeout
        ):
            self._reject(state, "candidate timed out")
            return
        if state.missing_count <= self._maximum_missing:
            return
        state.phase = LaneRelationPhase.UNKNOWN
        if state.status is LaneChangeStatus.CANDIDATE:
            self._reject(state, "temporal evidence missing beyond tolerance")

    @staticmethod
    def _reject(state: _TrackState, reason: str) -> None:
        state.status = LaneChangeStatus.REJECTED
        state.reason = reason

    @staticmethod
    def _rearm(state: _TrackState) -> None:
        state.status = LaneChangeStatus.IDLE
        state.candidate_frame = None
        state.candidate_timestamp = None
        state.entered_frame = None
        state.entered_timestamp = None
        state.entered_count = 0
        state.valid_motion_count = 0
        state.saw_adjacent = state.phase is LaneRelationPhase.ADJACENT
        state.saw_inside = state.phase is LaneRelationPhase.ENTERED
        state.maneuver_relation = ManeuverRelation.UNKNOWN
        state.from_lane = LanePosition.UNKNOWN
        state.to_lane = LanePosition.UNKNOWN
        state.reason = None

    @staticmethod
    def _append_history(state: _TrackState, observation: TemporalLaneObservation) -> None:
        state.history.append(observation)

    @staticmethod
    def _snapshot(
        track_id: int, frame_id: int, timestamp: float, state: _TrackState
    ) -> TemporalLaneState:
        return TemporalLaneState(
            track_id=track_id,
            phase=state.phase,
            status=state.status,
            frame_id=frame_id,
            timestamp=timestamp,
            candidate_started_frame=state.candidate_frame,
            candidate_started_timestamp=state.candidate_timestamp,
            entered_started_frame=state.entered_frame,
            entered_started_timestamp=state.entered_timestamp,
            missing_observations=state.missing_count,
            valid_motion_observations=state.valid_motion_count,
            boundary_id=state.boundary_id,
            maneuver_relation=state.maneuver_relation,
            from_lane=state.from_lane,
            to_lane=state.to_lane,
            reason=state.reason,
            history=tuple(state.history),
        )

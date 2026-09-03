"""串接 lane、ego-motion、temporal state 與事件的 streaming scene analyzer。"""

from __future__ import annotations

from typing import Any, Protocol

from dashcam_ai.domain.events import EventStatus, LaneChangeEvent
from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionEstimate, EgoMotionQuality, EgoMotionStatus
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.scene import FrameSceneAnalysis, TrackSceneAnalysis
from dashcam_ai.domain.temporal import TemporalLaneState
from dashcam_ai.events.lane_change import LaneChangeEventBuilder
from dashcam_ai.lane.base import LaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.motion.base import EgoMotionEstimator
from dashcam_ai.motion.relative import RelativeMotionEvaluator

StructuredEvent = LaneChangeEvent

EVENT_ELIGIBLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})


class SceneAnalysisBackend(Protocol):
    def process(
        self,
        frame: Any,
        objects: list[TrackedObject],
        frame_id: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> FrameSceneAnalysis: ...

    def finalize(self, frame_id: int, timestamp: float) -> None: ...

    def events(self) -> list[StructuredEvent]: ...


class StreamingSceneAnalyzer:
    """維護前一影格與有界 per-track 狀態，輸出最新事件快照。"""

    def __init__(
        self,
        *,
        lane_detector: LaneDetector,
        membership_evaluator: LaneMembershipEvaluator,
        motion_estimator: EgoMotionEstimator,
        relative_motion_evaluator: RelativeMotionEvaluator,
        temporal_tracker: TemporalLaneTracker,
        lane_change_builder: LaneChangeEventBuilder,
        maximum_missing_frames: int,
    ) -> None:
        if maximum_missing_frames < 0:
            raise ValueError("maximum_missing_frames must not be negative")
        self._lane_detector = lane_detector
        self._membership_evaluator = membership_evaluator
        self._motion_estimator = motion_estimator
        self._relative_motion_evaluator = relative_motion_evaluator
        self._temporal = temporal_tracker
        self._lane_change_builder = lane_change_builder
        self._retention_frames = maximum_missing_frames + 1
        self._previous_frame: Any | None = None
        self._previous_boxes: dict[int, BBox] = {}
        self._last_anchors: dict[int, Point2D] = {}
        self._missing_counts: dict[int, int] = {}
        self._event_eligible_tracks: set[int] = set()
        self._events: dict[str, StructuredEvent] = {}
        self._last_topology_id: str | None = None

    def process(
        self,
        frame: Any,
        objects: list[TrackedObject],
        frame_id: int,
        timestamp: float,
        width: int,
        height: int,
    ) -> FrameSceneAnalysis:
        geometry = self._lane_detector.detect(frame, width, height)
        if (
            geometry.topology_id is not None
            and self._last_topology_id is not None
            and geometry.topology_id != self._last_topology_id
        ):
            self._temporal.reset()
            self._reject_candidates_for_topology_change(frame_id, timestamp)
        if geometry.topology_id is not None:
            self._last_topology_id = geometry.topology_id
        motion = (
            self._unknown_motion("previous frame unavailable")
            if self._previous_frame is None
            else self._motion_estimator.estimate(
                self._previous_frame, frame, list(self._previous_boxes.values())
            )
        )
        relative_motion = {
            obj.track_id: self._relative_motion_evaluator.evaluate(
                self._last_anchors.get(obj.track_id),
                obj.bbox.bottom_center,
                motion,
                width,
                height,
            )
            for obj in objects
        }
        eligible_relative_motion = {
            obj.track_id: relative_motion[obj.track_id]
            for obj in objects
            if obj.class_name.casefold() in EVENT_ELIGIBLE_CLASSES
        }
        relative_motion.update(
            self._relative_motion_evaluator.apply_scene_consistency(eligible_relative_motion)
        )
        track_results: list[TrackSceneAnalysis] = []
        frame_lane_events: list[LaneChangeEvent] = []
        current_ids = {obj.track_id for obj in objects}
        for obj in objects:
            event_eligible = obj.class_name.casefold() in EVENT_ELIGIBLE_CLASSES
            if event_eligible:
                self._event_eligible_tracks.add(obj.track_id)
            else:
                self._event_eligible_tracks.discard(obj.track_id)
            membership = self._membership_evaluator.evaluate(obj.bbox.bottom_center, geometry)
            temporal = self._temporal.update(
                obj.track_id,
                frame_id,
                timestamp,
                membership,
                motion.status,
                relative_motion[obj.track_id],
            )
            track_results.append(
                TrackSceneAnalysis(track_id=obj.track_id, membership=membership, temporal=temporal)
            )
            self._record_lane_event(temporal, frame_lane_events, event_eligible=event_eligible)
            self._last_anchors[obj.track_id] = obj.bbox.bottom_center
            self._missing_counts[obj.track_id] = 0

        for track_id in tuple(self._last_anchors):
            if track_id in current_ids:
                continue
            missing = self._missing_counts.get(track_id, 0) + 1
            self._missing_counts[track_id] = missing
            unknown = LaneMembershipFeature(
                membership=LaneMembership.UNKNOWN,
                anchor=self._last_anchors[track_id],
                geometry_confidence=geometry.confidence,
            )
            temporal = self._temporal.update(track_id, frame_id, timestamp, unknown, motion.status)
            self._record_lane_event(
                temporal,
                frame_lane_events,
                event_eligible=track_id in self._event_eligible_tracks,
            )
            if missing > self._retention_frames:
                self._temporal.forget(track_id)
                self._last_anchors.pop(track_id, None)
                self._missing_counts.pop(track_id, None)
                self._event_eligible_tracks.discard(track_id)

        self._previous_frame = frame.copy() if hasattr(frame, "copy") else frame
        self._previous_boxes = {obj.track_id: obj.bbox for obj in objects}
        return FrameSceneAnalysis(
            lane_geometry=geometry,
            ego_motion=motion,
            tracks=tuple(track_results),
            lane_change_events=tuple(frame_lane_events),
        )

    def events(self) -> list[StructuredEvent]:
        return [self._events[key] for key in sorted(self._events)]

    def finalize(self, frame_id: int, timestamp: float) -> None:
        """影片結束時拒絕仍未完成的候選事件。"""
        for event_id, event in tuple(self._events.items()):
            if event.status is not EventStatus.CANDIDATE:
                continue
            rejected = event.model_copy(
                update={
                    "status": EventStatus.REJECTED,
                    "completed_frame": None,
                    "completed_at": None,
                    "confidence": min(event.confidence, 0.4),
                    "reason": "video ended before lane change confirmation",
                }
            )
            self._events[event_id] = rejected

    def _record_lane_event(
        self,
        temporal: TemporalLaneState,
        output: list[LaneChangeEvent],
        *,
        event_eligible: bool,
    ) -> LaneChangeEvent | None:
        if not event_eligible:
            return None
        event = self._lane_change_builder.build(temporal)
        if event is None:
            return None
        existing = self._events.get(event.event_id)
        if existing is not None and existing.status in {
            EventStatus.CONFIRMED,
            EventStatus.REJECTED,
        }:
            return None
        self._events[event.event_id] = event
        output.append(event)
        return event

    def _reject_candidates_for_topology_change(self, frame_id: int, timestamp: float) -> None:
        for event_id, event in tuple(self._events.items()):
            if event.status is not EventStatus.CANDIDATE:
                continue
            self._events[event_id] = event.model_copy(
                update={
                    "status": EventStatus.REJECTED,
                    "completed_frame": None,
                    "completed_at": None,
                    "confidence": min(event.confidence, 0.4),
                    "reason": (f"lane topology changed at frame {frame_id} ({timestamp:.3f}s)"),
                }
            )

    @staticmethod
    def _unknown_motion(reason: str) -> EgoMotionEstimate:
        return EgoMotionEstimate(
            status=EgoMotionStatus.UNKNOWN,
            quality=EgoMotionQuality(
                detected_features=0,
                tracked_features=0,
                inlier_count=0,
                inlier_ratio=0,
                confidence=0,
            ),
            reason=reason,
        )

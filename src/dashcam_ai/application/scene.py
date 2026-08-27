"""串接 lane、ego-motion、temporal state 與事件的 streaming scene analyzer。"""

from __future__ import annotations

from typing import Any, Protocol

from dashcam_ai.domain.events import CutInEvent, EventStatus, LaneChangeEvent
from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import LaneMembership, LaneMembershipFeature
from dashcam_ai.domain.motion import EgoMotionEstimate, EgoMotionQuality, EgoMotionStatus
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.scene import FrameSceneAnalysis, TrackSceneAnalysis
from dashcam_ai.domain.temporal import TemporalLaneState
from dashcam_ai.events.corridor import ConfiguredForwardCorridor
from dashcam_ai.events.cutin import CutInDetector
from dashcam_ai.events.lane_change import LaneChangeEventBuilder
from dashcam_ai.lane.base import LaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.motion.base import EgoMotionEstimator

StructuredEvent = LaneChangeEvent | CutInEvent


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
        temporal_tracker: TemporalLaneTracker,
        corridor: ConfiguredForwardCorridor,
        lane_change_builder: LaneChangeEventBuilder,
        cut_in_detector: CutInDetector,
        maximum_missing_frames: int,
    ) -> None:
        if maximum_missing_frames < 0:
            raise ValueError("maximum_missing_frames must not be negative")
        self._lane_detector = lane_detector
        self._membership_evaluator = membership_evaluator
        self._motion_estimator = motion_estimator
        self._temporal = temporal_tracker
        self._corridor_factory = corridor
        self._lane_change_builder = lane_change_builder
        self._cut_in_detector = cut_in_detector
        self._retention_frames = maximum_missing_frames + 1
        self._previous_frame: Any | None = None
        self._previous_boxes: dict[int, BBox] = {}
        self._last_anchors: dict[int, Point2D] = {}
        self._missing_counts: dict[int, int] = {}
        self._events: dict[str, StructuredEvent] = {}

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
        corridor = self._corridor_factory.resolve(width, height)
        motion = (
            self._unknown_motion("previous frame unavailable")
            if self._previous_frame is None
            else self._motion_estimator.estimate(
                self._previous_frame, frame, list(self._previous_boxes.values())
            )
        )
        track_results: list[TrackSceneAnalysis] = []
        frame_lane_events: list[LaneChangeEvent] = []
        frame_cutin_events: list[CutInEvent] = []
        current_ids = {obj.track_id for obj in objects}
        for obj in objects:
            membership = self._membership_evaluator.evaluate(obj.bbox.bottom_center, geometry)
            temporal = self._temporal.update(
                obj.track_id, frame_id, timestamp, membership, motion.status
            )
            track_results.append(
                TrackSceneAnalysis(
                    track_id=obj.track_id, membership=membership, temporal=temporal
                )
            )
            lane_event = self._record_lane_event(temporal, frame_lane_events)
            previous_bbox = self._previous_boxes.get(obj.track_id)
            if lane_event is not None:
                cutin = self._cut_in_detector.detect(
                    lane_event, obj.bbox, previous_bbox, corridor
                )
                self._events[cutin.event_id] = cutin
                frame_cutin_events.append(cutin)
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
            temporal = self._temporal.update(
                track_id, frame_id, timestamp, unknown, motion.status
            )
            lane_event = self._record_lane_event(temporal, frame_lane_events)
            if lane_event is not None and lane_event.status is EventStatus.REJECTED:
                self._cascade_cutin_rejection(lane_event)
            if missing > self._retention_frames:
                self._temporal.forget(track_id)
                self._last_anchors.pop(track_id, None)
                self._missing_counts.pop(track_id, None)

        self._previous_frame = frame.copy() if hasattr(frame, "copy") else frame
        self._previous_boxes = {obj.track_id: obj.bbox for obj in objects}
        return FrameSceneAnalysis(
            lane_geometry=geometry,
            ego_motion=motion,
            forward_corridor=corridor,
            tracks=tuple(track_results),
            lane_change_events=tuple(frame_lane_events),
            cut_in_events=tuple(frame_cutin_events),
        )

    def events(self) -> list[StructuredEvent]:
        return [self._events[key] for key in sorted(self._events)]

    def finalize(self, frame_id: int, timestamp: float) -> None:
        """影片結束時拒絕仍未完成的候選事件。"""
        for event_id, event in tuple(self._events.items()):
            if not isinstance(event, LaneChangeEvent):
                continue
            if event.status is not EventStatus.CANDIDATE:
                continue
            rejected = event.model_copy(
                update={
                    "status": EventStatus.REJECTED,
                    "end_frame": frame_id,
                    "end_timestamp": timestamp,
                    "confidence": min(event.confidence, 0.4),
                    "reason": "video ended before lane change confirmation",
                }
            )
            self._events[event_id] = rejected
            self._cascade_cutin_rejection(rejected)

        for event_id, event in tuple(self._events.items()):
            if not isinstance(event, CutInEvent):
                continue
            if event.status is not EventStatus.CANDIDATE:
                continue
            self._events[event_id] = event.model_copy(
                update={
                    "status": EventStatus.REJECTED,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "reason": "video ended before cut-in confirmation",
                }
            )

    def _record_lane_event(
        self, temporal: TemporalLaneState, output: list[LaneChangeEvent]
    ) -> LaneChangeEvent | None:
        event = self._lane_change_builder.build(temporal)
        if event is None:
            return None
        existing = self._events.get(event.event_id)
        if isinstance(existing, LaneChangeEvent) and existing.status in {
            EventStatus.CONFIRMED,
            EventStatus.REJECTED,
        }:
            return None
        self._events[event.event_id] = event
        output.append(event)
        return event

    def _cascade_cutin_rejection(self, lane_event: LaneChangeEvent) -> None:
        event_id = f"cut-in:{lane_event.track_id}:{lane_event.start_frame}"
        existing = self._events.get(event_id)
        if isinstance(existing, CutInEvent):
            if existing.status in {EventStatus.CONFIRMED, EventStatus.REJECTED}:
                return
            self._events[event_id] = existing.model_copy(
                update={
                    "status": EventStatus.REJECTED,
                    "frame_id": lane_event.end_frame,
                    "timestamp": lane_event.end_timestamp,
                    "reason": "lane change was rejected",
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

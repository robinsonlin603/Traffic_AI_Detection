from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.application.scene import StreamingSceneAnalyzer
from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.lane import (
    LaneGeometryProvenance,
    LaneGeometryStatus,
    NormalizedLaneBoundary,
    NormalizedLaneRegion,
    NormalizedPoint2D,
)
from dashcam_ai.domain.motion import (
    EgoMotionEstimate,
    EgoMotionQuality,
    EgoMotionStatus,
    HomographyTransform,
)
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.video import VideoMetadata
from dashcam_ai.events.lane_change import LaneChangeEventBuilder
from dashcam_ai.lane.base import LaneDetector
from dashcam_ai.lane.configured import ConfiguredLaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.motion.relative import RelativeMotionEvaluator
from dashcam_ai.video.reader import VideoFrame


class SequenceBackend:
    def __init__(self, observations: list[list[TrackedObject]]) -> None:
        self._observations = iter(observations)

    def process(self, frame: Any) -> list[TrackedObject]:
        return next(self._observations)


class ArrayReader:
    def __init__(self, source: Path, frame_count: int) -> None:
        self.metadata = VideoMetadata(
            source=str(source),
            width=1000,
            height=500,
            fps=10,
            frame_count=frame_count,
            duration_seconds=frame_count / 10,
        )

    def __iter__(self):
        for frame_id in range(self.metadata.frame_count):
            yield VideoFrame(
                frame_id=frame_id,
                timestamp=frame_id / 10,
                image=np.zeros((500, 1000, 3), dtype=np.uint8),
            )

    def __enter__(self) -> ArrayReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class ValidMotionEstimator:
    def estimate(
        self, previous_frame: Any, current_frame: Any, excluded_boxes: list[BBox]
    ) -> EgoMotionEstimate:
        return EgoMotionEstimate(
            status=EgoMotionStatus.VALID,
            transform=HomographyTransform(values=(1, 0, 0, 0, 1, 0, 0, 0, 1)),
            quality=EgoMotionQuality(
                detected_features=50,
                tracked_features=48,
                inlier_count=46,
                inlier_ratio=46 / 48,
                mean_reprojection_error=0.2,
                confidence=0.9,
            ),
        )


def tracked(center_x: float) -> TrackedObject:
    return TrackedObject(
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox=BBox(x1=center_x - 20, y1=380, x2=center_x + 20, y2=420),
    )


def scene_analyzer(lane_detector: LaneDetector | None = None) -> StreamingSceneAnalyzer:
    points = {
        "left": (
            NormalizedPoint2D(x=0, y=0),
            NormalizedPoint2D(x=0.4, y=0),
            NormalizedPoint2D(x=0.4, y=1),
            NormalizedPoint2D(x=0, y=1),
        ),
        "center": (
            NormalizedPoint2D(x=0.4, y=0),
            NormalizedPoint2D(x=0.6, y=0),
            NormalizedPoint2D(x=0.6, y=1),
            NormalizedPoint2D(x=0.4, y=1),
        ),
        "right": (
            NormalizedPoint2D(x=0.6, y=0),
            NormalizedPoint2D(x=1, y=0),
            NormalizedPoint2D(x=1, y=1),
            NormalizedPoint2D(x=0.6, y=1),
        ),
    }
    configured = ConfiguredLaneDetector(
        lanes=[
            NormalizedLaneRegion(lane_id="lane_left", lateral_order=0, polygon=points["left"]),
            NormalizedLaneRegion(lane_id="lane_center", lateral_order=1, polygon=points["center"]),
            NormalizedLaneRegion(lane_id="lane_right", lateral_order=2, polygon=points["right"]),
        ],
        boundaries=[
            NormalizedLaneBoundary(
                boundary_id="boundary_left_center",
                left_lane_id="lane_left",
                right_lane_id="lane_center",
                points=(points["center"][0], points["center"][3]),
            ),
            NormalizedLaneBoundary(
                boundary_id="boundary_center_right",
                left_lane_id="lane_center",
                right_lane_id="lane_right",
                points=(points["center"][1], points["center"][2]),
            ),
        ],
    )
    return StreamingSceneAnalyzer(
        lane_detector=lane_detector or configured,
        membership_evaluator=LaneMembershipEvaluator(boundary_margin=5),
        motion_estimator=ValidMotionEstimator(),
        relative_motion_evaluator=RelativeMotionEvaluator(),
        temporal_tracker=TemporalLaneTracker(
            smoothing_window_frames=1,
            debounce_frames=1,
            minimum_confirmation_frames=2,
            minimum_confirmation_duration_seconds=0.1,
            maximum_missing_frames=1,
            history_size=10,
        ),
        lane_change_builder=LaneChangeEventBuilder(evidence_history_size=10),
        maximum_missing_frames=1,
    )


def test_pipeline_writes_only_general_lane_change_events(tmp_path: Path) -> None:
    observations = [[tracked(x)] for x in (500, 500, 595, 602, 650, 660)]
    analyzer = Analyzer(
        SequenceBackend(observations),
        save_video=False,
        save_frames=True,
        reader_factory=lambda source: ArrayReader(source, len(observations)),
        scene_analyzer=scene_analyzer(),
    )
    output = tmp_path / "scene"

    summary = analyzer.analyze(Path("synthetic.mp4"), output)

    events: list[dict[str, Any]] = json.loads((output / "events.json").read_text())
    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert summary.events_created == 1
    assert len(events) == 1
    assert events[0]["event_type"] == "lane_change"
    assert events[0]["source_lane"] == "lane_center"
    assert events[0]["target_lane"] == "lane_right"
    assert events[0]["direction"] == "right"
    assert events[0]["started_at"] <= events[0]["lane_crossed_at"] <= events[0]["completed_at"]
    assert "cut_in_events" not in frames[-1]["analysis"]
    assert "forward_corridor" not in frames[-1]["analysis"]


def test_degraded_geometry_cannot_create_pipeline_events(tmp_path: Path) -> None:
    class DegradedDetector:
        def __init__(self, delegate: LaneDetector) -> None:
            self.delegate = delegate

        def detect(self, frame: Any, width: int, height: int):
            return self.delegate.detect(frame, width, height).model_copy(
                update={
                    "status": LaneGeometryStatus.DEGRADED,
                    "provenance": LaneGeometryProvenance.DYNAMIC,
                    "reason": "untrusted road edge",
                }
            )

    baseline = scene_analyzer()
    detector = DegradedDetector(baseline._lane_detector)
    observations = [[tracked(x)] for x in (500, 500, 595, 602, 650, 660)]
    analyzer = Analyzer(
        SequenceBackend(observations),
        save_video=False,
        save_frames=True,
        reader_factory=lambda source: ArrayReader(source, len(observations)),
        scene_analyzer=scene_analyzer(detector),
    )

    summary = analyzer.analyze(Path("synthetic.mp4"), tmp_path / "degraded")

    assert summary.events_created == 0
    assert json.loads((tmp_path / "degraded" / "events.json").read_text()) == []

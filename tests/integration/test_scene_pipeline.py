from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import pytest

from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.application.scene import StreamingSceneAnalyzer
from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.lane import NormalizedPoint2D
from dashcam_ai.domain.motion import (
    EgoMotionEstimate,
    EgoMotionQuality,
    EgoMotionStatus,
    HomographyTransform,
)
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.video import VideoMetadata
from dashcam_ai.events.corridor import ConfiguredForwardCorridor
from dashcam_ai.events.cutin import CutInDetector
from dashcam_ai.events.lane_change import LaneChangeEventBuilder
from dashcam_ai.lane.configured import ConfiguredLaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.video.reader import VideoFrame
from dashcam_ai.visualization.annotator import OpenCVAnnotator


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


def tracked(center_x: float, size: float) -> TrackedObject:
    return TrackedObject(
        track_id=7,
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox=BBox(
            x1=center_x - size / 2,
            y1=420 - size,
            x2=center_x + size / 2,
            y2=420,
        ),
    )


def scene_analyzer(maximum_missing_frames: int = 1) -> StreamingSceneAnalyzer:
    lane_polygon = [
        NormalizedPoint2D(x=0.4, y=0),
        NormalizedPoint2D(x=0.6, y=0),
        NormalizedPoint2D(x=0.6, y=1),
        NormalizedPoint2D(x=0.4, y=1),
    ]
    corridor_polygon = [
        NormalizedPoint2D(x=0.35, y=0.5),
        NormalizedPoint2D(x=0.65, y=0.5),
        NormalizedPoint2D(x=0.7, y=1),
        NormalizedPoint2D(x=0.3, y=1),
    ]
    return StreamingSceneAnalyzer(
        lane_detector=ConfiguredLaneDetector(lane_polygon),
        membership_evaluator=LaneMembershipEvaluator(boundary_margin=5),
        motion_estimator=ValidMotionEstimator(),
        temporal_tracker=TemporalLaneTracker(
            smoothing_window_frames=1,
            approaching_distance_pixels=50,
            entered_distance_pixels=20,
            debounce_frames=1,
            minimum_confirmation_frames=2,
            minimum_confirmation_duration_seconds=0.1,
            maximum_missing_frames=maximum_missing_frames,
            candidate_timeout_seconds=2,
            history_size=10,
        ),
        corridor=ConfiguredForwardCorridor(corridor_polygon),
        lane_change_builder=LaneChangeEventBuilder(evidence_history_size=10),
        cut_in_detector=CutInDetector(minimum_bbox_expansion_ratio=0.05),
        maximum_missing_frames=maximum_missing_frames,
    )


def test_scene_pipeline_writes_analysis_and_deduplicated_events(tmp_path: Path) -> None:
    observations = [
        [tracked(300, 40)],
        [tracked(300, 42)],
        [tracked(380, 46)],
        [tracked(410, 50)],
        [tracked(440, 56)],
        [tracked(450, 64)],
    ]
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
    assert summary.events_created == 2
    assert {event["event_type"] for event in events} == {"lane_change", "cut_in"}
    assert all(event["status"] == "confirmed" for event in events)
    assert len({event["event_id"] for event in events}) == 2
    assert frames[0]["analysis"]["ego_motion"]["status"] == "unknown"
    assert frames[-1]["analysis"]["lane_change_events"][0]["status"] == "confirmed"


def test_missing_track_rejects_candidate_without_false_confirmation(tmp_path: Path) -> None:
    observations = [
        [tracked(300, 40)],
        [tracked(300, 42)],
        [tracked(380, 46)],
        [],
        [],
    ]
    analyzer = Analyzer(
        SequenceBackend(observations),
        save_video=False,
        save_frames=True,
        reader_factory=lambda source: ArrayReader(source, len(observations)),
        scene_analyzer=scene_analyzer(maximum_missing_frames=1),
    )
    output = tmp_path / "missing"

    analyzer.analyze(Path("synthetic.mp4"), output)

    events: list[dict[str, Any]] = json.loads((output / "events.json").read_text())
    lane_event = next(event for event in events if event["event_type"] == "lane_change")
    assert lane_event["status"] == "rejected"
    assert all(event["status"] == "rejected" for event in events)


def test_video_end_rejects_unfinished_candidate(tmp_path: Path) -> None:
    observations = [
        [tracked(300, 40)],
        [tracked(300, 42)],
        [tracked(380, 46)],
        [tracked(390, 48)],
    ]
    analyzer = Analyzer(
        SequenceBackend(observations),
        save_video=False,
        save_frames=True,
        reader_factory=lambda source: ArrayReader(source, len(observations)),
        scene_analyzer=scene_analyzer(),
    )
    output = tmp_path / "unfinished"

    analyzer.analyze(Path("synthetic.mp4"), output)

    events: list[dict[str, Any]] = json.loads((output / "events.json").read_text())
    lane_event = next(event for event in events if event["event_type"] == "lane_change")
    assert lane_event["status"] == "rejected"
    assert lane_event["end_frame"] == 3
    assert lane_event["reason"] == "video ended before lane change confirmation"
    assert all(event["status"] == "rejected" for event in events)


def test_terminal_event_is_not_overwritten_by_later_frames() -> None:
    scene = scene_analyzer()
    sequence = [(300, 40), (300, 42), (380, 46), (410, 50), (440, 56), (450, 64)]
    for frame_id, (center_x, size) in enumerate(sequence):
        scene.process(
            np.zeros((500, 1000, 3), dtype=np.uint8),
            [tracked(center_x, size)],
            frame_id,
            frame_id / 10,
            1000,
            500,
        )
    confirmed = next(
        event
        for event in scene.events()
        if event.event_type == "lane_change" and event.status.value == "confirmed"
    )

    scene.process(
        np.zeros((500, 1000, 3), dtype=np.uint8),
        [tracked(300, 70)],
        len(sequence),
        len(sequence) / 10,
        1000,
        500,
    )
    frozen = next(event for event in scene.events() if event.event_id == confirmed.event_id)

    assert frozen == confirmed


@pytest.mark.cv
def test_scene_annotation_draws_geometry_and_confirmed_event() -> None:
    scene = scene_analyzer()
    analysis = None
    current = tracked(300, 40)
    sequence = [(300, 40), (300, 42), (380, 46), (410, 50), (440, 56), (450, 64)]
    for frame_id, (center_x, size) in enumerate(sequence):
        current = tracked(center_x, size)
        analysis = scene.process(
            np.zeros((500, 1000, 3), dtype=np.uint8),
            [current],
            frame_id,
            frame_id / 10,
            1000,
            500,
        )
    assert analysis is not None

    annotated = OpenCVAnnotator().annotate(
        np.zeros((500, 1000, 3), dtype=np.uint8), [current], analysis
    )

    assert np.count_nonzero(annotated) > 0
    assert analysis.lane_change_events[-1].status.value == "confirmed"
    assert analysis.cut_in_events[-1].status.value == "confirmed"


def test_track_labels_avoid_existing_label_boxes() -> None:
    occupied = [(100, 80, 220, 110)]

    label = OpenCVAnnotator._place_label(
        frame_width=500,
        frame_height=300,
        label_width=120,
        label_height=30,
        bbox=(100, 110, 180, 180),
        occupied=occupied,
    )

    assert label == (100, 110, 220, 140)
    assert not OpenCVAnnotator._boxes_overlap(label, occupied[0])

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any

from dashcam_ai.application.adapters import DetectorTrackerBackend
from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.detection.fake import FakeDetector
from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import Detection
from dashcam_ai.domain.video import VideoMetadata
from dashcam_ai.tracking.centroid import CentroidTracker
from dashcam_ai.video.reader import VideoFrame


class FakeReader:
    def __init__(self, source: Path) -> None:
        self.metadata = VideoMetadata(
            source=str(source), width=1920, height=1080, fps=10, frame_count=2, duration_seconds=0.2
        )

    def __iter__(self):
        yield VideoFrame(frame_id=0, timestamp=0.0, image=object())
        yield VideoFrame(frame_id=1, timestamp=0.1, image=object())

    def __enter__(self) -> FakeReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def car(x: float) -> Detection:
    return Detection(
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox=BBox(x1=x, y1=100, x2=x + 100, y2=250),
    )


def test_fake_pipeline_writes_normalized_artifacts(tmp_path: Path) -> None:
    backend = DetectorTrackerBackend(
        FakeDetector([[car(100)], [car(110)]]), CentroidTracker(maximum_distance=50)
    )
    analyzer = Analyzer(
        backend,
        save_video=False,
        save_frames=True,
        reader_factory=FakeReader,
    )
    output = tmp_path / "trip"

    summary = analyzer.analyze(Path("synthetic.mp4"), output)

    assert summary.frames_processed == 2
    assert summary.tracks_created == 1
    tracks: list[dict[str, Any]] = json.loads((output / "tracks.json").read_text())
    assert len(tracks) == 1
    assert len(tracks[0]["observations"]) == 2
    assert tracks[0]["observations"][0]["bottom_center"] == {"x": 150.0, "y": 250.0}
    assert json.loads((output / "events.json").read_text()) == []
    assert len((output / "frames.jsonl").read_text().splitlines()) == 2


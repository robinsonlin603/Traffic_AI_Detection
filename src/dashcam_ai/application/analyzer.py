"""協調影片讀取、感知分析、標註及結果輸出的核心流程。"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashcam_ai.application.perception import PerceptionBackend
from dashcam_ai.domain.perception import Track, TrackObservation
from dashcam_ai.domain.video import FrameRecord
from dashcam_ai.logging import get_logger
from dashcam_ai.storage.artifacts import ArtifactStore
from dashcam_ai.video.reader import OpenCVVideoReader
from dashcam_ai.video.writer import OpenCVVideoWriter
from dashcam_ai.visualization.annotator import OpenCVAnnotator


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """一次影片分析完成後的效能與輸出摘要。"""
    frames_processed: int
    tracks_created: int
    elapsed_seconds: float
    processing_fps: float
    output_directory: Path


class Analyzer:
    """逐幀執行感知分析，並產生結構化資料與標註影片。"""

    def __init__(
        self,
        perception: PerceptionBackend,
        save_video: bool = True,
        save_frames: bool = True,
        codec: str = "mp4v",
        reader_factory: Callable[[Path], Any] = OpenCVVideoReader,
    ) -> None:
        self.perception = perception
        self.save_video = save_video
        self.save_frames = save_frames
        self.codec = codec
        self.reader_factory = reader_factory

    def analyze(self, source: Path, output_directory: Path) -> AnalysisSummary:
        """分析來源影片，將所有成果寫入指定輸出目錄。"""
        logger = get_logger()
        started = time.monotonic()
        # 依追蹤 ID 累積跨幀觀察，最後組合成完整 Track。
        observations: dict[int, list[TrackObservation]] = defaultdict(list)
        track_classes: dict[int, str] = {}
        frames_processed = 0
        writer: OpenCVVideoWriter | None = None
        annotator = OpenCVAnnotator()
        with self.reader_factory(source) as reader, ArtifactStore(
            output_directory, save_frames=self.save_frames
        ) as store:
            runtime_metadata = getattr(self.perception, "runtime_metadata", None)
            metadata = reader.metadata.model_copy(update={"runtime": runtime_metadata})
            store.write_metadata(metadata)
            store.write_events_placeholder()
            logger.info("video_loaded", **metadata.model_dump())
            if runtime_metadata is not None:
                logger.info("runtime_resolved", **runtime_metadata.model_dump())
            if self.save_video:
                writer = OpenCVVideoWriter(
                    output_directory / "annotated.mp4", reader.metadata, self.codec
                )
            try:
                for video_frame in reader:
                    tracked = self.perception.process(video_frame.image)
                    store.write_frame(
                        FrameRecord(
                            frame_id=video_frame.frame_id,
                            timestamp=video_frame.timestamp,
                            objects=tracked,
                        )
                    )
                    for obj in tracked:
                        track_classes[obj.track_id] = obj.class_name
                        observations[obj.track_id].append(
                            TrackObservation.from_tracked_object(
                                obj, video_frame.frame_id, video_frame.timestamp
                            )
                        )
                    if writer is not None:
                        writer.write(annotator.annotate(video_frame.image, tracked))
                    frames_processed += 1
            finally:
                if writer is not None:
                    writer.close()
            # 每個 ID 的觀察已按影片處理順序排列，可直接取得首次與末次時間。
            tracks = [
                Track(
                    track_id=track_id,
                    class_name=track_classes[track_id],
                    first_seen=items[0].timestamp,
                    last_seen=items[-1].timestamp,
                    observations=items,
                )
                for track_id, items in sorted(observations.items())
            ]
            store.write_tracks(tracks)
        elapsed = time.monotonic() - started
        summary = AnalysisSummary(
            frames_processed=frames_processed,
            tracks_created=len(observations),
            elapsed_seconds=elapsed,
            processing_fps=frames_processed / elapsed if elapsed else 0.0,
            output_directory=output_directory,
        )
        logger.info(
            "video_completed",
            frames_processed=summary.frames_processed,
            tracks_created=summary.tracks_created,
            elapsed_seconds=round(summary.elapsed_seconds, 3),
            processing_fps=round(summary.processing_fps, 3),
        )
        return summary

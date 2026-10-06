"""將影片中繼資料、逐幀結果及完整軌跡寫入磁碟。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel

from dashcam_ai.domain.lane_lines import LaneLineRecord
from dashcam_ai.domain.perception import Track
from dashcam_ai.domain.video import FrameRecord, VideoMetadata


class ArtifactStore:
    """集中管理一次分析工作產生的所有結構化成果檔案。"""
    def __init__(self, root: Path, save_frames: bool = True) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._frames_file: TextIO | None = None
        self._lane_lines_file: TextIO | None = None
        if save_frames:
            self._frames_file = (root / "frames.jsonl").open("w", encoding="utf-8")

    def write_metadata(self, metadata: VideoMetadata) -> None:
        """寫入來源影片的中繼資料。"""
        self._write_json("metadata.json", metadata)

    def write_frame(self, frame: FrameRecord) -> None:
        """以 JSONL 格式附加單一影格的分析結果。"""
        if self._frames_file is not None:
            self._frames_file.write(frame.model_dump_json() + "\n")

    def write_tracks(self, tracks: list[Track]) -> None:
        """寫入所有物件的完整跨幀軌跡。"""
        self._write_json_value("tracks.json", [track.model_dump(mode="json") for track in tracks])

    def write_lane_lines(self, record: LaneLineRecord) -> None:
        """延後建立白線 JSONL；停用 Milestone 2 時不產生空檔案。"""
        if self._lane_lines_file is None:
            self._lane_lines_file = (self.root / "lane-lines.jsonl").open(
                "w", encoding="utf-8"
            )
        self._lane_lines_file.write(record.model_dump_json() + "\n")

    def write_events(self, events: list[object]) -> None:
        """Milestone 1 僅輸出空事件占位資料。"""
        if events:
            raise ValueError("Milestone 1 does not produce events")
        self._write_json_value("events.json", [])

    def close(self) -> None:
        """關閉仍開啟的逐幀輸出檔案。"""
        if self._frames_file is not None:
            self._frames_file.close()
            self._frames_file = None
        if self._lane_lines_file is not None:
            self._lane_lines_file.close()
            self._lane_lines_file = None

    def _write_json(self, filename: str, model: BaseModel) -> None:
        self._write_json_value(filename, model.model_dump(mode="json"))

    def _write_json_value(self, filename: str, value: object) -> None:
        with (self.root / filename).open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False)
            file.write("\n")

    def __enter__(self) -> ArtifactStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

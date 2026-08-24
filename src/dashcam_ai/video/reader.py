"""透過 OpenCV 讀取影片中繼資料並逐幀產生影像。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashcam_ai.domain.video import VideoMetadata


def _cv2() -> Any:
    """延遲載入選配的 OpenCV，缺少依賴時提供明確錯誤。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for video processing. Install the 'cv' dependency group."
        ) from error
    return cv2


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """含幀編號、時間戳與影像矩陣的單一影片影格。"""
    frame_id: int
    timestamp: float
    image: Any


class OpenCVVideoReader:
    """管理 OpenCV 影片資源並以 iterator 逐幀讀取。"""
    def __init__(self, source: Path) -> None:
        if not source.is_file():
            raise FileNotFoundError(f"video does not exist: {source}")
        self.source = source
        cv2 = _cv2()
        self._capture = cv2.VideoCapture(str(source))
        if not self._capture.isOpened():
            raise ValueError(f"could not open video: {source}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            self.close()
            raise ValueError(f"invalid video metadata: {source}")
        self.metadata = VideoMetadata(
            source=str(source.resolve()),
            width=width,
            height=height,
            fps=fps,
            frame_count=max(frame_count, 0),
            duration_seconds=max(frame_count, 0) / fps,
        )

    def __iter__(self) -> Iterator[VideoFrame]:
        """依影片順序產生各影格及其對應時間戳。"""
        cv2 = _cv2()
        frame_id = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            position_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            # 部分影片不提供可靠時間戳，此時以幀編號和 FPS 推算。
            timestamp = position_ms / 1000 if position_ms > 0 else frame_id / self.metadata.fps
            yield VideoFrame(frame_id=frame_id, timestamp=timestamp, image=image)
            frame_id += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> OpenCVVideoReader:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

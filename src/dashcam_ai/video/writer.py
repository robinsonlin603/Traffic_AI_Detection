"""透過 OpenCV 建立並寫入標註後的影片。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dashcam_ai.domain.video import VideoMetadata
from dashcam_ai.video.reader import _cv2


class OpenCVVideoWriter:
    """使用來源影片的解析度與 FPS 寫出新影片。"""
    def __init__(self, path: Path, metadata: VideoMetadata, codec: str = "mp4v") -> None:
        cv2 = _cv2()
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(path), fourcc, metadata.fps, (metadata.width, metadata.height)
        )
        if not self._writer.isOpened():
            raise ValueError(f"could not open output video: {path}")

    def write(self, frame: Any) -> None:
        """將單一標註影格寫入輸出影片。"""
        self._writer.write(frame)

    def close(self) -> None:
        """釋放 OpenCV 影片寫入資源。"""
        self._writer.release()

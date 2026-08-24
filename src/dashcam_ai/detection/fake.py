"""提供不需要模型或 GPU 的測試用偵測器。"""

from typing import Any

from dashcam_ai.domain.perception import Detection


class FakeDetector:
    """依序回傳預先準備的逐幀偵測結果。"""
    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = iter(frames)

    def detect(self, frame: Any) -> list[Detection]:
        del frame
        return next(self._frames, [])

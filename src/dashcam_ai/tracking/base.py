"""定義可替換的物件追蹤器介面。"""

from typing import Any, Protocol

from dashcam_ai.domain.perception import Detection, TrackedObject


class Tracker(Protocol):
    """將單幀偵測結果配對成帶有持續 ID 的物件。"""
    def update(self, detections: list[Detection], frame: Any) -> list[TrackedObject]: ...

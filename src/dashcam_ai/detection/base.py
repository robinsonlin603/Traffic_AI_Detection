"""定義可替換的物件偵測器介面。"""

from typing import Any, Protocol

from dashcam_ai.domain.perception import Detection


class Detector(Protocol):
    """接收影格並回傳尚未分配追蹤 ID 的偵測結果。"""
    def detect(self, frame: Any) -> list[Detection]: ...

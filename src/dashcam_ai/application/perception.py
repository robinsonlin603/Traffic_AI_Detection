"""定義 Analyzer 所依賴的感知後端介面。"""

from typing import Any, Protocol

from dashcam_ai.domain.perception import TrackedObject


class PerceptionBackend(Protocol):
    """可將單一影格轉換為追蹤物件的後端契約。"""
    def process(self, frame: Any) -> list[TrackedObject]: ...

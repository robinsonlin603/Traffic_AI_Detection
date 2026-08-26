"""定義可替換的車道幾何偵測器介面。"""

from typing import Any, Protocol

from dashcam_ai.domain.lane import LaneGeometry


class LaneDetector(Protocol):
    """將單幀影像轉換為標準化車道幾何。"""

    def detect(self, frame: Any, width: int, height: int) -> LaneGeometry: ...

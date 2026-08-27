"""定義可替換的自車相機運動估算介面。"""

from typing import Any, Protocol

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.motion import EgoMotionEstimate


class EgoMotionEstimator(Protocol):
    """從相鄰影格的背景特徵估算相機運動。"""

    def estimate(
        self, previous_frame: Any, current_frame: Any, excluded_boxes: list[BBox]
    ) -> EgoMotionEstimate: ...

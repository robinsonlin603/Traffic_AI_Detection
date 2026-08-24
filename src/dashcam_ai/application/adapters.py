"""將獨立的偵測器與追蹤器組合成統一的感知後端。"""

from typing import Any

from dashcam_ai.detection.base import Detector
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.tracking.base import Tracker


class DetectorTrackerBackend:
    """先執行物件偵測，再將結果交給追蹤器配對。"""
    def __init__(self, detector: Detector, tracker: Tracker) -> None:
        self.detector = detector
        self.tracker = tracker

    def process(self, frame: Any) -> list[TrackedObject]:
        """處理單一影格並回傳帶有持續 ID 的物件。"""
        detections = self.detector.detect(frame)
        return self.tracker.update(detections, frame)

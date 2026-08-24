"""公開物件偵測器介面及內建實作。"""

from dashcam_ai.detection.base import Detector
from dashcam_ai.detection.fake import FakeDetector
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker

__all__ = ["Detector", "FakeDetector", "UltralyticsDetectorTracker"]

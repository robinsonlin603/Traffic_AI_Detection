"""提供供測試使用的輕量中心點追蹤器。"""

from __future__ import annotations

from typing import Any

from dashcam_ai.domain.perception import Detection, TrackedObject


class CentroidTracker:
    """結果可重現的測試用輕量追蹤器，不是正式環境預設值。"""

    def __init__(self, maximum_distance: float = 100.0) -> None:
        self.maximum_distance = maximum_distance
        self._next_id = 1
        self._centers: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection], frame: Any) -> list[TrackedObject]:
        """以偵測框中心點距離配對上一幀 ID，並回傳追蹤物件。"""
        del frame
        available_ids = set(self._centers)
        tracked: list[TrackedObject] = []
        next_centers: dict[int, tuple[float, float]] = {}
        for detection in detections:
            center = detection.bbox.center
            best_id: int | None = None
            best_distance = self.maximum_distance
            # 在尚未配對的舊軌跡中，尋找距離門檻內最近的中心點。
            for track_id in available_ids:
                old_x, old_y = self._centers[track_id]
                distance = ((center.x - old_x) ** 2 + (center.y - old_y) ** 2) ** 0.5
                if distance <= best_distance:
                    best_id = track_id
                    best_distance = distance
            if best_id is None:
                # 沒有合適舊軌跡時，代表這是新出現的物件。
                best_id = self._next_id
                self._next_id += 1
            else:
                available_ids.remove(best_id)
            next_centers[best_id] = (center.x, center.y)
            tracked.append(
                TrackedObject(
                    track_id=best_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                )
            )
        self._centers = next_centers
        return tracked

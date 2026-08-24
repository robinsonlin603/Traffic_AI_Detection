"""在影片影格上繪製追蹤框、標籤及移動軌跡。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.video.reader import _cv2


class OpenCVAnnotator:
    """以 OpenCV 將追蹤資訊疊加到影格副本上。"""
    def __init__(self, trail_length: int = 30) -> None:
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def annotate(self, frame: Any, objects: list[TrackedObject]) -> Any:
        """回傳加入物件框、ID、信心分數與歷史軌跡的影格。"""
        cv2 = _cv2()
        output = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = (round(value) for value in obj.bbox.as_xyxy())
            cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 80), 2)
            label = f"{obj.class_name} #{obj.track_id} {obj.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 220, 80),
                2,
                cv2.LINE_AA,
            )
            # 底部中心點較接近物件與地面的接觸位置，適合呈現行進軌跡。
            bottom = obj.bbox.bottom_center
            trail = self._trails[obj.track_id]
            trail.append((round(bottom.x), round(bottom.y)))
            points = list(trail)
            for start, end in zip(points, points[1:], strict=False):
                cv2.line(output, start, end, (0, 180, 255), 2)
        return output

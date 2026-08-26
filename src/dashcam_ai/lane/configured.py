"""由人工校正的 normalized polygon 產生自車車道幾何。"""

from typing import Any

from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
    NormalizedPoint2D,
)


class ConfiguredLaneDetector:
    """使用設定式 polygon，不綁定任何 learned lane model。"""

    def __init__(self, polygon: list[NormalizedPoint2D], confidence: float = 1.0) -> None:
        if len(polygon) != 4:
            raise ValueError("configured ego-lane polygon must contain four points")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        self._polygon = polygon
        self._confidence = confidence

    def detect(self, frame: Any, width: int, height: int) -> LaneGeometry:
        """依照左上、右上、右下、左下順序映射 configured trapezoid。"""
        del frame
        points = [point.to_original(width, height) for point in self._polygon]
        return LaneGeometry(
            status=LaneGeometryStatus.VALID,
            provenance=LaneGeometryProvenance.CONFIGURED,
            confidence=self._confidence,
            frame_width=width,
            frame_height=height,
            ego_lane=LaneRegion(region_id="ego", polygon=points),
            boundaries=[
                LaneBoundary(boundary_id="left", points=[points[0], points[3]]),
                LaneBoundary(boundary_id="right", points=[points[1], points[2]]),
            ],
        )

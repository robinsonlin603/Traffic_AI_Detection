"""將 configured normalized polygon 映射為原始影像 forward corridor。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane import NormalizedPoint2D


def _contains(point: Point2D, polygon: tuple[Point2D, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        cross_product = (point.x - previous.x) * (current.y - previous.y) - (
            point.y - previous.y
        ) * (current.x - previous.x)
        if (
            abs(cross_product) <= 1e-9
            and min(previous.x, current.x) <= point.x <= max(previous.x, current.x)
            and min(previous.y, current.y) <= point.y <= max(previous.y, current.y)
        ):
            return True
        crosses_y = (current.y > point.y) != (previous.y > point.y)
        if crosses_y:
            crossing_x = (previous.x - current.x) * (point.y - current.y) / (
                previous.y - current.y
            ) + current.x
            if point.x < crossing_x:
                inside = not inside
        previous = current
    return inside


class ForwardCorridor(BaseModel):
    """原始影像座標中的前方互動區域。"""

    model_config = ConfigDict(frozen=True)
    polygon: tuple[Point2D, ...] = Field(min_length=3)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)

    def contains(self, point: Point2D) -> bool:
        return _contains(point, self.polygon)

    def interaction_score(self, bbox: BBox) -> float:
        """bottom-center 是主要證據，bbox center 僅提供部分輔助分數。"""
        if self.contains(bbox.bottom_center):
            return 1.0
        if self.contains(bbox.center):
            return 0.5
        return 0.0


class ConfiguredForwardCorridor:
    """由 normalized polygon 建立可跨解析度使用的 forward corridor。"""

    def __init__(self, polygon: list[NormalizedPoint2D]) -> None:
        if len(polygon) < 3:
            raise ValueError("forward corridor polygon requires at least three points")
        self._polygon = polygon

    def resolve(self, width: int, height: int) -> ForwardCorridor:
        return ForwardCorridor(
            polygon=tuple(point.to_original(width, height) for point in self._polygon),
            frame_width=width,
            frame_height=height,
        )

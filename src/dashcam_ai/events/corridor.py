"""將 configured normalized polygon 映射為原始影像 forward corridor。"""

from __future__ import annotations

from dashcam_ai.domain.lane import NormalizedPoint2D
from dashcam_ai.domain.scene import ForwardCorridor


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

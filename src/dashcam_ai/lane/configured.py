"""由人工校正的 normalized lanes 產生一般車道幾何。"""

from typing import Any

from dashcam_ai.domain.lane import (
    LaneBoundary,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
    LaneRegion,
    NormalizedLaneBoundary,
    NormalizedLaneRegion,
)


class ConfiguredLaneDetector:
    """使用設定式 polygon，不綁定任何 learned lane model。"""

    def __init__(
        self,
        lanes: list[NormalizedLaneRegion],
        boundaries: list[NormalizedLaneBoundary],
        confidence: float = 1.0,
    ) -> None:
        if not lanes:
            raise ValueError("configured geometry requires at least one lane")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        self._lanes = tuple(lanes)
        self._boundaries = tuple(boundaries)
        self._confidence = confidence

        lane_ids = [lane.lane_id for lane in lanes]
        orders = [lane.lateral_order for lane in lanes]
        boundary_ids = [boundary.boundary_id for boundary in boundaries]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("configured lane IDs must be unique")
        if len(set(orders)) != len(orders):
            raise ValueError("configured lane lateral orders must be unique")
        if len(set(boundary_ids)) != len(boundary_ids):
            raise ValueError("configured boundary IDs must be unique")
        known_lanes = set(lane_ids)
        for boundary in boundaries:
            if {boundary.left_lane_id, boundary.right_lane_id} - known_lanes:
                raise ValueError("configured boundary references an unknown lane")

    def detect(self, frame: Any, width: int, height: int) -> LaneGeometry:
        """將所有 configured normalized points 映射到原始影像座標。"""
        del frame
        return LaneGeometry(
            status=LaneGeometryStatus.VALID,
            provenance=LaneGeometryProvenance.CONFIGURED,
            confidence=self._confidence,
            frame_width=width,
            frame_height=height,
            lanes=tuple(
                LaneRegion(
                    lane_id=lane.lane_id,
                    lateral_order=lane.lateral_order,
                    polygon=tuple(point.to_original(width, height) for point in lane.polygon),
                )
                for lane in self._lanes
            ),
            boundaries=tuple(
                LaneBoundary(
                    boundary_id=boundary.boundary_id,
                    left_lane_id=boundary.left_lane_id,
                    right_lane_id=boundary.right_lane_id,
                    points=tuple(
                        point.to_original(width, height) for point in boundary.points
                    ),
                )
                for boundary in self._boundaries
            ),
        )

"""以 bottom-center 等錨點計算車道歸屬與 signed distance。"""

from __future__ import annotations

import math

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import (
    LaneGeometry,
    LaneGeometryStatus,
    LaneMembership,
    LaneMembershipFeature,
)


def _distance_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point.x - start.x, point.y - start.y)
    projection = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared
    projection = min(max(projection, 0.0), 1.0)
    nearest_x = start.x + projection * dx
    nearest_y = start.y + projection * dy
    return math.hypot(point.x - nearest_x, point.y - nearest_y)


def _inside_polygon(point: Point2D, polygon: list[Point2D]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _distance_to_segment(point, previous, current) <= 1e-9:
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


class LaneMembershipEvaluator:
    """使用邊界帶避免錨點在車道線附近反覆跳動。"""

    def __init__(
        self, boundary_margin: float, minimum_geometry_confidence: float = 0.0
    ) -> None:
        if boundary_margin < 0:
            raise ValueError("boundary margin must not be negative")
        if not 0 <= minimum_geometry_confidence <= 1:
            raise ValueError("minimum geometry confidence must be between zero and one")
        self.boundary_margin = boundary_margin
        self.minimum_geometry_confidence = minimum_geometry_confidence

    def evaluate(self, anchor: Point2D, geometry: LaneGeometry) -> LaneMembershipFeature:
        """正 signed distance 表示 polygon 內，負值表示外部。"""
        if (
            geometry.status is LaneGeometryStatus.UNKNOWN
            or not geometry.lanes
            or geometry.confidence < self.minimum_geometry_confidence
        ):
            return LaneMembershipFeature(
                membership=LaneMembership.UNKNOWN,
                anchor=anchor,
                geometry_confidence=geometry.confidence,
            )

        containing = [
            lane for lane in geometry.lanes if _inside_polygon(anchor, list(lane.polygon))
        ]
        distances = [
            (
                min(
                    _distance_to_segment(anchor, start, end)
                    for start, end in zip(boundary.points, boundary.points[1:], strict=False)
                ),
                boundary.boundary_id,
            )
            for boundary in geometry.boundaries
        ]
        if not distances:
            return LaneMembershipFeature(
                membership=LaneMembership.UNKNOWN,
                anchor=anchor,
                geometry_confidence=geometry.confidence,
            )
        distance, boundary_id = min(distances)
        boundary = next(
            item for item in geometry.boundaries if item.boundary_id == boundary_id
        )
        lanes_by_id = {lane.lane_id: lane for lane in geometry.lanes}
        signed_distance = distance if containing else -distance
        if distance <= self.boundary_margin:
            membership = LaneMembership.NEAR_BOUNDARY
            lane_id = containing[0].lane_id if len(containing) == 1 else None
        elif len(containing) == 1:
            membership = LaneMembership.INSIDE_LANE
            lane_id = containing[0].lane_id
        elif not containing:
            membership = LaneMembership.OUTSIDE_CONFIGURED_LANES
            lane_id = None
        else:
            membership = LaneMembership.UNKNOWN
            lane_id = None
        return LaneMembershipFeature(
            membership=membership,
            anchor=anchor,
            lane_id=lane_id,
            lane_lateral_order=(
                lanes_by_id[lane_id].lateral_order if lane_id is not None else None
            ),
            boundary_lane_ids=(boundary.left_lane_id, boundary.right_lane_id),
            boundary_lane_orders=(
                lanes_by_id[boundary.left_lane_id].lateral_order,
                lanes_by_id[boundary.right_lane_id].lateral_order,
            ),
            signed_boundary_distance=signed_distance,
            nearest_boundary_id=boundary_id,
            geometry_confidence=geometry.confidence,
        )

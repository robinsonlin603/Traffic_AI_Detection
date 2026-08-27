"""在影片影格上繪製追蹤框、標籤及移動軌跡。"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from dashcam_ai.domain.events import CutInEvent, EventStatus, LaneChangeEvent
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.domain.scene import FrameSceneAnalysis
from dashcam_ai.video.reader import _cv2

LabelBox = tuple[int, int, int, int]


class OpenCVAnnotator:
    """以 OpenCV 將追蹤資訊疊加到影格副本上。"""
    def __init__(self, trail_length: int = 30) -> None:
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def annotate(
        self,
        frame: Any,
        objects: list[TrackedObject],
        analysis: FrameSceneAnalysis | None = None,
    ) -> Any:
        """回傳加入物件框、ID、信心分數與歷史軌跡的影格。"""
        cv2 = _cv2()
        output = frame.copy()
        track_analysis = (
            {item.track_id: item for item in analysis.tracks} if analysis is not None else {}
        )
        occupied_labels: list[LabelBox] = []
        if analysis is not None:
            geometry = analysis.lane_geometry
            if geometry.ego_lane is not None:
                lane_points = np.asarray(
                    [(round(point.x), round(point.y)) for point in geometry.ego_lane.polygon],
                    dtype=np.int32,
                )
                cv2.polylines(output, [lane_points], True, (80, 220, 220), 2)
            for boundary in geometry.boundaries:
                boundary_points = np.asarray(
                    [(round(point.x), round(point.y)) for point in boundary.points],
                    dtype=np.int32,
                )
                cv2.polylines(output, [boundary_points], False, (40, 255, 120), 2)
            corridor_points = np.asarray(
                [
                    (round(point.x), round(point.y))
                    for point in analysis.forward_corridor.polygon
                ],
                dtype=np.int32,
            )
            cv2.polylines(output, [corridor_points], True, (220, 180, 40), 2)
        for obj in objects:
            x1, y1, x2, y2 = (round(value) for value in obj.bbox.as_xyxy())
            cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 80), 2)
            state = track_analysis.get(obj.track_id)
            lines = self._track_label_lines(obj, state)
            text_sizes = [
                cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                for line in lines
            ]
            text_width = max(size[0] for size in text_sizes)
            line_height = max(size[1] for size in text_sizes) + 5
            block_height = line_height * len(lines) + 6
            label_box = self._place_label(
                frame_width=output.shape[1],
                frame_height=output.shape[0],
                label_width=text_width + 8,
                label_height=block_height,
                bbox=(x1, y1, x2, y2),
                occupied=occupied_labels,
            )
            occupied_labels.append(label_box)
            left, top, right, bottom_edge = label_box
            cv2.rectangle(output, (left, top), (right, bottom_edge), (25, 25, 25), -1)
            label_color = self._track_label_color(state)
            for index, line in enumerate(lines):
                baseline = top + 5 + line_height * (index + 1) - 4
                cv2.putText(
                    output,
                    line,
                    (left + 4, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color,
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
        if analysis is not None:
            banners: list[LaneChangeEvent | CutInEvent] = [
                *analysis.lane_change_events,
                *analysis.cut_in_events,
            ]
            for index, event in enumerate(banners[-3:]):
                color = {
                    EventStatus.CANDIDATE: (0, 200, 255),
                    EventStatus.CONFIRMED: (0, 60, 255),
                    EventStatus.REJECTED: (160, 160, 160),
                }[event.status]
                cv2.putText(
                    output,
                    f"{event.event_type} #{event.track_id} {event.status.value}",
                    (20, 30 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return output

    @staticmethod
    def _track_label_lines(obj: TrackedObject, state: Any | None) -> tuple[str, ...]:
        primary = f"#{obj.track_id} {obj.class_name}"
        if state is None:
            return (primary,)
        membership = state.membership.membership.value
        status = state.temporal.status.value
        if status != "idle":
            return primary, status
        if membership in {"boundary", "unknown"}:
            return primary, membership
        return (primary,)

    @staticmethod
    def _track_label_color(state: Any | None) -> tuple[int, int, int]:
        if state is None:
            return (40, 220, 80)
        return {
            "candidate": (0, 200, 255),
            "confirmed": (0, 60, 255),
            "rejected": (160, 160, 160),
        }.get(state.temporal.status.value, (40, 220, 80))

    @classmethod
    def _place_label(
        cls,
        *,
        frame_width: int,
        frame_height: int,
        label_width: int,
        label_height: int,
        bbox: LabelBox,
        occupied: list[LabelBox],
    ) -> LabelBox:
        x1, y1, _, y2 = bbox
        left = min(max(x1, 0), max(frame_width - label_width, 0))
        candidates = [y1 - label_height, y1, y2]
        candidates.extend(y1 + offset * label_height for offset in range(1, 6))
        for candidate_top in candidates:
            top = min(max(candidate_top, 0), max(frame_height - label_height, 0))
            proposed = (left, top, left + label_width, top + label_height)
            if not any(cls._boxes_overlap(proposed, item) for item in occupied):
                return proposed
        top = min(max(y2, 0), max(frame_height - label_height, 0))
        return (left, top, left + label_width, top + label_height)

    @staticmethod
    def _boxes_overlap(first: LabelBox, second: LabelBox) -> bool:
        return not (
            first[2] <= second[0]
            or first[0] >= second[2]
            or first[3] <= second[1]
            or first[1] >= second[3]
        )

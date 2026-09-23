"""Resolve duplicate and briefly interrupted powered-vehicle track identities."""

from __future__ import annotations

from dataclasses import dataclass

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject


@dataclass(frozen=True, slots=True)
class _TrackState:
    frame_index: int
    bbox: BBox


class VehicleIdentityResolver:
    """Suppress duplicate tracks and reconnect IDs after one or two missing frames."""

    def __init__(
        self,
        *,
        duplicate_iou_threshold: float = 0.85,
        duplicate_containment_threshold: float = 0.9,
        duplicate_center_distance_ratio: float = 0.22,
        stitch_iou_threshold: float = 0.45,
        stitch_containment_threshold: float = 0.8,
        stitch_horizontal_center_ratio: float = 0.1,
        stitch_vertical_center_ratio: float = 0.3,
    ) -> None:
        self._duplicate_iou_threshold = duplicate_iou_threshold
        self._duplicate_containment_threshold = duplicate_containment_threshold
        self._duplicate_center_distance_ratio = duplicate_center_distance_ratio
        self._stitch_iou_threshold = stitch_iou_threshold
        self._stitch_containment_threshold = stitch_containment_threshold
        self._stitch_horizontal_center_ratio = stitch_horizontal_center_ratio
        self._stitch_vertical_center_ratio = stitch_vertical_center_ratio
        self._frame_index = -1
        self._aliases: dict[int, int] = {}
        self._last_seen: dict[int, _TrackState] = {}

    def update(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        """Return one object per vehicle with stable IDs where evidence is conservative."""
        self._frame_index += 1
        accepted: list[TrackedObject] = []
        accepted_raw_ids: list[int] = []
        for candidate in sorted(objects, key=lambda item: item.confidence, reverse=True):
            duplicate = next(
                (
                    item
                    for item in accepted
                    if self._boxes_are_duplicates(candidate.bbox, item.bbox)
                ),
                None,
            )
            if duplicate is not None:
                self._aliases[candidate.track_id] = duplicate.track_id
                continue
            canonical_id = self._aliases.get(candidate.track_id)
            if canonical_id is None:
                canonical_id = self._find_interrupted_track(candidate, accepted)
                self._aliases[candidate.track_id] = canonical_id
            collision_index = next(
                (index for index, item in enumerate(accepted) if item.track_id == canonical_id),
                None,
            )
            if collision_index is not None:
                existing_raw_id = accepted_raw_ids[collision_index]
                if candidate.track_id == canonical_id and existing_raw_id != canonical_id:
                    self._aliases[existing_raw_id] = existing_raw_id
                    accepted[collision_index] = accepted[collision_index].model_copy(
                        update={"track_id": existing_raw_id}
                    )
                else:
                    canonical_id = candidate.track_id
                    self._aliases[candidate.track_id] = candidate.track_id
            accepted.append(candidate.model_copy(update={"track_id": canonical_id}))
            accepted_raw_ids.append(candidate.track_id)

        for item in accepted:
            self._last_seen[item.track_id] = _TrackState(self._frame_index, item.bbox)
        return accepted

    def _find_interrupted_track(
        self, candidate: TrackedObject, accepted: list[TrackedObject]
    ) -> int:
        current_ids = {item.track_id for item in accepted}
        matches: list[tuple[float, float, int]] = []
        for track_id, state in self._last_seen.items():
            # Require one or two genuinely missing frames. Adjacent-frame replacements
            # such as two neighboring parked vehicles remain separate identities.
            frame_delta = self._frame_index - state.frame_index
            if frame_delta not in {2, 3} or track_id in current_ids:
                continue
            iou, containment, horizontal, vertical = self._geometry(candidate.bbox, state.bbox)
            if iou >= self._stitch_iou_threshold or (
                containment >= self._stitch_containment_threshold
                and horizontal <= self._stitch_horizontal_center_ratio
                and vertical <= self._stitch_vertical_center_ratio
            ):
                matches.append((iou, containment, track_id))
        if not matches:
            return candidate.track_id
        return max(matches)[2]

    def _boxes_are_duplicates(self, first: BBox, second: BBox) -> bool:
        iou, containment, horizontal, vertical = self._geometry(first, second)
        return iou >= self._duplicate_iou_threshold or (
            containment >= self._duplicate_containment_threshold
            and horizontal <= self._duplicate_center_distance_ratio
            and vertical <= self._duplicate_center_distance_ratio
        )

    @staticmethod
    def _geometry(first: BBox, second: BBox) -> tuple[float, float, float, float]:
        intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
        intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
        intersection = intersection_width * intersection_height
        first_width = first.x2 - first.x1
        first_height = first.y2 - first.y1
        second_width = second.x2 - second.x1
        second_height = second.y2 - second.y1
        first_area = first_width * first_height
        second_area = second_width * second_height
        union = first_area + second_area - intersection
        iou = intersection / union if union > 0 else 0.0
        smaller_area = min(first_area, second_area)
        containment = intersection / smaller_area if smaller_area > 0 else 0.0
        maximum_width = max(first_width, second_width)
        maximum_height = max(first_height, second_height)
        horizontal = abs(first.center.x - second.center.x) / maximum_width
        vertical = abs(first.center.y - second.center.y) / maximum_height
        return iou, containment, horizontal, vertical

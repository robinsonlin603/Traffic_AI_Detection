"""動態 lane-evidence backend 介面。"""

from __future__ import annotations

from typing import Any, Protocol

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane_evidence import LaneEvidenceFrame


class LaneEvidenceBackend(Protocol):
    def detect(
        self, frame: Any, road_roi: tuple[Point2D, ...]
    ) -> LaneEvidenceFrame: ...

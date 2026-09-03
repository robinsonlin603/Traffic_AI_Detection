"""安全的 dynamic/configured hybrid lane detector。"""

from __future__ import annotations

from typing import Any

from dashcam_ai.domain.lane import (
    LaneBoundaryEvidenceSource,
    LaneGeometry,
    LaneGeometryProvenance,
    LaneGeometryStatus,
)
from dashcam_ai.lane.base import LaneDetector


class HybridLaneDetector:
    """優先使用 dynamic；fallback 永遠標為 degraded，不能推進事件。"""

    def __init__(self, dynamic: LaneDetector, configured: LaneDetector) -> None:
        self._dynamic = dynamic
        self._configured = configured

    def detect(self, frame: Any, width: int, height: int) -> LaneGeometry:
        dynamic = self._dynamic.detect(frame, width, height)
        if dynamic.status is not LaneGeometryStatus.UNKNOWN:
            return dynamic
        configured = self._configured.detect(frame, width, height)
        return configured.model_copy(
            update={
                "status": LaneGeometryStatus.DEGRADED,
                "provenance": LaneGeometryProvenance.HYBRID,
                "boundaries": tuple(
                    boundary.model_copy(
                        update={"evidence_source": (LaneBoundaryEvidenceSource.CONFIGURED_FALLBACK)}
                    )
                    for boundary in configured.boundaries
                ),
                "reason": f"dynamic geometry unavailable: {dynamic.reason}",
            }
        )

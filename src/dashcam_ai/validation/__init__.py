"""Git-backed cross-platform validation records."""

from dashcam_ai.validation.records import (
    EnvironmentEvidence,
    GateResult,
    ResultStatus,
    ValidationRecord,
    calculate_verdict,
)

__all__ = [
    "EnvironmentEvidence",
    "GateResult",
    "ResultStatus",
    "ValidationRecord",
    "calculate_verdict",
]

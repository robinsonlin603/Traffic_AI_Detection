from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dashcam_ai.validation.records import (
    EnvironmentEvidence,
    GateResult,
    ResultStatus,
    ValidationRecord,
    calculate_verdict,
)


def environment(
    accelerator: str = "cuda", available: bool = True, operating_system: str = "Linux"
) -> EnvironmentEvidence:
    return EnvironmentEvidence(
        operating_system=operating_system,
        operating_system_release="6.6",
        python_version="3.12.1",
        accelerator=accelerator,
        accelerator_available=available,
        accelerator_name="Test GPU" if available else None,
    )


def gates(status: ResultStatus = ResultStatus.PASSED) -> list[GateResult]:
    return [
        GateResult(
            name=name,
            command=[name],
            status=status,
            exit_code=0 if status == ResultStatus.PASSED else 1,
            duration_seconds=0.1,
        )
        for name in ("pytest", "ruff", "mypy")
    ]


def test_clean_matching_platform_with_all_gates_passes() -> None:
    assert calculate_verdict(
        platform="linux-cuda",
        worktree_dirty=False,
        environment=environment(),
        gates=gates(),
    ) == (ResultStatus.PASSED, [])


def test_dirty_worktree_is_blocked() -> None:
    verdict, reasons = calculate_verdict(
        platform="linux-cuda",
        worktree_dirty=True,
        environment=environment(),
        gates=gates(),
    )
    assert verdict == ResultStatus.BLOCKED
    assert "uncommitted" in reasons[0]


def test_failed_gate_wins_over_blocked_environment() -> None:
    verdict, reasons = calculate_verdict(
        platform="linux-cuda",
        worktree_dirty=False,
        environment=environment(available=False),
        gates=gates(ResultStatus.FAILED),
    )
    assert verdict == ResultStatus.FAILED
    assert any("unavailable" in reason for reason in reasons)
    assert any("failed gates" in reason for reason in reasons)


def test_platform_result_cannot_be_transferred_to_another_accelerator() -> None:
    verdict, reasons = calculate_verdict(
        platform="macos-mps",
        worktree_dirty=False,
        environment=environment(accelerator="cuda"),
        gates=gates(),
    )
    assert verdict == ResultStatus.BLOCKED
    assert any("expected mps" in reason for reason in reasons)


def test_cuda_result_from_another_os_cannot_claim_linux() -> None:
    verdict, reasons = calculate_verdict(
        platform="linux-cuda",
        worktree_dirty=False,
        environment=environment(operating_system="Windows"),
        gates=gates(),
    )
    assert verdict == ResultStatus.BLOCKED
    assert any("expected Linux" in reason for reason in reasons)


def test_record_rejects_a_fabricated_verdict() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="do not match"):
        ValidationRecord(
            milestone="milestone-2",
            platform="linux-cuda",
            source_commit="a" * 40,
            worktree_dirty=True,
            started_at=now,
            finished_at=now,
            environment=environment(),
            gates=gates(),
            verdict=ResultStatus.PASSED,
            reasons=[],
        )

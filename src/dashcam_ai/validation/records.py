"""Validated schema and verdict rules for cross-platform evidence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORTED_PLATFORMS = {"macos-mps", "windows-cuda", "cpu"}
REQUIRED_GATES = {"pytest", "ruff", "mypy"}
MILESTONE_PLATFORMS = {"milestone-2": {"macos-mps", "windows-cuda"}}


class ResultStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    command: list[str]
    status: ResultStatus
    exit_code: int | None = None
    duration_seconds: float = Field(ge=0.0)
    output_excerpt: str = ""


class EnvironmentEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    operating_system: str
    operating_system_release: str
    python_version: str
    torch_version: str | None = None
    cuda_version: str | None = None
    ultralytics_version: str | None = None
    opencv_version: str | None = None
    accelerator: Literal["cpu", "mps", "cuda"]
    accelerator_available: bool
    accelerator_name: str | None = None


class AcceptanceEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    performed: bool = False
    input_sha256: str | None = None
    model_sha256: str | None = None
    config: str | None = None
    processed_frames: int | None = Field(default=None, ge=0)
    confirmed_events: int | None = Field(default=None, ge=0)
    rejected_events: int | None = Field(default=None, ge=0)


class ValidationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    milestone: str
    platform: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    worktree_dirty: bool
    started_at: datetime
    finished_at: datetime
    environment: EnvironmentEvidence
    gates: list[GateResult]
    acceptance_run: AcceptanceEvidence | None = None
    verdict: ResultStatus
    reasons: list[str]

    @model_validator(mode="after")
    def validate_consistency(self) -> ValidationRecord:
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(f"unsupported platform: {self.platform}")
        verdict, reasons = calculate_verdict(
            platform=self.platform,
            worktree_dirty=self.worktree_dirty,
            environment=self.environment,
            gates=self.gates,
        )
        if self.verdict != verdict or self.reasons != reasons:
            raise ValueError("verdict and reasons do not match the recorded evidence")
        return self


def expected_accelerator(platform_id: str) -> Literal["cpu", "mps", "cuda"]:
    mapping: dict[str, Literal["cpu", "mps", "cuda"]] = {
        "macos-mps": "mps",
        "windows-cuda": "cuda",
        "cpu": "cpu",
    }
    if platform_id not in mapping:
        raise ValueError(f"unsupported platform: {platform_id}")
    return mapping[platform_id]


def calculate_verdict(
    *,
    platform: str,
    worktree_dirty: bool,
    environment: EnvironmentEvidence,
    gates: list[GateResult],
) -> tuple[ResultStatus, list[str]]:
    """Return a deterministic verdict; blocked evidence never becomes a pass."""
    expected = expected_accelerator(platform)
    reasons: list[str] = []
    expected_os = {"macos-mps": "Darwin", "windows-cuda": "Windows"}.get(platform)
    gate_names = {gate.name for gate in gates}
    missing = sorted(REQUIRED_GATES - gate_names)
    if missing:
        reasons.append(f"missing required gates: {', '.join(missing)}")
    if worktree_dirty:
        reasons.append("worktree contained uncommitted changes")
    if expected_os is not None and environment.operating_system != expected_os:
        reasons.append(
            f"expected {expected_os} operating system but observed "
            f"{environment.operating_system}"
        )
    if environment.accelerator != expected:
        reasons.append(
            f"expected {expected} accelerator but observed {environment.accelerator}"
        )
    if not environment.accelerator_available:
        reasons.append(f"required {expected} accelerator was unavailable")
    blocked = [gate.name for gate in gates if gate.status == ResultStatus.BLOCKED]
    failed = [gate.name for gate in gates if gate.status == ResultStatus.FAILED]
    if blocked:
        reasons.append(f"blocked gates: {', '.join(blocked)}")
    if failed:
        reasons.append(f"failed gates: {', '.join(failed)}")
    if failed:
        return ResultStatus.FAILED, reasons
    if reasons:
        return ResultStatus.BLOCKED, reasons
    return ResultStatus.PASSED, []

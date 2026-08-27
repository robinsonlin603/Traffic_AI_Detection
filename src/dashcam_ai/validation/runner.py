"""Run validation gates and collect bounded, non-secret environment evidence."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import monotonic

from dashcam_ai.runtime.device import inspect_devices
from dashcam_ai.validation.records import (
    EnvironmentEvidence,
    GateResult,
    ResultStatus,
    ValidationRecord,
    calculate_verdict,
    expected_accelerator,
)

MAX_OUTPUT_CHARS = 4000


def _portable_command(command: list[str], root: Path) -> list[str]:
    """Remove machine-specific absolute paths from persisted command evidence."""
    portable: list[str] = []
    resolved_python = Path(sys.executable).resolve()
    for index, argument in enumerate(command):
        path = Path(argument)
        if index == 0 and path.is_absolute() and path.resolve() == resolved_python:
            portable.append("python")
        elif path.is_absolute():
            try:
                portable.append(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                portable.append(path.name)
        else:
            portable.append(argument)
    return portable


def _redact_text(value: str, root: Path) -> str:
    """Redact repository and home paths that tools may print in diagnostics."""
    redacted = value.replace(str(root.resolve()), "<repo>")
    return redacted.replace(str(Path.home().resolve()), "<home>")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class GateRunner:
    """Subprocess boundary kept injectable for deterministic tests."""

    def run(self, name: str, command: list[str], root: Path) -> GateResult:
        started = monotonic()
        persisted_command = _portable_command(command, root)
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        except (FileNotFoundError, OSError) as error:
            return GateResult(
                name=name,
                command=persisted_command,
                status=ResultStatus.BLOCKED,
                duration_seconds=monotonic() - started,
                output_excerpt=_redact_text(str(error), root)[-MAX_OUTPUT_CHARS:],
            )
        output = _redact_text(
            "\n".join(part for part in (result.stdout, result.stderr) if part), root
        )
        return GateResult(
            name=name,
            command=persisted_command,
            status=ResultStatus.PASSED if result.returncode == 0 else ResultStatus.FAILED,
            exit_code=result.returncode,
            duration_seconds=monotonic() - started,
            output_excerpt=output[-MAX_OUTPUT_CHARS:],
        )


def collect_environment(platform_id: str) -> EnvironmentEvidence:
    accelerator = expected_accelerator(platform_id)
    device_id = {"cpu": "cpu", "mps": "mps", "cuda": "cuda:0"}[accelerator]
    device = next(item for item in inspect_devices() if item.device == device_id)
    torch_version = _package_version("torch")
    cuda_version: str | None = None
    if torch_version is not None:
        try:
            import torch

            value = getattr(getattr(torch, "version", None), "cuda", None)
            cuda_version = str(value) if value is not None else None
        except ImportError:
            pass
    return EnvironmentEvidence(
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        ultralytics_version=_package_version("ultralytics"),
        opencv_version=_package_version("opencv-python"),
        accelerator=accelerator,
        accelerator_available=device.available,
        accelerator_name=device.name,
    )


def build_validation_record(
    root: Path, milestone: str, platform_id: str, gate_runner: GateRunner | None = None
) -> ValidationRecord:
    started_at = datetime.now(UTC)
    runner = gate_runner or GateRunner()
    commands = [
        ("pytest", [sys.executable, "-m", "pytest"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        ("mypy", [sys.executable, "-m", "mypy", "src"]),
    ]
    source_commit = _run_git(root, "rev-parse", "HEAD")
    worktree_dirty = bool(_run_git(root, "status", "--porcelain"))
    environment = collect_environment(platform_id)
    gates = [runner.run(name, command, root) for name, command in commands]
    verdict, reasons = calculate_verdict(
        platform=platform_id,
        worktree_dirty=worktree_dirty,
        environment=environment,
        gates=gates,
    )
    return ValidationRecord(
        milestone=milestone,
        platform=platform_id,
        source_commit=source_commit,
        worktree_dirty=worktree_dirty,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        environment=environment,
        gates=gates,
        verdict=verdict,
        reasons=reasons,
    )

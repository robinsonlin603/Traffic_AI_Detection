from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from dashcam_ai.cli import app
from dashcam_ai.validation.records import (
    EnvironmentEvidence,
    GateResult,
    ResultStatus,
    ValidationRecord,
)
from dashcam_ai.validation.render import write_report

runner = CliRunner()


def passing_record(commit: str, platform_id: str = "linux-cuda") -> ValidationRecord:
    now = datetime.now(UTC)
    accelerator = "cuda" if platform_id == "linux-cuda" else "mps"
    operating_system = "Linux" if platform_id == "linux-cuda" else "Darwin"
    return ValidationRecord(
        milestone="milestone-2",
        platform=platform_id,
        source_commit=commit,
        worktree_dirty=False,
        started_at=now,
        finished_at=now,
        environment=EnvironmentEvidence(
            operating_system=operating_system,
            operating_system_release="1",
            python_version="3.12",
            accelerator=accelerator,
            accelerator_available=True,
            accelerator_name="Test Accelerator",
        ),
        gates=[
            GateResult(
                name=name,
                command=[name],
                status=ResultStatus.PASSED,
                exit_code=0,
                duration_seconds=0.0,
            )
            for name in ("pytest", "ruff", "mypy")
        ],
        verdict=ResultStatus.PASSED,
        reasons=[],
    )


def test_validation_status_marks_an_old_commit_stale(monkeypatch, tmp_path: Path) -> None:
    write_report(tmp_path, passing_record("a" * 40))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dashcam_ai.validation.status._run_git", lambda root, *args: "b" * 40
    )
    monkeypatch.setattr("dashcam_ai.cli._run_git", lambda root, *args: "b" * 40)
    result = runner.invoke(
        app, ["validation-status", "validation/milestone-2/linux-cuda.json"]
    )
    assert result.exit_code == 1
    assert '"freshness": "stale"' in result.stdout
    assert '"verdict_for_current_commit": "invalid"' in result.stdout


def test_milestone_requires_both_platforms(monkeypatch, tmp_path: Path) -> None:
    commit = "c" * 40
    write_report(tmp_path, passing_record(commit, "macos-mps"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dashcam_ai.validation.status._run_git", lambda root, *args: commit
    )
    result = runner.invoke(app, ["milestone-status", "--milestone", "2"])
    assert result.exit_code == 1
    assert '"platform": "linux-cuda"' in result.stdout
    assert '"status": "missing"' in result.stdout
    assert '"verdict": "blocked"' in result.stdout


def test_validation_status_rejects_report_outside_validation(
    monkeypatch, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validation-status", str(outside)])
    assert result.exit_code == 2
    assert "must be inside" in result.output

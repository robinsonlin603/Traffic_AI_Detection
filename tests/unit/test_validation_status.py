from datetime import UTC, datetime
from pathlib import Path

from dashcam_ai.validation.records import (
    EnvironmentEvidence,
    GateResult,
    ResultStatus,
    ValidationRecord,
)
from dashcam_ai.validation.render import load_report, report_paths, write_report


def record(commit: str = "a" * 40) -> ValidationRecord:
    now = datetime.now(UTC)
    return ValidationRecord(
        milestone="milestone-2",
        platform="cpu",
        source_commit=commit,
        worktree_dirty=False,
        started_at=now,
        finished_at=now,
        environment=EnvironmentEvidence(
            operating_system="Test OS",
            operating_system_release="1",
            python_version="3.12",
            accelerator="cpu",
            accelerator_available=True,
            accelerator_name="Test CPU",
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


def test_report_round_trip_and_markdown_agree(tmp_path: Path) -> None:
    json_path, markdown_path = write_report(tmp_path, record())
    loaded = load_report(json_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert loaded.verdict == ResultStatus.PASSED
    assert loaded.source_commit in markdown
    assert "**passed**" in markdown


def test_repeated_write_replaces_stable_platform_files(tmp_path: Path) -> None:
    first_paths = write_report(tmp_path, record("a" * 40))
    second_paths = write_report(tmp_path, record("b" * 40))
    assert first_paths == second_paths
    assert load_report(second_paths[0]).source_commit == "b" * 40
    assert len(list(second_paths[0].parent.iterdir())) == 2


def test_report_paths_reject_path_traversal(tmp_path: Path) -> None:
    try:
        report_paths(tmp_path, "milestone-2", "../../outside")
    except ValueError as error:
        assert "unsupported platform" in str(error)
    else:
        raise AssertionError("unsafe platform was accepted")

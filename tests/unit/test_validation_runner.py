import sys
from pathlib import Path

from dashcam_ai.validation.records import ResultStatus
from dashcam_ai.validation.runner import MAX_OUTPUT_CHARS, GateRunner


def test_missing_gate_command_is_blocked(tmp_path: Path) -> None:
    result = GateRunner().run("missing", ["definitely-not-a-real-command"], tmp_path)
    assert result.status == ResultStatus.BLOCKED
    assert result.exit_code is None


def test_gate_output_is_bounded(tmp_path: Path) -> None:
    result = GateRunner().run(
        "large-output",
        [sys.executable, "-c", f"print('x' * {MAX_OUTPUT_CHARS + 100})"],
        tmp_path,
    )
    assert result.status == ResultStatus.PASSED
    assert len(result.output_excerpt) == MAX_OUTPUT_CHARS


def test_gate_evidence_redacts_private_absolute_paths(tmp_path: Path) -> None:
    result = GateRunner().run(
        "paths",
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        tmp_path,
    )
    assert result.status == ResultStatus.PASSED
    assert result.command[0] == "python"
    assert str(tmp_path) not in result.output_excerpt
    assert "<repo>" in result.output_excerpt

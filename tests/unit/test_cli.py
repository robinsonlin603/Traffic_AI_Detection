import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from dashcam_ai.cli import _resolve_model_path, _resolve_output_path, app


def test_devices_command_returns_json() -> None:
    result = CliRunner().invoke(app, ["devices"])

    assert result.exit_code == 0
    devices = json.loads(result.stdout)
    assert devices[0]["device"] == "cpu"
    assert devices[0]["available"] is True


def test_default_output_path_uses_input_filename() -> None:
    assert _resolve_output_path(Path("samples/test3.mp4"), None) == Path("output/test3")


def test_explicit_output_path_is_preserved() -> None:
    custom = Path("output/custom-name")

    assert _resolve_output_path(Path("samples/test3.mp4"), custom) == custom


def test_lane_overlay_command_is_removed() -> None:
    result = CliRunner().invoke(app, ["lane-overlay"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_analyze_passes_minimum_track_length_from_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("dashcam_ai.cli._resolve_model_path", lambda model: model)
    input_path = tmp_path / "input.mp4"
    input_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tracking:\n  minimum_track_length: 7\ndetection:\n  minimum_vehicle_area_ratio: 0.002\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeAnalyzer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def analyze(self, source: Path, output: Path) -> SimpleNamespace:
            return SimpleNamespace(
                frames_processed=0,
                tracks_created=0,
                events_created=0,
                elapsed_seconds=0.0,
                processing_fps=0.0,
                output_directory=output,
            )

    backend_options: dict[str, object] = {}

    def fake_backend(**kwargs: object) -> object:
        backend_options.update(kwargs)
        return object()

    monkeypatch.setattr("dashcam_ai.cli.UltralyticsDetectorTracker", fake_backend)
    monkeypatch.setattr("dashcam_ai.cli.Analyzer", FakeAnalyzer)

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--input",
            str(input_path),
            "--output",
            str(tmp_path / "output"),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["minimum_track_length"] == 7
    assert backend_options["minimum_vehicle_area_ratio"] == 0.002


def test_model_uses_current_directory_before_main_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "yolo26m.pt"
    weights.touch()
    assert _resolve_model_path("yolo26m.pt") == str(weights)


def test_model_falls_back_to_main_checkout(tmp_path, monkeypatch) -> None:
    main = tmp_path / "main"
    main.mkdir()
    weights = main / "yolo26m.pt"
    weights.touch()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    monkeypatch.chdir(worktree)
    monkeypatch.setattr(
        "dashcam_ai.cli.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=str(main / ".git")),
    )
    assert _resolve_model_path("yolo26m.pt") == str(weights)
    with pytest.raises(typer.BadParameter, match="不會自動下載"):
        _resolve_model_path("missing.pt")
    with pytest.raises(typer.BadParameter):
        _resolve_model_path("./yolo26m.pt")
    assert _resolve_model_path(str(weights)) == str(weights)


def test_missing_model_outside_git_has_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("dashcam_ai.cli.subprocess.run", no_git)
    with pytest.raises(typer.BadParameter, match="找不到本機模型"):
        _resolve_model_path("yolo26m.pt")

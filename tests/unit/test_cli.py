import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from dashcam_ai.cli import _build_scene_analyzer, _resolve_output_path, app
from dashcam_ai.config.models import AppConfig


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


def test_scene_analyzer_is_disabled_with_lane_geometry() -> None:
    config = AppConfig()
    disabled = config.model_copy(
        update={"lane_geometry": config.lane_geometry.model_copy(update={"enabled": False})}
    )

    assert _build_scene_analyzer(disabled) is None


def test_scene_analyzer_builds_from_default_configuration() -> None:
    assert _build_scene_analyzer(AppConfig()) is not None


def test_analyze_passes_minimum_track_length_from_config(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "tracking:\n  minimum_track_length: 7\nlane_geometry:\n  enabled: false\n",
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

    monkeypatch.setattr("dashcam_ai.cli.UltralyticsDetectorTracker", lambda **kwargs: object())
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

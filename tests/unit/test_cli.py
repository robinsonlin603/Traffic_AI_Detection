import json
from pathlib import Path

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

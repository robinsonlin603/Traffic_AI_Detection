import json
from pathlib import Path

from typer.testing import CliRunner

from dashcam_ai.cli import _resolve_output_path, app


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

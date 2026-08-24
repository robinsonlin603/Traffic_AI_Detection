import json

from typer.testing import CliRunner

from dashcam_ai.cli import app


def test_devices_command_returns_json() -> None:
    result = CliRunner().invoke(app, ["devices"])

    assert result.exit_code == 0
    devices = json.loads(result.stdout)
    assert devices[0]["device"] == "cpu"
    assert devices[0]["available"] is True

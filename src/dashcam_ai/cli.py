"""提供行車記錄器影片分析的命令列介面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.config.models import load_config
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker
from dashcam_ai.logging import configure_logging
from dashcam_ai.runtime.device import inspect_devices

app = typer.Typer(no_args_is_help=True, help="Analyze motorcycle dashcam videos locally.")


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """未指定輸出目錄時，以來源影片檔名建立預設目錄。"""
    return output_path if output_path is not None else Path("output") / input_path.stem


@app.callback()
def main() -> None:
    """Motorcycle dashcam analysis commands."""


@app.command()
def devices() -> None:
    """列出目前電腦可用的推論裝置。"""
    typer.echo(
        json.dumps(
            [item.model_dump(mode="json") for item in inspect_devices()],
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def analyze(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output_path: Annotated[Path | None, typer.Option("--output", file_okay=False)] = None,
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/default.yaml"
    ),
    model: Annotated[str | None, typer.Option("--model")] = None,
    imgsz: Annotated[int | None, typer.Option("--imgsz", min=1)] = None,
    confidence: Annotated[float | None, typer.Option("--confidence", min=0.0, max=1.0)] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    save_video: Annotated[bool | None, typer.Option("--save-video/--no-save-video")] = None,
    save_frames: Annotated[bool | None, typer.Option("--save-frames/--no-save-frames")] = None,
) -> None:
    """對 MP4 影片執行 YOLO 偵測與 BoT-SORT 追蹤。"""
    resolved_output_path = _resolve_output_path(input_path, output_path)
    config = load_config(config_path)
    configure_logging(config.logging.level)
    detection = config.detection
    output = config.output
    # 命令列參數優先於設定檔，未指定時才採用 YAML 中的預設值。
    backend = UltralyticsDetectorTracker(
        model=model or detection.model,
        confidence=confidence if confidence is not None else detection.confidence,
        imgsz=imgsz or detection.imgsz,
        class_names=detection.classes,
        tracker=config.tracking.tracker,
        device=device if device is not None else detection.device,
    )
    analyzer = Analyzer(
        perception=backend,
        save_video=output.save_video if save_video is None else save_video,
        save_frames=output.save_frames if save_frames is None else save_frames,
        codec=output.codec,
    )
    summary = analyzer.analyze(input_path, resolved_output_path)
    typer.echo(
        json.dumps(
            {
                "frames_processed": summary.frames_processed,
                "tracks_created": summary.tracks_created,
                "elapsed_seconds": round(summary.elapsed_seconds, 3),
                "processing_fps": round(summary.processing_fps, 3),
                "output_directory": str(summary.output_directory.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()

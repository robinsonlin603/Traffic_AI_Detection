"""提供行車記錄器影片分析的命令列介面。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.config.models import load_config
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector
from dashcam_ai.logging import configure_logging
from dashcam_ai.runtime.device import inspect_devices
from dashcam_ai.validation.records import SUPPORTED_PLATFORMS
from dashcam_ai.validation.render import write_report
from dashcam_ai.validation.runner import _run_git, build_validation_record
from dashcam_ai.validation.status import inspect_report, milestone_status

app = typer.Typer(no_args_is_help=True, help="Analyze motorcycle dashcam videos locally.")


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """未指定輸出目錄時，以來源影片檔名建立預設目錄。"""
    return output_path if output_path is not None else Path("output") / input_path.stem


def _resolve_model_path(model: str) -> str:
    """Find local weights, sharing the main checkout's weights with Git worktrees."""
    path = Path(model).expanduser()
    if path.is_file():
        return str(path.resolve())
    # Only bare filenames may fall back. Explicit paths must not silently select
    # a different model when misspelled or missing.
    if model == path.name:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            candidate = Path(result.stdout.strip()).parent / path.name
            if candidate.is_file():
                return str(candidate.resolve())
        except (OSError, subprocess.SubprocessError):
            pass
    raise typer.BadParameter(
        f"找不到本機模型：{model}。請將權重放在目前或主要 Git 工作目錄，"
        "或使用 --model 指定既有檔案；不會自動下載。",
        param_hint="--model / detection.model",
    )


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


@app.command("validate")
def validate_platform(
    milestone: Annotated[str, typer.Option("--milestone")] = "1",
    platform_id: Annotated[str, typer.Option("--platform")] = "cpu",
) -> None:
    """執行共通 gates 並寫入目前平台的 Git-friendly 驗證報告。"""
    normalized_milestone = (
        milestone if milestone.startswith("milestone-") else f"milestone-{milestone}"
    )
    if normalized_milestone != "milestone-1":
        raise typer.BadParameter("currently supported milestone: 1", param_hint="--milestone")
    if platform_id not in SUPPORTED_PLATFORMS:
        raise typer.BadParameter(
            f"supported platforms: {', '.join(sorted(SUPPORTED_PLATFORMS))}",
            param_hint="--platform",
        )
    root = Path.cwd()
    record = build_validation_record(root, normalized_milestone, platform_id)
    json_path, markdown_path = write_report(root, record)
    typer.echo(
        json.dumps(
            {
                "report": str(json_path),
                "summary": str(markdown_path),
                "source_commit": record.source_commit,
                "verdict": record.verdict.value,
                "reasons": record.reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if record.verdict.value != "passed":
        raise typer.Exit(code=1)


@app.command("validation-status")
def validation_status(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """檢查單一報告是否適用於目前 checkout。"""
    root = Path.cwd()
    try:
        record, fresh = inspect_report(report, root)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="report") from error
    typer.echo(
        json.dumps(
            {
                "milestone": record.milestone,
                "platform": record.platform,
                "tested_commit": record.source_commit,
                "current_commit": _run_git(root, "rev-parse", "HEAD"),
                "freshness": "current" if fresh else "stale",
                "worktree": "dirty" if record.worktree_dirty else "clean",
                "gates": {gate.name: gate.status.value for gate in record.gates},
                "accelerator_available": record.environment.accelerator_available,
                "accelerator_name": record.environment.accelerator_name,
                "recorded_verdict": record.verdict.value,
                "verdict_for_current_commit": record.verdict.value if fresh else "invalid",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not fresh or record.verdict.value != "passed":
        raise typer.Exit(code=1)


@app.command("milestone-status")
def show_milestone_status(
    milestone: Annotated[str, typer.Option("--milestone")] = "1",
) -> None:
    """彙整目前 commit 所需的 macOS 與 Linux 平台證據。"""
    normalized = milestone if milestone.startswith("milestone-") else f"milestone-{milestone}"
    try:
        statuses, overall = milestone_status(Path.cwd(), normalized)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--milestone") from error
    typer.echo(
        json.dumps(
            {
                "milestone": normalized,
                "platforms": [item.__dict__ for item in statuses],
                "verdict": overall,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if overall != "passed":
        raise typer.Exit(code=1)


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
    lane = config.lane_detection
    if lane.enabled and lane.backend != "yolop_onnx":
        raise typer.BadParameter(
            f"不支援的白線模型後端：{lane.backend}",
            param_hint="lane_detection.backend",
        )
    # 命令列參數優先於設定檔，未指定時才採用 YAML 中的預設值。
    backend = UltralyticsDetectorTracker(
        model=_resolve_model_path(model if model is not None else detection.model),
        confidence=confidence if confidence is not None else detection.confidence,
        imgsz=imgsz or detection.imgsz,
        class_names=detection.classes,
        minimum_vehicle_area_ratio=detection.minimum_vehicle_area_ratio,
        ego_vehicle_polygon=detection.ego_vehicle_polygon,
        ego_vehicle_overlap_threshold=detection.ego_vehicle_overlap_threshold,
        duplicate_vehicle_iou_threshold=detection.duplicate_vehicle_iou_threshold,
        duplicate_vehicle_containment_threshold=(
            detection.duplicate_vehicle_containment_threshold
        ),
        duplicate_vehicle_center_distance_ratio=(
            detection.duplicate_vehicle_center_distance_ratio
        ),
        tracker=config.tracking.tracker,
        device=device if device is not None else detection.device,
    )
    analyzer = Analyzer(
        perception=backend,
        save_video=output.save_video if save_video is None else save_video,
        save_frames=output.save_frames if save_frames is None else save_frames,
        codec=output.codec,
        minimum_track_length=config.tracking.minimum_track_length,
        lane_detector=(
            YoloPLaneLineDetector(
                model=_resolve_model_path(lane.model),
                expected_sha256=lane.model_sha256 or None,
                input_size=lane.input_size,
                probability_threshold=lane.probability_threshold,
                minimum_drivable_probability=lane.minimum_drivable_probability,
                white_lightness_threshold=lane.white_lightness_threshold,
                white_saturation_threshold=lane.white_saturation_threshold,
                local_contrast_threshold=lane.local_contrast_threshold,
                strong_probability_threshold=lane.strong_probability_threshold,
                roi_top_ratio=lane.roi_top_ratio,
                minimum_component_area_ratio=lane.minimum_component_area_ratio,
                minimum_vertical_span_ratio=lane.minimum_vertical_span_ratio,
                minimum_fragment_vertical_span_ratio=(
                    lane.minimum_fragment_vertical_span_ratio
                ),
                maximum_horizontal_to_vertical_ratio=(
                    lane.maximum_horizontal_to_vertical_ratio
                ),
                maximum_fit_error_ratio=lane.maximum_fit_error_ratio,
                maximum_arrow_fit_error_ratio=lane.maximum_arrow_fit_error_ratio,
                maximum_row_width_ratio=lane.maximum_row_width_ratio,
                maximum_row_width_variation_ratio=(
                    lane.maximum_row_width_variation_ratio
                ),
                maximum_fragment_gap_ratio=lane.maximum_fragment_gap_ratio,
                sample_count=lane.sample_count,
                smoothing_alpha=lane.smoothing_alpha,
                maximum_missing_frames=lane.maximum_missing_frames,
                minimum_curve_confidence=lane.minimum_curve_confidence,
                maximum_boundaries=lane.maximum_boundaries,
                temporal_association_distance_ratio=(
                    lane.temporal_association_distance_ratio
                ),
                excluded_bbox_margin_ratio=lane.excluded_bbox_margin_ratio,
                minimum_confirmation_frames=lane.minimum_confirmation_frames,
            )
            if lane.enabled
            else None
        ),
    )
    summary = analyzer.analyze(input_path, resolved_output_path)
    typer.echo(
        json.dumps(
            {
                "frames_processed": summary.frames_processed,
                "tracks_created": summary.tracks_created,
                "events_created": summary.events_created,
                "elapsed_seconds": round(summary.elapsed_seconds, 3),
                "processing_fps": round(summary.processing_fps, 3),
                "output_directory": str(summary.output_directory.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()

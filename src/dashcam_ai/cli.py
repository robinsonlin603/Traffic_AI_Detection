"""提供行車記錄器影片分析的命令列介面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.application.scene import StreamingSceneAnalyzer
from dashcam_ai.config.models import AppConfig, load_config
from dashcam_ai.detection.ultralytics import UltralyticsDetectorTracker
from dashcam_ai.events.corridor import ConfiguredForwardCorridor
from dashcam_ai.events.cutin import CutInDetector
from dashcam_ai.events.lane_change import LaneChangeEventBuilder
from dashcam_ai.lane.configured import ConfiguredLaneDetector
from dashcam_ai.lane.membership import LaneMembershipEvaluator
from dashcam_ai.lane.temporal import TemporalLaneTracker
from dashcam_ai.logging import configure_logging
from dashcam_ai.motion.opencv import OpenCVEgoMotionEstimator
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
        scene_analyzer=_build_scene_analyzer(config),
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


def _build_scene_analyzer(config: AppConfig) -> StreamingSceneAnalyzer | None:
    """由 validated application config 建立 Milestone 2 scene pipeline。"""
    lane = config.lane_geometry
    if not lane.enabled:
        return None
    motion = config.ego_motion
    temporal = config.temporal_lane
    cutin = config.cut_in
    return StreamingSceneAnalyzer(
        lane_detector=ConfiguredLaneDetector(lane.ego_lane_polygon, lane.confidence),
        membership_evaluator=LaneMembershipEvaluator(
            config.lane_membership.boundary_margin_pixels
        ),
        motion_estimator=OpenCVEgoMotionEstimator(
            max_features=motion.max_features,
            feature_quality_level=motion.feature_quality_level,
            feature_min_distance=motion.feature_min_distance,
            optical_flow_window_size=motion.optical_flow_window_size,
            optical_flow_max_level=motion.optical_flow_max_level,
            ransac_reprojection_threshold=motion.ransac_reprojection_threshold,
            minimum_tracked_features=motion.minimum_tracked_features,
            minimum_inliers=motion.minimum_inliers,
            minimum_inlier_ratio=motion.minimum_inlier_ratio,
            maximum_mean_reprojection_error=motion.maximum_mean_reprojection_error,
            mask_padding_pixels=motion.mask_padding_pixels,
        ),
        temporal_tracker=TemporalLaneTracker(
            smoothing_window_frames=temporal.smoothing_window_frames,
            approaching_distance_pixels=temporal.approaching_distance_pixels,
            entered_distance_pixels=temporal.entered_distance_pixels,
            debounce_frames=temporal.debounce_frames,
            minimum_confirmation_frames=temporal.minimum_confirmation_frames,
            minimum_confirmation_duration_seconds=(
                temporal.minimum_confirmation_duration_seconds
            ),
            maximum_missing_frames=temporal.maximum_missing_frames,
            candidate_timeout_seconds=temporal.candidate_timeout_seconds,
            history_size=temporal.history_size,
        ),
        corridor=ConfiguredForwardCorridor(config.forward_corridor.polygon),
        lane_change_builder=LaneChangeEventBuilder(cutin.evidence_history_size),
        cut_in_detector=CutInDetector(
            minimum_bbox_expansion_ratio=cutin.minimum_bbox_expansion_ratio,
            minimum_confirmed_confidence=cutin.minimum_confirmed_confidence,
            minimum_motion_quality_ratio=cutin.minimum_motion_quality_ratio,
            lane_change_weight=cutin.lane_change_weight,
            corridor_weight=cutin.corridor_weight,
            bbox_expansion_weight=cutin.bbox_expansion_weight,
            motion_quality_weight=cutin.motion_quality_weight,
        ),
        maximum_missing_frames=temporal.maximum_missing_frames,
    )


if __name__ == "__main__":
    app()

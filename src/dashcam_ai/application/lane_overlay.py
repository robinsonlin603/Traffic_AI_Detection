"""不接事件 pipeline 的 lane-evidence 視覺驗證工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import Point2D
from dashcam_ai.domain.lane import LaneGeometry, LaneGeometryStatus, NormalizedPoint2D
from dashcam_ai.domain.lane_evidence import LaneEvidenceFrame
from dashcam_ai.lane.dynamic import TemporalLaneGeometryTracker
from dashcam_ai.lane.evidence import LaneEvidenceBackend
from dashcam_ai.video.reader import OpenCVVideoReader, _cv2
from dashcam_ai.video.writer import OpenCVVideoWriter


@dataclass(frozen=True, slots=True)
class LaneOverlaySummary:
    frames_processed: int
    valid_frames: int
    degraded_frames: int
    unknown_frames: int
    output_directory: Path


def run_lane_overlay(
    source: Path,
    output_directory: Path,
    backend: LaneEvidenceBackend,
    road_roi: tuple[NormalizedPoint2D, ...],
    *,
    codec: str = "mp4v",
    maximum_frames: int | None = None,
    temporal_tracker: TemporalLaneGeometryTracker | None = None,
) -> LaneOverlaySummary:
    """逐幀寫出候選曲線與 JSONL，不執行偵測、追蹤或事件分析。"""
    if maximum_frames is not None and maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    counts = {status: 0 for status in LaneGeometryStatus}
    processed = 0
    with OpenCVVideoReader(source) as reader:
        roi = tuple(
            point.to_original(reader.metadata.width, reader.metadata.height) for point in road_roi
        )
        writer = OpenCVVideoWriter(output_directory / "lane-overlay.mp4", reader.metadata, codec)
        evidence_path = output_directory / "lane-evidence.jsonl"
        try:
            with evidence_path.open("w", encoding="utf-8") as stream:
                for video_frame in reader:
                    if maximum_frames is not None and processed >= maximum_frames:
                        break
                    evidence = backend.detect(video_frame.image, roi)
                    geometry = (
                        temporal_tracker.update(
                            evidence,
                            width=reader.metadata.width,
                            height=reader.metadata.height,
                            road_roi=roi,
                        )
                        if temporal_tracker is not None
                        else None
                    )
                    output_status = geometry.status if geometry is not None else evidence.status
                    counts[output_status] += 1
                    stream.write(
                        json.dumps(
                            {
                                "frame_id": video_frame.frame_id,
                                "timestamp": video_frame.timestamp,
                                "evidence": evidence.model_dump(mode="json"),
                                "geometry": (
                                    geometry.model_dump(mode="json")
                                    if geometry is not None
                                    else None
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    writer.write(
                        annotate_lane_evidence(video_frame.image, roi, evidence, geometry=geometry)
                    )
                    processed += 1
        finally:
            writer.close()
    summary = LaneOverlaySummary(
        frames_processed=processed,
        valid_frames=counts[LaneGeometryStatus.VALID],
        degraded_frames=counts[LaneGeometryStatus.DEGRADED],
        unknown_frames=counts[LaneGeometryStatus.UNKNOWN],
        output_directory=output_directory,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(
            {
                "frames_processed": summary.frames_processed,
                "valid_frames": summary.valid_frames,
                "degraded_frames": summary.degraded_frames,
                "unknown_frames": summary.unknown_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def annotate_lane_evidence(
    frame: Any,
    road_roi: tuple[Point2D, ...],
    evidence: LaneEvidenceFrame,
    *,
    geometry: LaneGeometry | None = None,
) -> Any:
    """以 prototype 專用樣式畫 ROI、曲線、品質與安全狀態。"""
    cv2 = _cv2()
    output = frame.copy()
    roi_array = np.asarray([[round(point.x), round(point.y)] for point in road_roi], dtype=np.int32)
    cv2.polylines(output, [roi_array], True, (180, 120, 40), 2)
    status_color = {
        LaneGeometryStatus.VALID: (40, 220, 80),
        LaneGeometryStatus.DEGRADED: (0, 200, 255),
        LaneGeometryStatus.UNKNOWN: (120, 120, 120),
    }[evidence.status]
    for curve in evidence.curves:
        points = np.asarray(
            [[round(point.x), round(point.y)] for point in curve.points], dtype=np.int32
        )
        cv2.polylines(output, [points], False, status_color, 3)
    if geometry is not None:
        geometry_color = {
            LaneGeometryStatus.VALID: (40, 220, 80),
            LaneGeometryStatus.DEGRADED: (0, 200, 255),
            LaneGeometryStatus.UNKNOWN: (120, 120, 120),
        }[geometry.status]
        for boundary in geometry.boundaries:
            points = np.asarray(
                [[round(point.x), round(point.y)] for point in boundary.points],
                dtype=np.int32,
            )
            cv2.polylines(output, [points], False, geometry_color, 6)
    lane_count = (
        str(evidence.estimated_lane_count)
        if evidence.estimated_lane_count is not None
        else "unknown"
    )
    cv2.putText(
        output,
        (
            f"LANE EVIDENCE {evidence.status.value.upper()} "
            f"conf={evidence.confidence:.2f} curves={len(evidence.curves)} "
            f"lanes={lane_count}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    if geometry is not None:
        cv2.putText(
            output,
            (
                f"TEMPORAL {geometry.status.value.upper()} "
                f"topology={geometry.topology_id or 'pending'} "
                f"lanes={len(geometry.lanes)}"
            ),
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            geometry_color,
            2,
            cv2.LINE_AA,
        )
    return output

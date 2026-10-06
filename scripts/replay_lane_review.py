"""Replay lane detection with identical model output and optional recorded vehicle boxes.

Run from the repository root with PYTHONPATH=src. The output directory must not exist.
This is CPU lane evidence, not a vehicle-tracker or accelerator acceptance run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashcam_ai.config.models import load_config
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector
from dashcam_ai.visualization.annotator import OpenCVAnnotator


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def detector_settings(path: Path) -> dict[str, Any]:
    settings = load_config(path).lane_detection.model_dump()
    settings.pop("enabled")
    settings.pop("backend")
    settings["expected_sha256"] = settings.pop("model_sha256") or None
    return settings


def verify_video(path: Path, expected: int) -> None:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    if decoded != expected:
        raise ValueError(f"{path.name}: decoded {decoded}, expected {expected} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--boxes", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--snapshot-interval", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.start < 0 or args.snapshot_interval < 1:
        parser.error("start must be nonnegative; snapshot interval must be positive")
    cv2.setNumThreads(4)
    settings = detector_settings(args.config)
    detector = YoloPLaneLineDetector(**settings)
    baseline = None
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location("lane_review_baseline", args.baseline_source)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load the supplied baseline source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = module.YoloPLaneLineDetector(**settings)
    boxes = {}
    if args.boxes:
        for line in args.boxes.read_text().splitlines():
            row = json.loads(line)
            if row["frame_id"] in boxes:
                raise ValueError("duplicate vehicle frame")
            boxes[row["frame_id"]] = row
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError("cannot open input video")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = total if args.stop is None else args.stop
    if not 0 <= args.start < stop <= total or fps <= 0:
        raise ValueError("invalid frame range or fps")
    if args.boxes and any(i not in boxes for i in range(args.start, stop)):
        raise ValueError("recorded vehicle boxes are missing requested frames")
    inputs = [args.input, args.config, Path(settings["model"])]
    inputs += [path for path in (args.boxes, args.baseline_source) if path is not None]
    source_files = sorted(Path("src/dashcam_ai").rglob("*.py")) + [Path(__file__)]
    source_hashes = {str(path): digest(path) for path in source_files}
    manifest = {
        "schema_version": 1,
        "scope": "CPU lane replay; vehicle boxes reused" if args.boxes else "CPU lane-only replay",
        "inputs": {str(path): digest(path) for path in inputs},
        "source_hashes": source_hashes,
        "frame_range": [args.start, stop],
        "width": width, "height": height, "fps": fps,
        "settings": settings,
        "complete": False,
    }
    args.output.mkdir(parents=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    writers = {}
    for name, video_width in (("annotated", width), ("comparison", width * 2)):
        if name == "comparison" and baseline is None:
            continue
        writer = cv2.VideoWriter(
            str(args.output / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (video_width, height),
        )
        if not writer.isOpened():
            raise ValueError("cannot open output video")
        writers[name] = writer
    counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    baseline_counts: Counter[str] = Counter()
    annotator = OpenCVAnnotator()
    baseline_annotator = OpenCVAnnotator()
    infer = detector._segmentation_probabilities
    started = time.monotonic()
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    try:
        with (args.output / "lane-lines.jsonl").open("w") as log, (
            args.output / "before.jsonl"
        ).open("w") as before_log:
            for frame_id in range(args.start, stop):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"input cannot decode frame {frame_id}")
                objects = [
                    TrackedObject(**obj) for obj in boxes.get(frame_id, {}).get("objects", [])
                ]
                probability, road = infer(frame)
                detector._segmentation_probabilities = lambda _, p=probability, r=road: (p, r)
                result = detector.detect(frame, [obj.bbox for obj in objects])
                occlusion_runs = []
                if detector.occlusion_mask is not None:
                    transitions = np.flatnonzero(np.diff(np.r_[
                        False, detector.occlusion_mask.ravel() > 0, False,
                    ]))
                    occlusion_runs = transitions.reshape((-1, 2)).tolist()
                log.write(json.dumps({
                    "frame_id": frame_id, "timestamp": frame_id / fps,
                    "lane_lines": result.model_dump(mode="json"),
                    "occlusion_runs": occlusion_runs,
                }) + "\n")
                counts[result.status.value] += 1
                rejections.update(result.diagnostics.rejection_reasons)
                annotated = annotator.annotate(frame, objects, result, detector.occlusion_mask)
                writers["annotated"].write(annotated)
                comparison = None
                if baseline is not None:
                    baseline._segmentation_probabilities = lambda _, p=probability, r=road: (p, r)
                    old = baseline.detect(frame, [obj.bbox for obj in objects])
                    baseline_runs = []
                    if baseline.occlusion_mask is not None:
                        transitions = np.flatnonzero(np.diff(np.r_[
                            False, baseline.occlusion_mask.ravel() > 0, False,
                        ]))
                        baseline_runs = transitions.reshape((-1, 2)).tolist()
                    before_log.write(json.dumps({
                        "frame_id": frame_id, "timestamp": frame_id / fps,
                        "lane_lines": old.model_dump(mode="json"),
                        "occlusion_runs": baseline_runs,
                    }) + "\n")
                    baseline_counts[old.status.value] += 1
                    old_frame = baseline_annotator.annotate(
                        frame, objects, old, baseline.occlusion_mask
                    )
                    comparison = np.hstack((old_frame, annotated))
                    for label, x in (("before", 10), ("after", width + 10)):
                        cv2.putText(comparison, f"{label} / {frame_id / fps:.2f}s",
                                    (x, 30), 0, 0.8, (255, 255, 255), 2)
                    writers["comparison"].write(comparison)
                if frame_id % args.snapshot_interval == 0:
                    cv2.imwrite(str(args.output / f"review-{frame_id}.jpg"),
                                annotated if comparison is None else comparison)
                if (frame_id - args.start + 1) % 120 == 0:
                    print(json.dumps({"processed": frame_id - args.start + 1,
                                      "expected": stop - args.start}), flush=True)
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
    for name in writers:
        verify_video(args.output / f"{name}.mp4", stop - args.start)
    if source_hashes != {str(path): digest(path) for path in source_files}:
        raise ValueError("source changed during replay")
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "frames": stop - args.start, "counts": dict(counts),
        "before_counts": dict(baseline_counts), "rejections": dict(rejections),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_hashes": {path.name: digest(path) for path in args.output.iterdir()
                          if path.suffix in (".mp4", ".jsonl")},
        "decoded_videos": list(writers),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

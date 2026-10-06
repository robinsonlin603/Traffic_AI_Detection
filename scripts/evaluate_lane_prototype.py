"""Train and evaluate opt-in lane candidates and body masks on fixed temporal windows.

Run from repository root with PYTHONPATH=src. Neither subcommand changes analyze defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from check_lane_review import distance_to_curves, evaluate_marking_cases, summarize_markings
from replay_lane_review import detector_settings, digest, verify_video

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.lane_lines import LaneCurve
from dashcam_ai.lane.prototype import (
    ExperimentalLaneDetector,
    MarkingClassifier,
    VehicleInstance,
    curve_features,
    mask_runs,
    pixel_occlusion,
)
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector
from dashcam_ai.visualization.annotator import OpenCVAnnotator

VARIANTS = ("baseline", "pixel_mask", "classifier", "combined")


def read_boxes(path: Path) -> dict[int, list[BBox]]:
    records: dict[int, list[BBox]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["frame_id"] in records:
            raise ValueError("duplicate recorded vehicle frame")
        records[row["frame_id"]] = [BBox(**obj["bbox"]) for obj in row["objects"]]
    return records


def source_hashes() -> dict[str, str]:
    paths = sorted(Path("src/dashcam_ai").rglob("*.py")) + sorted(Path("scripts").glob("*.py"))
    return {str(path): digest(path) for path in paths}


def validate_fixture(fixture: dict[str, Any]) -> None:
    """Published regressions may train; extra windows and their warmup may not."""
    for video in fixture["videos"]:
        training = [r for r in fixture["training_regions"] if r["video"] == video["name"]]
        extra = [c["frame_id"] for c in video["cases"] if c["split"] == "extra"]
        if not extra or not training:
            raise ValueError("each video requires training and extra cases")
        if any(r["split"] != "train" for r in training):
            raise ValueError("training regions must explicitly be train")
        if any(abs(r["frame_id"] - fid) <= 15 for r in training for fid in extra):
            raise ValueError("training overlaps an extra window or its temporal buffer")
        for case in [*video["cases"], *training]:
            points = np.asarray(case.get("polyline", case.get("polygon")), float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError("invalid annotation coordinates")
            if np.any(points < 0) or np.any(points >= [video["width"], video["height"]]):
                raise ValueError("annotation outside frame")


def annotated_class(curve: LaneCurve, regions: list[dict[str, Any]]) -> str | None:
    """Only supervised candidates whose sampled centers mostly occupy a reviewed region."""
    points = [[p.x, p.y] for p in curve.points]
    matches = []
    for region in regions:
        if "polyline" in region:
            line = {"points": [{"x": x, "y": y} for x, y in region["polyline"]]}
            fraction = np.mean([distance_to_curves(p, [line]) <= 10 for p in points])
        else:
            polygon = np.asarray(region["polygon"], np.float32)
            fraction = np.mean(
                [cv2.pointPolygonTest(polygon, tuple(p), False) >= 0 for p in points]
            )
        if fraction >= 0.6:
            matches.append(region["label"])
    return matches[0] if matches and len(set(matches)) == 1 else None


def train(args: argparse.Namespace, fixture: dict[str, Any]) -> None:
    settings = detector_settings(args.config)
    detector = YoloPLaneLineDetector(**settings)
    features, labels, provenance = [], [], []
    hashes = source_hashes()
    inputs = [args.cases, args.config, Path(settings["model"])] + [
        args.samples / (video["name"] + ".mp4") for video in fixture["videos"]
    ]
    input_hashes = {str(path): digest(path) for path in inputs}
    for video in fixture["videos"]:
        path = args.samples / (video["name"] + ".mp4")
        if digest(path) != video["input_sha256"]:
            raise ValueError("training input SHA256 differs from annotations")
        regions = [r for r in fixture["training_regions"] if r["video"] == video["name"]]
        capture = cv2.VideoCapture(str(path))
        try:
            for fid in sorted({r["frame_id"] for r in regions}):
                capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"cannot decode training frame {fid}")
                probability, road = detector._segmentation_probabilities(frame)
                # Proposals before context rejection and before box masking allow
                # supervision of visible paint that the existing boxes suppress.
                mask, _ = detector._candidate_mask(frame, probability, [], road)
                candidates, _, _ = detector._component_curves(
                    mask, probability, road, use_context=True
                )
                selected = [r for r in regions if r["frame_id"] == fid]
                for curve in candidates:
                    label = annotated_class(curve, selected)
                    if label is None:
                        continue
                    features.append(curve_features(frame, curve))
                    labels.append(label)
                    provenance.append(
                        {
                            "video": video["name"],
                            "frame_id": fid,
                            "label": label,
                            "curve": curve.model_dump(mode="json"),
                        }
                    )
        finally:
            capture.release()
    classifier = MarkingClassifier.train(np.asarray(features), labels)
    args.output.mkdir(parents=True, exist_ok=False)
    classifier.save(args.output / "classifier.xml")
    (args.output / "candidates.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest = {
        "complete": source_hashes() == hashes
        and input_hashes == {str(path): digest(path) for path in inputs},
        "source_hashes": hashes,
        "input_hashes": input_hashes,
        "fixture_sha256": digest(args.cases),
        "config_sha256": digest(args.config),
        "yolop_sha256": digest(Path(settings["model"])),
        "class_counts": dict(Counter(labels)),
        "candidates": len(labels),
        "groups": sorted({r["group"] for r in fixture["training_regions"]}),
        "extra_windows_used": False,
        "training_predictions_are_not_quality_evidence": True,
        "classifier_sha256": digest(args.output / "classifier.xml"),
    }
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["complete"]:
        raise ValueError("source changed during training")
    print(json.dumps({"candidates": len(labels), "classes": manifest["class_counts"]}), flush=True)


def windows(frame_ids: list[int], total: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for fid in sorted(set(frame_ids)):
        if fid < 0 or fid >= total:
            raise ValueError("requested frame outside video")
        start, stop = max(0, fid - 15), fid + 1
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def body_scores(mask: np.ndarray[Any, Any], annotation: dict[str, Any]) -> dict[str, float]:
    truth = np.zeros_like(mask)
    cv2.fillPoly(truth, [np.asarray(annotation["polygon"], np.int32)], 255)
    x0, y0, x1, y1 = annotation["audit_box"]
    actual = mask[y0:y1, x0:x1] > 0
    expected = truth[y0:y1, x0:x1] > 0
    intersection = int(np.count_nonzero(actual & expected))
    return {
        "precision": round(intersection / max(int(actual.sum()), 1), 4),
        "recall": round(intersection / max(int(expected.sum()), 1), 4),
        "iou": round(intersection / max(int(np.count_nonzero(actual | expected)), 1), 4),
    }


def score_variant(
    video: dict[str, Any],
    rows: dict[int, Any],
    split: str,
) -> dict[str, Any]:
    cases = dict(video, cases=[c for c in video["cases"] if c["split"] == split])
    records = {fid: row["lane_lines"] for fid, row in rows.items()}
    occlusions = {fid: row["occlusion_runs"] for fid, row in rows.items()}
    results = evaluate_marking_cases(cases, records, records, occlusions)
    for item in results:
        if not item["expected_line"]:
            continue
        points = np.asarray(item["polyline"], float)
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        sampled = np.linspace(0, lengths[-1], max(2, math.ceil(lengths[-1] / 2) + 1))
        xs = np.rint(np.interp(sampled, lengths, points[:, 0])).astype(int)
        ys = np.rint(np.interp(sampled, lengths, points[:, 1])).astype(int)
        positions = ys * video["width"] + xs
        hidden = np.zeros(len(positions), dtype=bool)
        for start, stop in occlusions[item["frame_id"]]:
            hidden |= (positions >= start) & (positions < stop)
        item["masked_visible_fraction"] = round(float(hidden.mean()), 4)
    return {"summary": summarize_markings(results, video["fps"])["after"], "cases": results}


def controls_score(cases: dict[str, Any], records: dict[int, Any]) -> dict[str, Any]:
    results = []
    for case in cases["cases"]:
        curves = records[case["frame_id"]]["lane_lines"]["curves"]
        hits = [distance_to_curves(p, curves) <= cases["tolerance_pixels"] for p in case["points"]]
        results.append(dict(case, passed=all(hit == case["expected_line"] for hit in hits)))
    return {"passed": sum(c["passed"] for c in results), "cases": len(results), "results": results}


def evaluate(args: argparse.Namespace, fixture: dict[str, Any]) -> None:
    import torch
    from ultralytics import YOLO, __version__

    if not args.vehicle_model.is_file():
        raise ValueError("supply downloaded weights explicitly; evaluation never auto-downloads")
    classifier = MarkingClassifier.load(args.classifier)
    training_path = args.classifier.parent / "training-manifest.json"
    training = json.loads(training_path.read_text())
    if (
        not training["complete"]
        or training["extra_windows_used"]
        or training["fixture_sha256"] != digest(args.cases)
        or training["classifier_sha256"] != digest(args.classifier)
    ):
        raise ValueError("classifier training provenance does not match the frozen fixture")
    settings = detector_settings(args.config)
    sources = source_hashes()
    inputs = [
        args.cases,
        args.config,
        args.vehicle_model,
        args.classifier,
        training_path,
        args.classifier.with_suffix(".json"),
        Path(settings["model"]),
    ]
    inputs += [args.samples / (v["name"] + ".mp4") for v in fixture["videos"]]
    inputs += [
        args.boxes / (v["name"] + "_milestone2_yolop_v7") / "frames.jsonl"
        for v in fixture["videos"]
    ]
    inputs += [
        Path("tests/fixtures/lane_stability_control.json"),
        Path("tests/fixtures/lane_stability_review.json"),
    ]
    input_hashes = {str(p): digest(p) for p in inputs}
    for video in fixture["videos"]:
        if digest(args.samples / (video["name"] + ".mp4")) != video["input_sha256"]:
            raise ValueError("evaluation input SHA256 differs from annotations")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "complete": False,
        "source_hashes": sources,
        "input_hashes": input_hashes,
        "device": "cpu",
        "scope": "opt-in temporal-window prototype; not GPU acceptance",
        "vehicle_settings": {"imgsz": 640, "conf": 0.25, "retina_masks": True},
        "warmup_frames": 15,
        "variants": VARIANTS,
        "lane_settings": settings,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "ultralytics": __version__,
        },
        "vehicle_weight_source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    segmenter = YOLO(str(args.vehicle_model))
    timings: dict[str, list[float]] = {key: [] for key in (*VARIANTS, "yolop", "vehicle")}
    reports: dict[str, Any] = {}
    body_results, processed = [], 0
    started = time.monotonic()
    for video in fixture["videos"]:
        name = video["name"]
        boxes = read_boxes(args.boxes / (name + "_milestone2_yolop_v7") / "frames.jsonl")
        control_path = Path(
            "tests/fixtures/lane_stability_"
            + ("control" if name == "sample_1" else "review")
            + ".json"
        )
        controls = json.loads(control_path.read_text())
        if controls["input_sha256"] != video["input_sha256"]:
            raise ValueError("control input SHA256 differs")
        annotations = [a for a in fixture["vehicle_annotations"] if a["video"] == name]
        targets = [c["frame_id"] for c in video["cases"] + controls["cases"] + annotations]
        capture = cv2.VideoCapture(str(args.samples / (name + ".mp4")))
        total = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ranges = windows(targets, total)
        ids = [fid for start, stop in ranges for fid in range(start, stop)]
        if any(fid not in boxes for fid in ids):
            raise ValueError("recorded vehicle boxes missing requested frames")
        rows: dict[str, dict[int, Any]] = {variant: {} for variant in VARIANTS}
        detectors = {
            variant: ExperimentalLaneDetector(
                classifier=classifier if variant in ("classifier", "combined") else None,
                **settings,
            )
            for variant in VARIANTS
        }
        infer = detectors["baseline"]._segmentation_probabilities
        painter = OpenCVAnnotator()
        writer = cv2.VideoWriter(
            str(args.output / (name + "-comparison.mp4")),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video["fps"],
            (video["width"] * 2, video["height"] * 2),
        )
        if not writer.isOpened():
            raise ValueError("cannot open prototype comparison video")
        try:
            for start, stop in ranges:
                for detector in detectors.values():
                    detector.reset()
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                for fid in range(start, stop):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"cannot decode evaluation frame {fid}")
                    tick = time.monotonic()
                    probability, road = infer(frame)
                    timings["yolop"].append(time.monotonic() - tick)
                    tick = time.monotonic()
                    predicted = segmenter.predict(
                        frame,
                        device="cpu",
                        imgsz=640,
                        conf=0.25,
                        retina_masks=True,
                        classes=[0, 1, 2, 3, 5, 7],
                        verbose=False,
                    )[0]
                    instances = []
                    if predicted.masks is not None:
                        for bbox, category, mask in zip(
                            predicted.boxes.xyxy.cpu().numpy(),
                            predicted.boxes.cls.cpu().numpy(),
                            predicted.masks.data.cpu().numpy(),
                            strict=True,
                        ):
                            instances.append(
                                VehicleInstance(
                                    BBox(
                                        x1=float(bbox[0]),
                                        y1=float(bbox[1]),
                                        x2=float(bbox[2]),
                                        y2=float(bbox[3]),
                                    ),
                                    (mask > 0).astype(np.uint8) * 255,
                                    predicted.names[int(category)],
                                )
                            )
                    timings["vehicle"].append(time.monotonic() - tick)
                    baseline = detectors["baseline"]
                    baseline._candidate_mask(frame, probability, boxes[fid], road)
                    fallback = baseline.occlusion_mask.copy()
                    box_masks = []
                    for box in boxes[fid]:
                        baseline._candidate_mask(frame, probability, [box], road)
                        box_masks.append(baseline.occlusion_mask.copy())
                    refined, replaced = pixel_occlusion(boxes[fid], instances, fallback, box_masks)
                    panels, masks = [], {}
                    for variant, detector in detectors.items():
                        detector.predictions = []
                        detector.pixel_mask = (
                            refined if variant in ("pixel_mask", "combined") else None
                        )
                        detector._segmentation_probabilities = lambda _, p=probability, r=road: (
                            p,
                            r,
                        )
                        tick = time.monotonic()
                        result = detector.detect(frame, boxes[fid])
                        timings[variant].append(time.monotonic() - tick)
                        masks[variant] = detector.occlusion_mask.copy()
                        row = {
                            "frame_id": fid,
                            "lane_lines": result.model_dump(mode="json"),
                            "occlusion_runs": mask_runs(detector.occlusion_mask),
                            "predictions": detector.predictions,
                            "replaced_boxes": replaced,
                            "vehicle_instances": len(instances),
                        }
                        rows[variant][fid] = row
                        panel = painter.annotate(frame, [], result, detector.occlusion_mask)
                        cv2.putText(
                            panel, f"{variant} / frame {fid}", (12, 35), 0, 0.9, (255, 255, 255), 2
                        )
                        panels.append(panel)
                    comparison = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
                    writer.write(comparison)
                    if fid in set(targets):
                        cv2.imwrite(str(args.output / f"{name}-{fid}.jpg"), comparison)
                    for annotation in [a for a in annotations if a["frame_id"] == fid]:
                        body_results.append(
                            {
                                "video": name,
                                "frame_id": fid,
                                "split": annotation["split"],
                                "baseline": body_scores(masks["baseline"], annotation),
                                "pixel_mask": body_scores(masks["pixel_mask"], annotation),
                            }
                        )
                    processed += 1
                    if processed % 60 == 0:
                        print(json.dumps({"processed": processed, "video": name}), flush=True)
        finally:
            capture.release()
            writer.release()
            for variant in VARIANTS:
                with (args.output / (name + "-" + variant + ".jsonl")).open("w") as stream:
                    for row in rows[variant].values():
                        stream.write(json.dumps(row) + "\n")
        verify_video(args.output / (name + "-comparison.mp4"), len(ids))
        reports[name] = {
            "frame_ranges": ranges,
            "decoded_frames": len(ids),
            "quality": {
                variant: {
                    split: score_variant(video, rows[variant], split)
                    for split in ("regression", "extra")
                }
                for variant in VARIANTS
            },
            "controls": {variant: controls_score(controls, rows[variant]) for variant in VARIANTS},
        }
    if source_hashes() != sources or {str(p): digest(p) for p in inputs} != input_hashes:
        raise ValueError("source or input changed during evaluation")
    report = {
        "videos": reports,
        "body_masks": body_results,
        "timing_seconds": {
            key: {
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
            }
            for key, values in timings.items()
        },
        "frames": processed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report["promotion_gate_passed"] = all(
        data["quality"]["combined"][split]["summary"]["passed_cases"]
        == data["quality"]["combined"][split]["summary"]["cases"]
        for data in reports.values()
        for split in ("regression", "extra")
    ) and all(
        data["controls"]["combined"]["passed"] == data["controls"]["combined"]["cases"]
        for data in reports.values()
    )
    report["body_gate_passed"] = all(
        row["pixel_mask"]["precision"] >= 0.9
        and row["pixel_mask"]["recall"] >= 0.9
        and row["pixel_mask"]["iou"] >= 0.85
        for row in body_results
    ) and bool(body_results)
    report["promotion_gate_passed"] &= report["body_gate_passed"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest["complete"] = True
    manifest["output_hashes"] = {
        p.name: digest(p)
        for p in args.output.iterdir()
        if p.suffix in (".jsonl", ".mp4") or p.name == "report.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps({"frames": processed, "promotion_gate_passed": report["promotion_gate_passed"]}),
        flush=True,
    )
    if not report["promotion_gate_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/lane_prototype_review.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--boxes", type=Path, default=Path("output"))
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--vehicle-model", type=Path, default=Path("models/yolo26n-seg.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.command == "evaluate" and args.classifier is None:
        parser.error("evaluate requires --classifier")
    cv2.setNumThreads(4)
    if args.command == "evaluate":
        import torch

        torch.set_num_threads(4)
    fixture = json.loads(args.cases.read_text())
    validate_fixture(fixture)
    (train if args.command == "train" else evaluate)(args, fixture)


if __name__ == "__main__":
    main()

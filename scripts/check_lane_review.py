"""Compare visible, manually reviewed lane locations; not dataset accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[int, Any]:
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        frame_id = row["frame_id"]
        if frame_id in records:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        records[frame_id] = row["lane_lines"]
    return records


def distance_to_curves(point: list[float], curves: list[Any]) -> float:
    closest = float("inf")
    for curve in curves:
        for a, b in zip(curve["points"][:-1], curve["points"][1:], strict=True):
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            length_squared = dx * dx + dy * dy
            t = (
                ((point[0] - a["x"]) * dx + (point[1] - a["y"]) * dy) / length_squared
                if length_squared else 0
            )
            t = min(1, max(0, t))
            closest = min(closest, math.hypot(
                point[0] - a["x"] - t * dx, point[1] - a["y"] - t * dy,
            ))
    return closest


def visible_point_distances(
    points: list[list[float]], curves: list[Any],
    width: int, height: int, runs: list[list[int]],
) -> list[float]:
    """Legacy point controls must judge the curve actually visible in the overlay."""
    import cv2
    import numpy as np

    centerline = np.zeros((height, width), np.uint8)
    for curve in curves:
        vertices = np.rint([[p["x"], p["y"]] for p in curve["points"]]).astype(np.int32)
        cv2.polylines(centerline, [vertices], False, 1, 1)
    for start, stop in runs:
        if not 0 <= start < stop <= width * height:
            raise ValueError("invalid occlusion runs")
        centerline.ravel()[start:stop] = 0
    coordinates = np.rint(points).astype(int)
    if np.any(coordinates < 0) or np.any(coordinates[:, 0] >= width) or np.any(
        coordinates[:, 1] >= height
    ):
        raise ValueError("reviewed point outside image")
    if not np.any(centerline):
        return [float("inf")] * len(points)
    distances = cv2.distanceTransform(1 - centerline, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return [float(distances[y, x]) for x, y in coordinates]


def straight_geometry_error(
    polyline: list[list[float]], curves: list[Any], hidden: Any,
) -> float | None:
    """Audit the matched output, so a local bend cannot hide in 90% coverage.

    Only the annotated stroke's longitudinal extent is judged. Other lanes
    and extensions beyond this manually reviewed extent are not ground truth.
    """
    import numpy as np

    if not curves:
        return None
    start, stop = np.asarray(polyline[0]), np.asarray(polyline[-1])
    direction = stop - start
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError("straight review has no length")
    axis = direction / length
    truth = np.linspace(start, stop, 24)
    curve = min(curves, key=lambda c: sum(distance_to_curves(p.tolist(), [c]) for p in truth))
    errors = []
    height, width = hidden.shape
    for first, last in zip(curve["points"][:-1], curve["points"][1:], strict=True):
        a = np.asarray([first["x"], first["y"]])
        b = np.asarray([last["x"], last["y"]])
        samples = np.linspace(a, b, max(2, math.ceil(float(np.linalg.norm(b - a)) / 2) + 1))
        relative = samples - start
        along = relative @ axis
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
        visible = (along >= 0) & (along <= length) & (hidden[ys, xs] == 0)
        perpendicular = np.abs(relative[:, 0] * axis[1] - relative[:, 1] * axis[0])
        errors.extend(perpendicular[visible].tolist())
    return max(errors) if errors else None


def evaluate_marking_cases(
    cases: dict[str, Any], before: dict[int, Any], after: dict[int, Any],
    occlusions: dict[int, Any],
    *, baseline_occlusions: dict[int, Any] | None = None,
) -> list[dict[str, Any]]:
    """Measure visible polyline length and rendered marking overlap, not isolated points."""
    import cv2
    import numpy as np

    width, height = cases["width"], cases["height"]
    results = []
    for case in cases["cases"]:
        frame_id = case["frame_id"]
        item = dict(case)
        for name, records in (("before", before), ("after", after)):
            masks = baseline_occlusions if name == "before" else occlusions
            if masks is None:
                masks = occlusions
            hidden = np.zeros(height * width, np.uint8)
            for start, stop in masks.get(frame_id, []):
                if not 0 <= start < stop <= width * height:
                    raise ValueError("invalid occlusion runs")
                hidden[start:stop] = 1
            hidden = hidden.reshape((height, width))
            if frame_id not in records:
                raise ValueError(f"{name} is missing reviewed frame {frame_id}")
            centerline = np.zeros((height, width), np.uint8)
            rendered = np.zeros_like(centerline)
            for curve in records[frame_id]["curves"]:
                points = np.rint([[p["x"], p["y"]] for p in curve["points"]]).astype(np.int32)
                cv2.polylines(centerline, [points], False, 1, 1)
                cv2.polylines(rendered, [points], False, 255, 5, cv2.LINE_AA)
            centerline[hidden > 0] = 0
            rendered[hidden > 0] = 0
            if case["expected_line"]:
                points = np.asarray(case["polyline"], np.float64)
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                distances = np.r_[0.0, np.cumsum(lengths)]
                if distances[-1] <= 0:
                    raise ValueError("reviewed polyline has no length")
                samples = np.linspace(0, distances[-1], max(2, math.ceil(distances[-1] / 2) + 1))
                xs = np.rint(np.interp(samples, distances, points[:, 0])).astype(int)
                ys = np.rint(np.interp(samples, distances, points[:, 1])).astype(int)
                if np.any(xs < 0) or np.any(xs >= width) or np.any(ys < 0) or np.any(ys >= height):
                    raise ValueError("reviewed polyline outside image")
                # Annotations contain only manually confirmed visible paint.
                # A detector's oversized vehicle mask must count as a missed
                # visible line, rather than erase ground truth from the denominator.
                visible = np.ones(len(xs), dtype=bool)
                distance_map = cv2.distanceTransform(
                    1 - centerline, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                )
                hits = distance_map[ys[visible], xs[visible]] <= cases["tolerance_pixels"]
                coverage = float(hits.mean())
                item[name] = {
                    "coverage": round(coverage, 4), "visible_samples": int(visible.sum()),
                    "covered_samples": int(hits.sum()),
                    "passed": coverage >= case.get("minimum_coverage", 0.9),
                }
                if case.get("check_straight_geometry"):
                    error = straight_geometry_error(case["polyline"], records[frame_id]["curves"],
                                                    hidden)
                    item[name]["maximum_geometry_error_pixels"] = (
                        round(error, 2) if error is not None else None
                    )
                    item[name]["passed"] &= (
                        error is not None and error <= cases["tolerance_pixels"]
                    )
            else:
                region = np.zeros_like(centerline)
                cv2.fillPoly(region, [np.asarray(case["polygon"], np.int32)], 1)
                overlap = int(np.count_nonzero((rendered > 0) & (region > 0)))
                item[name] = {"wrong_pixels": overlap, "passed": overlap == 0}
        results.append(item)
    return results


def summarize_markings(results: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in ("before", "after"):
        positive = [r[name] for r in results if r["expected_line"]]
        visible = sum(r["visible_samples"] for r in positive)
        covered = sum(r["covered_samples"] for r in positive)
        negatives = {}
        for category in sorted({r["category"] for r in results if not r["expected_line"]}):
            items = [r for r in results if not r["expected_line"] and r["category"] == category]
            negatives[category] = {
                "tested_frames": len({r["frame_id"] for r in items}),
                "wrong_frames": len({r["frame_id"] for r in items if not r[name]["passed"]}),
            }
        longest = 0
        for group in {r["group"] for r in results if r["expected_line"]}:
            items = sorted([r for r in results if r["expected_line"] and r["group"] == group],
                           key=lambda r: r["frame_id"])
            run, previous = 0, -2
            for item in items:
                run = (run + 1 if item["frame_id"] == previous + 1 else 1) if (
                    not item[name]["passed"]
                ) else 0
                longest = max(longest, run)
                previous = item["frame_id"]
        summary[name] = {
            "visible_line_coverage": round(covered / visible, 4) if visible else None,
            "negative_regions": negatives,
            "longest_below_target_run_frames": longest,
            "longest_below_target_run_seconds": round(longest / fps, 3),
            "passed_cases": sum(r[name]["passed"] for r in results),
            "cases": len(results),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "development", "held_out"), default="all")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if "videos" in cases:
        matches = [video for video in cases["videos"] if video["input_sha256"] == digest]
        if len(matches) != 1:
            raise ValueError("input is not uniquely identified in review annotations")
        cases = matches[0]
    if args.split != "all":
        cases["cases"] = [case for case in cases["cases"] if case["split"] == args.split]
    if not cases["cases"]:
        raise ValueError("no reviewed cases selected")
    if digest != cases["input_sha256"]:
        raise ValueError("input does not match the reviewed video")
    before, after = load_records(args.baseline), load_records(args.records)
    rows = [json.loads(line) for line in args.records.read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in args.baseline.read_text().splitlines()]
    has_masks = any("occlusion_runs" in row for row in rows + baseline_rows)
    if has_masks and any("occlusion_runs" not in row for row in rows + baseline_rows):
        raise ValueError("incomplete effective occlusion masks")
    occlusions = {row["frame_id"]: row["occlusion_runs"] for row in rows} if has_masks else {}
    baseline_occlusions = {
        row["frame_id"]: row["occlusion_runs"] for row in baseline_rows
    } if has_masks else {}
    if any("polyline" in case or "polygon" in case for case in cases["cases"]):
        if not has_masks:
            raise ValueError("marking review requires recorded effective occlusion masks")
        results = evaluate_marking_cases(
            cases, before, after, occlusions, baseline_occlusions=baseline_occlusions,
        )
    else:
        results = []
    for case in ([] if results else cases["cases"]):
        frame_id = case["frame_id"]
        item = dict(case)
        for name, records in (("before", before), ("after", after)):
            if frame_id not in records:
                raise ValueError(f"{name} is missing reviewed frame {frame_id}")
            distances = (
                visible_point_distances(
                    case["points"], records[frame_id]["curves"], cases["width"], cases["height"],
                    (baseline_occlusions if name == "before" else occlusions)[frame_id],
                ) if has_masks else [
                    distance_to_curves(p, records[frame_id]["curves"]) for p in case["points"]
                ]
            )
            hits = [d <= cases["tolerance_pixels"] for d in distances]
            item[name] = {
                "distance_pixels": [round(d, 2) if math.isfinite(d) else None for d in distances],
                "hits": sum(hits),
                "passed": all(hit == case["expected_line"] for hit in hits),
            }
        results.append(item)
    report = {
        "input_sha256": digest,
        "scope": cases["scope"],
        "tolerance_pixels": cases["tolerance_pixels"],
        "landmarks": results,
        "after_all_cases_passed": all(item["after"]["passed"] for item in results),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.cases, args.baseline, args.records)
        },
    }
    if "polyline" in cases["cases"][0] or "polygon" in cases["cases"][0]:
        report["summary"] = summarize_markings(results, cases["fps"])
        report["split"] = args.split
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "passed": sum(r["after"]["passed"] for r in results)}))
    if not report["after_all_cases_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

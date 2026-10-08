"""Audit frozen pixel masks with complete manual vehicle unions inside reviewed regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from replay_lane_review import digest


def score_body(
    mask: np.ndarray[Any, Any],
    polygons: list[Any],
    audit_box: list[int],
) -> dict[str, float]:
    if mask.ndim != 2 or not polygons:
        raise ValueError("body audit needs a two-dimensional mask and reviewed polygons")
    height, width = mask.shape
    x0, y0, x1, y1 = audit_box
    if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError("audit region outside image")
    truth = np.zeros_like(mask)
    for polygon in polygons:
        points = np.asarray(polygon, float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points >= [width, height])
        ):
            raise ValueError("invalid vehicle polygon")
        cv2.fillPoly(truth, [points.astype(np.int32)], 1)
    expected, actual = truth[y0:y1, x0:x1] > 0, mask[y0:y1, x0:x1] > 0
    if not expected.any():
        raise ValueError("audit region contains no reviewed body")
    intersection = int(np.count_nonzero(expected & actual))
    return {
        "precision": round(intersection / max(int(actual.sum()), 1), 4),
        "recall": round(intersection / max(int(expected.sum()), 1), 4),
        "iou": round(intersection / max(int(np.count_nonzero(expected | actual)), 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/lane_prototype_body_review.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    manifest_path = args.records_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"]:
        raise ValueError("inference manifest is incomplete")
    fixture = json.loads(args.cases.read_text())
    hashes = {
        str(args.cases): digest(args.cases),
        str(manifest_path): digest(manifest_path),
        str(Path(__file__)): digest(Path(__file__)),
    }
    measurements = []
    for video in fixture["videos"]:
        if manifest["input_hashes"]["samples/" + video["name"] + ".mp4"] != video["input_sha256"]:
            raise ValueError("manual body truth does not match the recorded input")
        cases = [c for c in fixture["cases"] if c["video"] == video["name"]]
        rows: dict[str, dict[int, Any]] = {}
        for variant in ("baseline", "pixel_mask"):
            path = args.records_dir / (video["name"] + "-" + variant + ".jsonl")
            if digest(path) != manifest["output_hashes"][path.name]:
                raise ValueError("recorded mask output hash mismatch")
            hashes[str(path)] = digest(path)
            rows[variant] = {}
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["frame_id"] in rows[variant]:
                    raise ValueError("duplicate mask frame")
                rows[variant][row["frame_id"]] = row
        for case in cases:
            item = {key: case[key] for key in ("video", "frame_id", "split")}
            for variant in rows:
                mask = np.zeros(video["height"] * video["width"], np.uint8)
                for start, stop in rows[variant][case["frame_id"]]["occlusion_runs"]:
                    if not 0 <= start < stop <= mask.size:
                        raise ValueError("invalid recorded mask runs")
                    mask[start:stop] = 255
                item[variant] = score_body(
                    mask.reshape((video["height"], video["width"])),
                    case["polygons"],
                    case["audit_box"],
                )
            measurements.append(item)
    if hashes != {path: digest(Path(path)) for path in hashes}:
        raise ValueError("body audit source or evidence changed")
    passed = bool(measurements) and all(
        m["pixel_mask"]["precision"] >= 0.9
        and m["pixel_mask"]["recall"] >= 0.9
        and m["pixel_mask"]["iou"] >= 0.85
        for m in measurements
    )
    report = {
        "complete": True,
        "scope": fixture["scope"],
        "measurements": measurements,
        "body_gate_passed": passed,
        "input_hashes": hashes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(measurements), "body_gate_passed": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

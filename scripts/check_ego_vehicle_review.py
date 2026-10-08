"""Audit frozen full-video ego exclusion and original-image neighbor controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dashcam_ai.config.models import load_config
from dashcam_ai.detection.ego_mask import EgoVehicleMask


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def iou(first: list[float], second: list[float]) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0


def box_coordinates(obj: dict[str, Any]) -> list[float]:
    return [obj["bbox"][key] for key in ("x1", "y1", "x2", "y2")]


def ego_like(box: list[float], mask: EgoVehicleMask, shape: tuple[int, int]) -> bool:
    """Diagnostic for this camera's large bottom-left self boxes, not an ID filter.

    Requires frame-edge anchoring, the known body height band and >=45% body overlap.
    It is a review aid; absence alone does not prove whole-video vehicle precision.
    """
    height, width = shape
    return (
        box[0] <= width * 0.02
        and box[3] >= height * 0.96
        and height * 0.48 <= box[1] <= height * 0.62
        and mask.excludes(box, shape)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path("tests/fixtures/ego_vehicle_review.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("incomplete replay")
    for group in ("source_hashes", "input_hashes"):
        for path, expected in manifest[group].items():
            if digest(Path(path)) != expected:
                raise ValueError(f"{group} mismatch: {path}")
    for path, expected in manifest["output_hashes"].items():
        if digest(args.run_dir / path) != expected:
            raise ValueError(f"output mismatch: {path}")
    if digest(args.cases) != manifest["source_hashes"][str(args.cases)]:
        raise ValueError("fixture mismatch")
    config_path = args.run_dir / "tracking.yaml"
    if not config_path.exists():
        config_path = args.run_dir / "cpu-tracking.yaml"
    config = load_config(config_path)
    diagnostic = EgoVehicleMask(config.detection.ego_vehicle_polygon, 0.45)
    fixture = json.loads(args.cases.read_text())
    results = []
    for video in fixture["videos"]:
        name = video["name"]
        rows = [json.loads(line) for line in (args.run_dir / name / "frames.jsonl").open()]
        metadata = json.loads((args.run_dir / name / "metadata.json").read_text())
        shape = (metadata["height"], metadata["width"])
        if [r["frame_id"] for r in rows] != list(range(metadata["frame_count"])):
            raise ValueError("missing or duplicate frames")
        if metadata["runtime"]["ego_vehicle_polygon"] != [list(p) for p in diagnostic.polygon]:
            raise ValueError("runtime polygon mismatch")
        failures = []
        for row in rows:
            for obj in row["objects"]:
                if ego_like(box_coordinates(obj), diagnostic, shape):
                    failures.append({"frame_id": row["frame_id"], "track_id": obj["track_id"]})
        controls = []
        for control in video["neighbor_controls"]:
            scores = [
                iou(control["bbox"], box_coordinates(o))
                for o in rows[control["frame_id"]]["objects"]
            ]
            score = max(scores, default=0)
            controls.append({**control, "best_iou": score, "passed": score >= 0.5})
        results.append(
            {
                "video": name,
                "frame_count": len(rows),
                "ego_like_boxes": failures,
                "neighbor_controls": controls,
            }
        )
    report = {
        "complete": True,
        "scope": (
            "Fixed-camera self-box diagnostic plus 11 reviewed neighbors; "
            "not whole-road recall or GPU acceptance."
        ),
        "passed": all(
            not r["ego_like_boxes"] and all(c["passed"] for c in r["neighbor_controls"])
            for r in results
        ),
        "results": results,
        "evidence_hashes": {
            "checker": digest(Path(__file__)),
            "fixture": digest(args.cases),
            "manifest": digest(args.run_dir / "manifest.json"),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

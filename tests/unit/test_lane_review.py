"""Regression review must measure full visible lines and reject incomplete evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_spec = importlib.util.spec_from_file_location(
    "lane_review", Path(__file__).resolve().parents[2] / "scripts/check_lane_review.py",
)
assert _spec is not None and _spec.loader is not None
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def _records(end: int = 90) -> dict[int, Any]:
    return {1: {"curves": [{"points": [{"x": 10, "y": 50}, {"x": end, "y": 50}]}]}}


def _cases() -> dict[str, Any]:
    return {
        "width": 100, "height": 100, "tolerance_pixels": 2,
        "cases": [{"frame_id": 1, "expected_line": True, "polyline": [[10, 50], [90, 50]]}],
    }


def test_short_line_does_not_pass_full_visible_line_review() -> None:
    result = review.evaluate_marking_cases(_cases(), _records(), _records(30), {1: []})[0]
    assert result["before"]["passed"]
    assert not result["after"]["passed"]
    assert result["after"]["coverage"] < 0.35


def test_small_wrong_bend_cannot_hide_inside_high_average_coverage() -> None:
    cases = _cases()
    cases["cases"][0]["check_straight_geometry"] = True
    bent = {1: {"curves": [{"points": [
        {"x": 10, "y": 50}, {"x": 73, "y": 50}, {"x": 75, "y": 56},
        {"x": 77, "y": 50}, {"x": 90, "y": 50},
    ]}]}}

    result = review.evaluate_marking_cases(cases, _records(), bent, {1: []})[0]

    assert result["before"]["passed"]
    assert result["after"]["coverage"] >= 0.9
    assert result["after"]["maximum_geometry_error_pixels"] > 2
    assert not result["after"]["passed"]


def test_detector_mask_does_not_remove_manually_visible_line_from_denominator() -> None:
    result = review.evaluate_marking_cases(
        _cases(), _records(), _records(30), {1: [[5032, 5100]]},
    )[0]
    assert result["after"]["coverage"] < 0.35
    assert not result["after"]["passed"]


def test_invalid_occlusion_cannot_pass() -> None:
    with pytest.raises(ValueError, match="occlusion|occluded"):
        review.evaluate_marking_cases(_cases(), _records(), _records(), {1: [[0, 10001]]})


def test_masked_visible_ground_truth_is_a_quality_failure() -> None:
    result = review.evaluate_marking_cases(
        _cases(), _records(), _records(), {1: [[5010, 5091]]},
    )[0]
    assert result["after"]["coverage"] == 0
    assert not result["after"]["passed"]


def test_paired_review_uses_each_versions_own_mask() -> None:
    result = review.evaluate_marking_cases(
        _cases(), _records(), _records(), {1: []},
        baseline_occlusions={1: [[5000, 5100]]},
    )[0]
    assert result["before"]["coverage"] == 0
    assert not result["before"]["passed"]
    assert result["after"]["coverage"] == 1
    assert result["after"]["passed"]


def test_legacy_visible_point_cannot_pass_when_detector_mask_hides_the_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"synthetic visible road paint")
    cases = _cases()
    cases.update({
        "input_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "scope": "unit visible-point review", "fps": 30,
        "cases": [{"frame_id": 1, "expected_line": True, "points": [[50, 50]]}],
    })
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps(cases))
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "frame_id": 1, "lane_lines": _records()[1], "occlusion_runs": [[5000, 5100]],
    }) + "\n")
    output = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", [
        "review", "--input", str(video), "--cases", str(fixture),
        "--baseline", str(records), "--records", str(records), "--output", str(output),
    ])

    with pytest.raises(SystemExit) as result:
        review.main()

    assert result.value.code == 1
    assert not json.loads(output.read_text())["after_all_cases_passed"]


def test_negative_region_counts_rendered_white_stroke_outside_its_centerline() -> None:
    cases = _cases()
    cases["cases"] = [{
        "frame_id": 1, "expected_line": False,
        "polygon": [[40, 52], [60, 52], [60, 54], [40, 54]],
    }]
    result = review.evaluate_marking_cases(cases, _records(), {1: {"curves": []}}, {1: []})[0]
    assert result["before"]["wrong_pixels"] > 0
    assert not result["before"]["passed"]
    assert result["after"]["passed"]


def test_missing_frame_is_an_error_instead_of_a_successful_blank_result() -> None:
    with pytest.raises(ValueError, match="missing reviewed frame"):
        review.evaluate_marking_cases(_cases(), _records(), {}, {1: []})


def test_duplicate_record_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    row = json.dumps({"frame_id": 1, "lane_lines": {"curves": []}})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        review.load_records(path)


@pytest.mark.parametrize("mismatched_input", [False, True])
def test_cli_rejects_quality_failure_or_wrong_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched_input: bool,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"reviewed-video")
    cases = _cases()
    cases.update({
        "input_sha256": (
            "wrong" if mismatched_input else hashlib.sha256(video.read_bytes()).hexdigest()
        ),
        "scope": "unit review", "fps": 30,
    })
    fixture = tmp_path / "cases.json"
    fixture.write_text(json.dumps(cases))
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({
        "frame_id": 1, "lane_lines": {"curves": []}, "occlusion_runs": [],
    }) + "\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "review", "--input", str(video), "--cases", str(fixture),
        "--baseline", str(records), "--records", str(records), "--output", str(output),
    ])
    if mismatched_input:
        with pytest.raises(ValueError, match="does not match"):
            review.main()
        assert not output.exists()
    else:
        cases["cases"][0].update({"group": "left-lane", "category": "left_lane"})
        fixture.write_text(json.dumps(cases))
        with pytest.raises(SystemExit) as result:
            review.main()
        assert result.value.code == 1
        assert not json.loads(output.read_text())["after_all_cases_passed"]

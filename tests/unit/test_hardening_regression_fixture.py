from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_hardening_labels_are_video_relative_and_cover_key_scenarios() -> None:
    fixture = Path("tests/fixtures/milestone2_hardening_labels.json")
    payload: dict[str, Any] = json.loads(fixture.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = payload["cases"]

    assert payload["schema_version"] == 1
    assert payload["time_basis"] == "video-relative"
    assert all(case["start"] < case["end"] for case in cases)
    assert all(case["track_ids"] for case in cases)
    assert not any("path" in key for key in payload)

    by_track = {
        track_id: case
        for case in cases
        for track_id in case["track_ids"]
        if track_id not in {7015, 10348}
    }
    assert by_track[7769]["expected"] == ["leaving_ego"]
    assert by_track[13021]["expected"] == ["no_event"]
    assert by_track[3864]["scene"] == ["parked"]
    assert by_track[7683]["duplicate_tracks"] is True

    track_7015 = [case for case in cases if 7015 in case["track_ids"]]
    assert [case["expected"] for case in track_7015] == [
        ["entering_ego", "cut_in"],
        ["leaving_ego"],
    ]

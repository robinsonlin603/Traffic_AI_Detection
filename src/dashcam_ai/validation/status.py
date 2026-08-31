"""Freshness and multi-platform milestone status."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dashcam_ai.validation.records import MILESTONE_PLATFORMS, ResultStatus, ValidationRecord
from dashcam_ai.validation.render import load_report, report_paths
from dashcam_ai.validation.runner import _run_git


@dataclass(frozen=True)
class PlatformStatus:
    platform: str
    status: str
    reason: str


def inspect_report(path: Path, root: Path) -> tuple[ValidationRecord, bool]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    validation_root = resolved_root / "validation"
    if not resolved_path.is_relative_to(validation_root):
        raise ValueError("report must be inside the repository validation directory")
    record = load_report(resolved_path)
    current_commit = _run_git(root, "rev-parse", "HEAD")
    return record, record.source_commit == current_commit


def milestone_status(root: Path, milestone: str) -> tuple[list[PlatformStatus], str]:
    required = MILESTONE_PLATFORMS.get(milestone)
    if required is None:
        raise ValueError(f"no platform policy for {milestone}")
    current_commit = _run_git(root, "rev-parse", "HEAD")
    statuses: list[PlatformStatus] = []
    for platform_id in sorted(required):
        json_path, _ = report_paths(root, milestone, platform_id)
        if not json_path.exists():
            statuses.append(PlatformStatus(platform_id, "missing", "report does not exist"))
            continue
        record = load_report(json_path)
        if record.source_commit != current_commit:
            statuses.append(PlatformStatus(platform_id, "stale", "source commit does not match"))
        else:
            statuses.append(
                PlatformStatus(platform_id, record.verdict.value, "; ".join(record.reasons))
            )
    overall = (
        ResultStatus.PASSED.value
        if all(item.status == ResultStatus.PASSED.value for item in statuses)
        else ResultStatus.BLOCKED.value
    )
    return statuses, overall

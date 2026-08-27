"""Atomic JSON and Markdown serialization for validation records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dashcam_ai.validation.records import ValidationRecord


def report_paths(root: Path, milestone: str, platform_id: str) -> tuple[Path, Path]:
    safe_milestone = milestone if milestone.startswith("milestone-") else f"milestone-{milestone}"
    if safe_milestone != "milestone-2":
        raise ValueError(f"unsupported milestone: {safe_milestone}")
    if platform_id not in {"macos-mps", "windows-cuda", "cpu"}:
        raise ValueError(f"unsupported platform: {platform_id}")
    directory = root / "validation" / safe_milestone
    return directory / f"{platform_id}.json", directory / f"{platform_id}.md"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def render_markdown(record: ValidationRecord) -> str:
    gates = "\n".join(f"- `{gate.name}`: **{gate.status.value}**" for gate in record.gates)
    reasons = "\n".join(f"- {reason}" for reason in record.reasons) or "- None"
    return (
        f"# {record.milestone} — {record.platform}\n\n"
        f"- Tested source commit: `{record.source_commit}`\n"
        f"- Worktree clean: **{'yes' if not record.worktree_dirty else 'no'}**\n"
        f"- Accelerator: `{record.environment.accelerator}` "
        f"({'available' if record.environment.accelerator_available else 'unavailable'})\n"
        f"- Accelerator name: `{record.environment.accelerator_name or 'unknown'}`\n"
        f"- Verdict: **{record.verdict.value}**\n\n"
        f"## Gates\n\n{gates}\n\n"
        f"## Reasons\n\n{reasons}\n\n"
        "This result applies only to the named platform and exact source commit. "
        "Git history retains earlier runs; rerun validation after source changes.\n"
    )


def write_report(root: Path, record: ValidationRecord) -> tuple[Path, Path]:
    json_path, markdown_path = report_paths(root, record.milestone, record.platform)
    payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write(json_path, payload)
    _atomic_write(markdown_path, render_markdown(record))
    return json_path, markdown_path


def load_report(path: Path) -> ValidationRecord:
    return ValidationRecord.model_validate_json(path.read_text(encoding="utf-8"))

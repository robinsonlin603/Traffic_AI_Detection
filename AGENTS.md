# Project Agent Workflow

These instructions apply to every agent working anywhere in this repository.

## Shared project context

Git-tracked plans and validation records are the durable context shared between macOS and
Windows. Conversation history is not evidence of project state.

Before changing code, inspect the current branch, commit, worktree status, the active milestone
documents under `docs/`, and the latest reports under `validation/`. Compare every report's
`source_commit` with the commit being evaluated and describe each required platform as `passed`,
`failed`, `blocked`, `stale`, or `missing`.

## Platform validation

Platform-specific validation is symmetric and non-transferable. A passing macOS MPS, Windows
CUDA, or CPU result validates only the recorded platform and exact source commit. Never infer
Windows CUDA success from macOS MPS, macOS MPS success from Windows CUDA, or either GPU platform
from CPU. Platform-independent pytest, Ruff, and Mypy results may be useful for the exact recorded
source commit, but they do not prove another platform's accelerator behavior.

A report is authoritative only when its required gates pass, its requested accelerator was
observed, its `source_commit` matches the code being judged, and its worktree was clean. A report
for an older commit is stale. A dirty, interrupted, malformed, or incomplete run is blocked or
failed and must not complete a milestone.

After changing code, run focused tests plus `pytest`, `ruff check .`, and `mypy src` whenever
practical. Run validation for the current machine, update only that platform's files under
`validation/`, and list platforms that must be rerun. Never fabricate or edit another platform's
machine-generated result.

## Git and data safety

Validation commands generate evidence but do not authorize commits or pushes. Review changes
before publishing them. Do not commit videos, model weights, large logs, credentials, environment
variables, usernames, or private home-directory paths. Store hashes and compact measurements for
large local inputs and outputs. Git history is the validation history; stable platform filenames
are replaced on each valid run.

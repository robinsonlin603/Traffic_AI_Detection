# Build durable cross-platform milestone validation records

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository includes the governing ExecPlan specification at
`codex-portable-settings/PLANS.md`. This document must be maintained in accordance with that
file.

## Purpose / Big Picture

Codex conversations on the Linux and macOS computers do not share conversational context.
After this change, either computer can instead write a small, structured validation record to
Git. The other computer can pull the repository, read exactly which source commit was tested,
see the operating system and accelerator used, inspect every required test gate, and decide
whether the current milestone passed, failed, or remains blocked.

The user will run one cross-platform Python command after checking out the source commit that
needs validation. The command will run the repository's automated gates, collect non-secret
runtime facts, and replace the latest record for that platform under `validation/`. The user can
review the record and commit it normally. Large videos, model weights, and generated annotated
videos remain outside Git; the record identifies relevant local inputs and outputs with hashes
and compact metrics where available.

The completed behavior is visible when a Linux CUDA run produces
`validation/milestone-2/linux-cuda.json` and a readable companion Markdown file, then a Mac
can pull those files and run a report inspection command that clearly states the tested commit,
whether it matches the current checkout, and the milestone verdict.

## Progress

- [x] (2026-08-27 06:12Z) Inspected the repository's milestone documents, test configuration,
  runtime metadata support, Git state, and ignore rules.
- [x] (2026-08-27 06:12Z) Authored this repository-specific ExecPlan for approval.
- [x] (2026-08-27 06:28Z) Added the root `AGENTS.md` shared workflow with symmetric,
  non-transferable platform evidence rules.
- [x] (2026-08-27 06:28Z) Defined validation record models and deterministic milestone verdict
  rules.
- [x] (2026-08-27 06:29Z) Implemented cross-platform generation, JSON/Markdown rendering, report
  inspection, and milestone aggregation commands.
- [x] (2026-08-27 06:30Z) Added unit and integration tests for passing, failing, blocked, stale,
  missing-tool, bounded-output, unsafe-path, and repeated-write cases.
- [x] (2026-08-27 06:31Z) Added the validation directory guidance without fabricating CUDA
  CUDA evidence; a formal Mac report remains intentionally deferred until a clean commit can be
  tested.
- [x] (2026-08-27 06:31Z) Documented the Linux-to-Mac Git handoff and milestone review workflow.
- [x] (2026-08-27 06:32Z) Verified 89 pytest tests, Ruff, and strict Mypy; exercised milestone
  aggregation and observed the expected blocked result while both platform reports are missing.

## Surprises & Discoveries

- Observation: The repository already records device and dependency details in each analysis
  output's `metadata.json`, but those outputs are ignored by Git through `output/*`.
  Evidence: `src/dashcam_ai/runtime/metadata.py` supplies runtime metadata and `.gitignore`
  excludes the generated output tree except for `output/.gitkeep`.

- Observation: Milestone 2 is documented as complete for automated regression and Apple MPS,
  while NVIDIA RTX hardware validation is explicitly still missing.
  Evidence: `docs/EXEC_PLAN.md` and `README.md` say CUDA configuration is tested automatically
  but the target NVIDIA hardware has not run the Milestone 2 acceptance workflow.

- Observation: Platform identity must include the operating system as well as the accelerator.
  Evidence: A CUDA-capable Windows host could otherwise satisfy accelerator-only checks while
  incorrectly claiming `linux-cuda`; verdict rules now require Linux for `linux-cuda` and
  Darwin for `macos-mps`.

## Decision Log

- Decision: Store the latest record in a separate file for each milestone and platform, and use
  Git history as the historical archive.
  Rationale: Stable filenames are easy for both people and Codex to find, keep repository growth
  bounded, and avoid macOS-versus-Linux merge conflicts because each platform writes only its own
  file.
  Date/Author: 2026-08-27 / Codex

- Decision: Record the tested source commit explicitly and report whether it matches the reader's
  current checkout.
  Rationale: The commit containing a generated report is normally newer than the code commit that
  was tested. Without a separate `source_commit`, a passing report could be incorrectly applied
  to later untested changes.
  Date/Author: 2026-08-27 / Codex

- Decision: Generate both canonical JSON and derived Markdown, with JSON controlling the verdict.
  Rationale: JSON is reliable for automated or agent-based decisions, while Markdown lets the user
  inspect the same evidence without tooling. The Markdown file must be regenerated from JSON so it
  cannot silently disagree.
  Date/Author: 2026-08-27 / Codex

- Decision: Do not make the validation command commit or push automatically.
  Rationale: Git publication is an external state change and may include credentials, branch
  policy, or unrelated local changes. Generation remains safe and reviewable; the documented
  workflow uses explicit `git add`, `git commit`, and `git push` steps after inspection.
  Date/Author: 2026-08-27 / Codex

- Decision: Keep large binary evidence out of Git and store hashes plus compact measurements.
  Rationale: Videos and model weights are already ignored and would make routine synchronization
  slow. A SHA-256 digest identifies the exact input or model when the same private file exists on
  both machines.
  Date/Author: 2026-08-27 / Codex

- Decision: Put stable operating rules in the repository-root `AGENTS.md`, while keeping dynamic
  evidence in `validation/`.
  Rationale: Both computers receive the same agent instructions through Git, but frequently
  changing results do not create conflicts in the instruction file. Platform-specific evidence
  remains symmetric and non-transferable in both directions.
  Date/Author: 2026-08-27 / Codex

- Decision: Milestone 2 requires current passing reports from both `macos-mps` and
  `linux-cuda`; the CPU report is optional evidence.
  Rationale: The project explicitly targets Apple MPS and NVIDIA CUDA. Requiring both prevents
  either machine from silently standing in for the other, while CPU remains useful for core
  diagnostics without claiming GPU acceptance.
  Date/Author: 2026-08-27 / Codex

## Outcomes & Retrospective

The shared workflow, report schema, gate runner, rendering, inspection, milestone aggregation,
tests, and operating documentation are implemented. The final regression passed with 89 pytest
tests, clean Ruff output, and strict Mypy success across 54 source files. Milestone aggregation
correctly reports `blocked` because neither platform has a report for the current uncommitted
implementation. No Linux CUDA result has been created or marked as passed; that remains valid
only after the command runs on a clean Linux checkout with CUDA selected. After these changes
are committed, each target machine must generate its own report for that source commit.

## Context and Orientation

This is a Python 3.12 project. `pyproject.toml` defines the package, the `pytest` test suite, Ruff
lint rules, and strict Mypy checks. The production package is under `src/dashcam_ai/`, while unit
and integration tests are under `tests/`. `docs/EXEC_PLAN.md` is the existing Milestone 2 delivery
plan; it requires `pytest`, `ruff check .`, and `mypy src` as final automated gates. `README.md`
contains the user workflow and current Mac-versus-NVIDIA acceptance status.

The repository-root `AGENTS.md` is the stable instruction layer shared by both computers. It tells
agents to read milestone and validation state before changes, forbids cross-platform inference in
both directions, and requires each machine to update only its own evidence. Dynamic results never
belong in `AGENTS.md`; they remain under `validation/`.

A validation record is a small JSON document containing evidence about one run. A gate is one
required command such as `pytest`; its record contains the command, start and end times, duration,
exit code, and a short bounded output excerpt. A verdict is `passed`, `failed`, or `blocked`.
`passed` means every required gate completed successfully and every requested hardware condition
was observed. `failed` means a gate ran and returned an unsuccessful result. `blocked` means the
run could not make a valid judgment, for example because Ruff is not installed or CUDA was
requested but unavailable. A stale record is one whose `source_commit` differs from the Git commit
currently being reviewed; it must never prove that the current checkout passed.

`src/dashcam_ai/runtime/device.py` already resolves CPU, Apple MPS, and NVIDIA CUDA availability.
`src/dashcam_ai/runtime/metadata.py` already collects Python and selected dependency versions plus
model hashes for video analysis. The new validation implementation should reuse concepts from
these modules but must not require a model, network access, or GPU merely to run the core automated
gates. A Linux CUDA acceptance record has stronger requirements: it must confirm that PyTorch
reports CUDA available, record the GPU name, and, when a real-video acceptance run is requested,
record the analysis metadata and compact event counts.

The new tracked directory is `validation/`. Each platform owns its filename, such as
`validation/milestone-2/linux-cuda.json` or
`validation/milestone-2/macos-mps.json`. Re-running the command for one platform replaces only
that platform's JSON and Markdown pair. Previous results remain recoverable through Git history.

## Plan of Work

First, add a focused module at `src/dashcam_ai/validation/records.py`. Define Pydantic models for
the schema and pure functions that calculate a verdict from gate results, requested platform
requirements, and observed runtime facts. Include a numeric schema version, milestone identifier,
platform identifier, UTC timestamps, `source_commit`, dirty-worktree flag, environment facts,
gate results, optional acceptance-run evidence, overall verdict, and reasons. Reject unknown or
malformed enum values. Never collect usernames, home paths, environment variables, repository
credentials, or full unbounded terminal output.

Add the root `AGENTS.md` before implementing the reporting workflow so future macOS and Linux
agents share the same rules. It must state that platform-specific evidence is symmetric and
non-transferable, describe stale and dirty evidence, and distinguish stable instructions from
dynamic records.

Next, add `src/dashcam_ai/validation/runner.py`. Use `subprocess` with argument arrays and the
current Python interpreter rather than shell strings. Run `pytest`, `ruff check .`, and `mypy src`
from the repository root, continuing after an ordinary failing exit code so that one record shows
all gate outcomes. Treat a missing executable or an interrupted command as blocked evidence. Keep
only a bounded tail or summary of stdout and stderr. Discover the source commit using a read-only
Git command and record whether tracked or untracked changes were present when testing. A dirty run
may describe useful diagnostics but cannot receive an authoritative `passed` verdict because its
exact source is not reproducible from the recorded commit.

Add `src/dashcam_ai/validation/render.py` to serialize canonical JSON with stable indentation and
derive a concise Markdown summary from the same validated model. Writes should be atomic: write a
temporary sibling and replace the destination only after serialization succeeds. Platform IDs
must come from a small allowlist so user input cannot escape the intended `validation/<milestone>/`
directory.

Expose the workflow through the existing Typer application in `src/dashcam_ai/cli.py` as a
`validate` command. Its minimum interface is `dashcam-ai validate --milestone 2 --platform
linux-cuda`. Support `macos-mps`, `linux-cuda`, and a generic `cpu` platform initially. Add a
`--check-report PATH` inspection mode, or a separate `validation-status` subcommand if that better
matches the existing CLI structure. Inspection validates the JSON, compares `source_commit` to
the current checkout, and prints a non-zero result for failed, blocked, malformed, or stale
reports. Do not add a new third-party runtime dependency.

Tests should be added under `tests/unit/test_validation_records.py` and
`tests/integration/test_validation_cli.py`. Unit tests must cover deterministic passed, failed,
blocked, dirty, and stale judgments. Integration tests must replace subprocess execution with
small controlled commands or injected fakes; they must not depend on CUDA, MPS, model weights,
network access, or the developer's installed global tools. Verify output bounding, safe platform
paths, stable JSON-to-Markdown agreement, and atomic replacement behavior. Existing CLI tests in
`tests/unit/test_cli.py` should be extended only where the command registration or output contract
requires it.

Add `validation/README.md` describing record ownership and the rule that reports are evidence, not
source code changes. Add initial records only from environments actually exercised during this
implementation. If the Mac can complete all gates, generate a real `macos-mps` or `cpu` record as
appropriate. For Linux, add documentation and optionally an explicitly blocked/unrun template;
never fabricate a GPU name or passing status.

Finally, update `README.md` with the short daily workflow and update `docs/EXEC_PLAN.md` only to
link the durable validation mechanism and replace the CUDA acceptance gap when genuine Linux
evidence later exists. The Linux operator checks out and pulls the source to test, runs the
validation command, reviews only the platform-owned files, commits them with a message such as
`test: record milestone 2 Linux CUDA validation`, and pushes. The Mac operator pulls that commit
and runs the inspection command before using the report to update milestone status or plan code
changes.

## Concrete Steps

All commands run from the checked-out `Traffic_AI_Detection` repository root on either Mac or
Linux. During implementation, inspect the current state first:

    git status --short --branch
    python --version

After the models, runner, renderer, CLI, and tests exist, run the focused tests:

    pytest tests/unit/test_validation_records.py tests/integration/test_validation_cli.py

Expected behavior is that all focused tests pass without a GPU or network connection. Then run the
repository gates:

    pytest
    ruff check .
    mypy src

Generate a real local record only after the gates pass:

    dashcam-ai validate --milestone 2 --platform macos-mps

If MPS cannot be confirmed in that environment, use `--platform cpu` or expect the MPS record to be
blocked. Inspect the generated record:

    dashcam-ai validation-status validation/milestone-2/macos-mps.json

The output must name the source commit, current commit, platform, three gate results, overall
verdict, and freshness. On Linux, the equivalent workflow is:

    git pull --ff-only
    python -m pip install -e ".[cv,dev]"
    dashcam-ai validate --milestone 2 --platform linux-cuda
    dashcam-ai validation-status validation/milestone-2/linux-cuda.json
    git add validation/milestone-2/linux-cuda.json validation/milestone-2/linux-cuda.md
    git commit -m "test: record milestone 2 Linux CUDA validation"
    git push

Dependency installation is documentation for the Linux operator and is not performed
automatically by the validation command. Before committing, `git diff -- validation/` must show no
secrets, absolute private paths, model contents, or large binary data.

## Validation and Acceptance

The feature is accepted when a clean checkout with development dependencies can run the new
focused tests and all existing tests successfully, Ruff reports no violations, and strict Mypy
reports no errors. The command must create valid JSON and matching Markdown at the platform-owned
path, and running it twice must safely replace that pair without creating timestamp-named file
sprawl.

A controlled failing gate must produce a `failed` verdict and preserve the other gate results. A
missing required tool, unavailable requested accelerator, interrupted command, or dirty worktree
must prevent an authoritative passing verdict and explain why. A report generated for commit A
must be reported as stale when inspected from commit B. Malformed JSON, a future unsupported schema
version, or a path outside `validation/` must fail clearly rather than be trusted.

No Linux CUDA milestone may be considered validated merely because Mac tests pass or because
`configs/nvidia.yaml` parses. Acceptance of that platform requires a record created on Linux in
which CUDA is observed, the GPU identity is present, required gates pass, the record is clean, and
its `source_commit` matches the code being judged. Real-video acceptance remains separately visible
through optional acceptance evidence and is not inferred from unit tests.

## Idempotence and Recovery

All collection commands are read-only except for replacing the selected JSON and Markdown report
pair. Re-running generation for the same platform is safe and leaves history to Git. Atomic writes
prevent a partial report if rendering fails. If a run is interrupted while gates execute, rerun the
same command; no cleanup beyond an implementation-owned temporary sibling should be necessary.

If an incorrect report is generated, do not rewrite unrelated source files or Git history. Correct
the environment and rerun the command before committing. If an incorrect report was already
committed, add a new correcting report commit so the audit trail remains visible. Large analysis
artifacts continue to live under ignored `output/`; only their hashes and metrics enter the report.

## Artifacts and Notes

The canonical record should resemble this abbreviated shape; exact field names may be refined
during implementation but all meanings must remain represented:

    {
      "schema_version": 1,
      "milestone": "milestone-2",
      "platform": "linux-cuda",
      "source_commit": "<40-character Git SHA>",
      "worktree_dirty": false,
      "environment": {
        "os": "Linux",
        "python": "3.12.x",
        "accelerator_available": true,
        "accelerator_name": "<observed GPU name>"
      },
      "gates": [
        {"name": "pytest", "exit_code": 0, "status": "passed"},
        {"name": "ruff", "exit_code": 0, "status": "passed"},
        {"name": "mypy", "exit_code": 0, "status": "passed"}
      ],
      "verdict": "passed",
      "reasons": []
    }

The Markdown companion must prominently show `source_commit`, platform, freshness guidance, each
gate result, and the overall verdict. It must state that Git history is the record history and that
the report is invalid for later source commits until rerun.

## Interfaces and Dependencies

Use the standard library modules `subprocess`, `pathlib`, `platform`, `sys`, `tempfile`, `json`,
`hashlib`, and timezone-aware `datetime` as needed. Reuse Pydantic, which is already a project
dependency, for schema validation and serialization. Reuse the existing Typer CLI rather than
introducing another command framework.

At the end, `src/dashcam_ai/validation/records.py` must expose validated record types and a pure
verdict function. `src/dashcam_ai/validation/runner.py` must expose an injectable gate runner so
tests can supply deterministic execution results. `src/dashcam_ai/validation/render.py` must expose
atomic JSON/Markdown writing and report loading. `src/dashcam_ai/cli.py` must expose generation and
inspection commands through the installed `dashcam-ai` entry point.

No Git hosting API, CI service, database, or new third-party package is required. Git remains the
transport and history mechanism, while people retain explicit control over commits and pushes.

Revision note (2026-08-27): Initial plan created after read-only repository inspection and user
approval to design a Git-backed Linux-to-Mac validation handoff.

Revision note (2026-08-27): Expanded the approved design to add a repository-root `AGENTS.md`,
symmetric non-transferable macOS/Linux evidence, and milestone aggregation. Updated progress,
decisions, context, plan of work, and interim outcomes after implementation; final regression is
still pending.

Revision note (2026-08-27): Completed implementation and regression, added OS-to-platform identity
checks discovered during final review, and recorded the intentionally blocked milestone status
until clean macOS and Linux evidence is produced after commit.

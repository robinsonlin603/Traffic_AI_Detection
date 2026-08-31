# Harden Milestone 2 against real-world lane, motion, and tracking failures

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository follows the ExecPlan specification in `~/.codex/PLANS.md`.
This document must be maintained in accordance with that file. It builds on the completed
Milestone 2 delivery recorded in `docs/EXEC_PLAN.md`, but repeats all context needed to execute
this hardening work without relying on conversation history.

## Purpose / Big Picture

Milestone 2 can process a dashcam video and emit lane-change and cut-in events, but a long Linux
CUDA acceptance video exposed systematic real-world failures. A configured straight lane polygon
does not follow curves, parked and oncoming vehicles can appear to cross that polygon, people can
enter the vehicle event pipeline, short occlusions fragment one physical vehicle into many track
IDs, and duplicate tracks can create conflicting evidence. These failures produced false
confirmed events and missed genuine lane changes even though the artifact pipeline completed.

After this plan is complete, the analyzer will distinguish entering the ego lane from leaving it,
reserve cut-in confirmation for same-direction motor vehicles entering the ego lane, compensate
tracked-object motion for camera motion, refuse confirmation when lane geometry is unreliable,
and prevent duplicate or fragmented raw tracks from independently confirming the same maneuver.
The annotated video will remain readable in dense traffic by showing only a compact track ID and
class code near each box. The behavior will be demonstrated with automated synthetic tests, the
human-reviewed time ranges in the Linux CUDA video, and fresh platform validation records for the
exact completed commit.

This work remains image-space analysis. It does not estimate legal time-to-collision, physical
distance, liability, turn signals, or enforcement evidence. It does not introduce an LLM or VLM.

## Progress

- [x] (2026-08-31 08:16Z) Inspected the current branch, commit, clean worktree, completed
  Milestone 2 plan, cross-platform plan, source layout, and validation reports.
- [x] (2026-08-31 08:16Z) Converted the user's long-video review into named regression scenarios
  and approved five implementation slices.
- [ ] Slice 1: add regression fixtures, motor-vehicle eligibility, and explicit maneuver
  semantics.
- [ ] Slice 2: add ego-motion-compensated relative motion, direction compatibility, cut-in safety
  gates, and explainable confidence.
- [ ] Slice 3: add dynamic lane geometry with temporal quality and safe unknown behavior.
- [ ] Slice 4: add duplicate suppression, event-level track continuity, occlusion tolerance, and
  progress-based temporal decisions.
- [ ] Slice 5: simplify visualization, complete artifact integration and documentation, run the
  full regression, and collect fresh platform evidence.

## Surprises & Discoveries

- Observation: Both required platform reports are stale before hardening starts.
  Evidence: the current source commit is `2ddc5a9de09bd8ef71715a81f7c2df34d6835ef6`, while
  `validation/milestone-2/macos-mps.json` records `cef12052f6c636ca64631f997d5ef2d7bf5937fa`
  and `validation/milestone-2/linux-cuda.json` records
  `c7d77e31d2b33a06e86cd19605fe8159f27ffdbe`.

- Observation: The local shell used to author this plan does not currently expose the
  `dashcam-ai` executable.
  Evidence: `dashcam-ai milestone-status --milestone 2` returned `command not found`. Report
  freshness was therefore classified directly from the canonical JSON records and Git commit.

- Observation: The only confirmed cut-in in the reviewed Linux CUDA output was a false positive.
  Evidence: Track `#13021` was an oncoming vehicle on a left curve; the fixed green boundary swept
  across it and the configured blue corridor overlapped the opposing lane.

- Observation: Some apparent ID switches are duplicate tracks with class instability rather than
  two physical vehicles exchanging identities.
  Evidence: the user confirmed that raw tracks `#7683` and `#7684` simultaneously boxed the same
  physical vehicle while its class oscillated between car and truck.

## Decision Log

- Decision: Implement the hardening work in five sequential slices based from
  `feature/milestone2-hardening`, without creating an additional worktree.
  Rationale: each slice has an independently observable outcome, while dynamic lane geometry is
  isolated from lower-risk event semantics and visualization changes. The repository already has
  `feature/milestone2-hardening` and `feature/milestone2-hardening-slice1` at the clean baseline,
  so their history must be preserved rather than recreated.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Treat `car`, `truck`, `bus`, and `motorcycle` as the event-eligible motor-vehicle
  family; retain all detections in perception artifacts but exclude people, bicycles, and other
  classes from lane-change and cut-in state.
  Rationale: the reviewed video included confirmed person events, while class oscillation within
  the motor-vehicle family should not break event continuity.
  Date/Author: 2026-08-31 / Codex and user

- Decision: A lane change and a cut-in are different facts. A lane change records source lane,
  destination lane, and whether the object enters or leaves the ego lane. A cut-in requires a
  same-direction motor vehicle entering the ego lane and interacting with the forward corridor.
  Rationale: Track `#7769` genuinely left the ego lane and must be a lane change but not a cut-in;
  Track `#7015` entered the ego lane at 13:25–13:29 and then left it at 13:30–13:33, requiring two
  separate maneuvers.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Do not promise complete re-identification or replace BoT-SORT in this plan. Add a
  bounded event-level continuity layer that suppresses duplicates and can bridge short, strongly
  supported fragmentation while retaining every raw track ID as evidence.
  Rationale: full multi-object tracker redesign is a separate research problem. Milestone 2 needs
  safe event behavior when tracking is imperfect, not an unsupported guarantee that IDs never
  change.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Keep configured lane geometry as an initial prior and bounded fallback, but require a
  dynamic detector and per-frame `valid`, `degraded`, or `unknown` quality before confirmation.
  Rationale: the configured polygon is useful for initialization but is visibly wrong on curves,
  in alleys without lane markings, and when the ego vehicle is not centered in a lane.
  Date/Author: 2026-08-31 / Codex and user

- Decision: An annotation label is `#<track-id> <class-code>`. Class codes are `C` for car, `T`
  for truck, `B` for bus, `M` for motorcycle, `P` for person, and `BC` for bicycle. Event state is
  communicated by box color rather than label text.
  Rationale: long labels obscure distant vehicles in dense traffic. `BC` avoids the bicycle/bus
  collision while preserving the user's compact format.
  Date/Author: 2026-08-31 / Codex and user

## Outcomes & Retrospective

The plan is approved and authored, but no production implementation has started. The current
branch remains a clean hardening baseline. Milestone 2 hardening cannot be declared complete until
all five slices pass their focused and full automated gates, the reviewed scenarios have expected
results, and fresh macOS MPS and Linux CUDA reports refer to the exact source commit. CPU evidence
is useful but does not replace either required accelerator report.

## Context and Orientation

The project is a Python 3.12 command-line dashcam analyzer. `src/dashcam_ai/cli.py` builds the
runtime from validated settings in `src/dashcam_ai/config/models.py` and the YAML files in
`configs/`. `src/dashcam_ai/application/scene.py` is the streaming scene orchestrator: for each
frame it obtains lane geometry, estimates camera motion, evaluates each tracked object's lane
membership, advances temporal state, and builds lane-change and cut-in events.

The original image pixel space is canonical. Lane geometry models live in
`src/dashcam_ai/domain/lane.py`. `src/dashcam_ai/lane/configured.py` currently maps a normalized
four-point polygon into every frame and returns it as valid. `src/dashcam_ai/lane/membership.py`
classifies a tracked object's bottom-center anchor as inside, outside, or near a boundary. A
dynamic lane detector in this plan means an implementation of the existing `LaneDetector`
protocol in `src/dashcam_ai/lane/base.py` that updates boundaries from the current frame rather
than returning the same configured lines forever. It may return `degraded` when short-term
fallback is usable and `unknown` when the image cannot safely establish an ego lane.

Camera motion models live in `src/dashcam_ai/domain/motion.py`, and
`src/dashcam_ai/motion/opencv.py` estimates a previous-frame-to-current-frame homography from
background features. A homography is a 3-by-3 image transform that predicts how a stationary
background point moves between two camera frames. Relative object motion means comparing a
tracked anchor's observed movement with the movement predicted by that homography. Subtracting
the camera-induced component is necessary before interpreting sideways image movement as a
vehicle maneuver.

Temporal state is defined in `src/dashcam_ai/domain/temporal.py` and implemented by
`src/dashcam_ai/lane/temporal.py`. It currently uses signed boundary distance, debounce, a fixed
candidate timeout, and limited missing-observation tolerance. Event models are in
`src/dashcam_ai/domain/events.py`; builders are in `src/dashcam_ai/events/lane_change.py` and
`src/dashcam_ai/events/cutin.py`. The forward corridor is a configured image polygon created by
`src/dashcam_ai/events/corridor.py`. Structured frame, track, and event artifacts are written by
`src/dashcam_ai/storage/artifacts.py`. Annotation is drawn by
`src/dashcam_ai/visualization/annotator.py`.

Raw track ID means the integer emitted by the object tracker. Event continuity ID means a bounded
identifier used only by scene analysis to keep one physical maneuver coherent through a short
occlusion or a safely detected duplicate. The event layer must always preserve the contributing
raw IDs and must prefer rejection or uncertainty over unsafe merging. Duplicate suppression means
preventing two highly overlapping raw tracks with compatible motion from independently building
two events for the same physical vehicle.

The human-reviewed Linux CUDA video is not committed. Its compact acceptance labels are:

- `#3379` at 06:40–06:52, `#4910` at 09:55–10:09, `#4984` at 10:01–10:09,
  `#10295` at 18:41–19:03, and `#11661` at 20:12–20:50 are genuine entries into the ego lane and
  genuine cut-ins with stable IDs.
- `#7015` enters the ego lane at 13:25–13:29 and leaves it at 13:30–13:33. Only the first maneuver
  is a cut-in.
- `#7683/#7684` at 14:18–14:33 is one physical vehicle with duplicate boxes and car/truck class
  oscillation. It enters and later leaves the ego lane, and each physical maneuver may produce at
  most one event.
- `#7769`, observed through a fragmented raw-ID chain from 13:41–16:57, leaves the ego lane for
  the right lane. It is a lane change, not a cut-in.
- `#13021` at 21:00–21:50 is an occluded, fragmented oncoming vehicle on a curve and must not be a
  lane change or cut-in. `#3864` at 07:47–07:52 is parked and must not be confirmed. `#11211` at
  19:33–19:56 remains in the right lane while the road turns and must not be confirmed.
- `#10348` is correctly negative at 19:21–19:22 and 19:41–19:45 when a turning road moves the
  configured line, but it genuinely enters the ego lane at 19:50–20:00 and must not be lost to a
  prior rejection or fixed timeout.
- `#21362/#21436` disappear across about two seconds of occlusion at 30:34–30:37. The user could
  not confirm whether they are the same vehicle or whether a lane change completed, so this is an
  uncertainty-safety case, not a positive or negative pass/fail label.

Additional reviewed cases `#10093`, `#9199`, and `#10927` are likely rejected false negatives,
and `#13995` is a parked false positive. They may be added to the regression data only after their
time boundaries and expected maneuver semantics are made precise.

## Plan of Work

Slice 1 establishes truthful event vocabulary before changing motion or geometry. Add a compact,
Git-safe regression fixture under `tests/fixtures/` containing only video-relative times, raw IDs,
scene tags, and expected behavior; do not store the video path or frames. Extend domain events to
represent source lane, destination lane, entering or leaving the ego lane, and completion. Define
lane labels narrowly enough to represent ego, left adjacent, right adjacent, opposing, outside,
and unknown without pretending that the initial configured polygon maps every road lane. Filter
event state to the motor-vehicle family at the scene boundary while leaving perception output
unchanged. Update builders and serialization tests so a leaving lane change cannot create a
cut-in, and a two-stage maneuver such as `#7015` can create two distinct lane changes. Preserve
backward-readable artifacts where practical; if the schema must change incompatibly, add an
explicit schema version and document it.

Slice 2 turns the existing homography into relative-motion evidence. Add serializable evidence
that records observed anchor displacement, background-predicted displacement, compensated
displacement, direction compatibility, and quality. Keep image coordinates canonical and reject
non-finite or low-quality transforms. Use compensated lateral progress and longitudinal behavior
to distinguish a moving same-direction vehicle from a parked object or an oncoming vehicle.
Tighten `CutInDetector` so entering, same-direction compatibility, forward-corridor interaction,
track maturity, and adequate motion quality are all required. Add a scene-level sanity gate: when
many otherwise unrelated tracks appear to cross together during a strong camera turn, lower lane
evidence quality instead of confirming a burst of events. Replace opaque all-or-nothing scores
with a bounded confidence breakdown containing geometry, motion, continuity, direction, crossing
progress, and corridor components.

Slice 3 introduces dynamic lane geometry behind `LaneDetector`. First add deterministic synthetic
tests for straight, curved, missing, and abruptly changing line evidence. Then implement an OpenCV
baseline that operates inside a configurable road region, extracts lane-like image evidence, fits
left and right polylines or curves, and reports quality. Use temporal smoothing and jump rejection
so a single noisy frame cannot move the lane abruptly. Allow the configured polygon to initialize
the search and bridge a short, configurable gap, but degrade its status while doing so. After the
gap expires, or when an alley or large turn has no credible paired lane evidence, return unknown.
Update membership and scene analysis so degraded evidence may maintain an existing candidate only
when explicitly allowed, while unknown evidence can never advance or confirm it. The detector
must remain replaceable behind the existing protocol and must not add an LLM/VLM dependency.

Slice 4 adds bounded continuity without rewriting the underlying tracker. Introduce event-level
track state that records raw IDs, class-family votes, age, last observation, and continuity
quality. Suppress simultaneous duplicate tracks only when their boxes overlap strongly and their
motion and class family are compatible; choose one primary observation and retain both raw IDs in
evidence. Bridge fragmentation only over a configurable short gap when spatial and motion
continuity are unambiguous. A newly seen ID must satisfy a cooldown and observation count before
confirmation. Ambiguous association, including `#21362/#21436`, must remain separate and lower
confidence. Replace the fixed temporal timeout with progress-aware behavior: a candidate making
consistent compensated progress may continue, a stalled candidate rejects after a bounded time,
and a previous rejection must not prevent a later independent crossing such as `#10348` at
19:50–20:00. Ensure terminal event deduplication uses the physical maneuver and continuity
evidence rather than one raw ID alone.

Slice 5 completes visualization, artifacts, documentation, and acceptance. In
`src/dashcam_ai/visualization/annotator.py`, show only `#<raw-or-continuity-id> <class-code>` beside
each box. Use the approved class-code mapping, communicate idle/candidate/confirmed/rejected only
through the established colors, and keep a small legend where needed. Place labels using a
deterministic collision-avoidance search; preserve confirmed and candidate labels before ordinary
tracks, hide text for extremely small distant idle objects when no safe position exists, and use
a short leader line if a label must move away from its box. Reduce or hide long history lines for
distant idle objects. Keep all omitted details in JSON/JSONL. Add the new semantics, continuity
IDs, raw-ID list, geometry quality, relative-motion evidence, rejection reason, confidence
breakdown, and a non-private configuration digest to structured artifacts. Update README and the
completed Milestone 2 plan with the actual results and limitations.

Each slice uses its matching branch
`feature/milestone2-hardening-slice1` through
`feature/milestone2-hardening-slice5`, based sequentially on the preceding accepted slice and
merged into `feature/milestone2-hardening`. Work remains in the existing repository directory;
do not create another worktree. Branch creation, commits, merges, and pushes remain separate Git
actions and require the user's requested workflow; never push implicitly.

## Concrete Steps

Run all commands from the checked-out `Traffic_AI_Detection` repository root on macOS or Linux.
Before every slice, inspect durable state:

    git branch --show-current
    git rev-parse HEAD
    git status --short --branch
    sed -n '1,260p' docs/EXEC_PLAN_MILESTONE2_HARDENING.md
    find validation/milestone-2 -maxdepth 1 -type f -print -exec sed -n '1,80p' {} \;

At the end of every slice, run its focused tests. Exact new filenames may be refined while the
plan is kept current, but the intended groups are:

    pytest tests/unit/test_event_semantics.py tests/unit/test_cutin_events.py
    pytest tests/unit/test_relative_motion.py tests/unit/test_cutin_events.py
    pytest tests/unit/test_dynamic_lane_geometry.py tests/unit/test_temporal_lane.py
    pytest tests/unit/test_track_continuity.py tests/unit/test_temporal_lane.py
    pytest tests/integration/test_scene_pipeline.py tests/integration/test_opencv_pipeline.py

After focused tests, every slice must run the repository gates using the active Python
environment. Prefer module invocation so the environment is explicit:

    python -m pytest
    python -m ruff check .
    python -m mypy src

Expected results are zero failing tests, `All checks passed!`, and `Success: no issues found`.
The test count will grow and must be recorded in `Progress`; this plan intentionally does not
predict an exact final count.

For a short functional exercise, run an analyzed clip through the platform configuration and
inspect its artifacts:

    dashcam-ai analyze --input ./samples/test1.mp4 \
      --output ./output/test1-milestone2-hardening --config ./configs/mac.yaml

The command must complete without candidate events left unfinished. The annotated video must use
compact labels, and `events.json` must expose maneuver direction, continuity evidence, geometry
quality, and rejection reasons. If the console script is not installed, use the project's active
environment or install the repository through the documented development setup; do not change
source merely to work around an inactive shell.

After all slices are committed, rerun the private Linux CUDA acceptance video with its actual
configuration. Compare the resulting events with the time-relative labels in this plan. Do not
copy the video, model, multi-gigabyte JSONL, annotated MP4, private absolute paths, or usernames
into Git. Record compact counts and hashes only through the existing validation workflow.

Finally, each physical platform generates only its own report for the exact clean commit:

    dashcam-ai validate --milestone 2 --platform macos-mps
    dashcam-ai validate --milestone 2 --platform linux-cuda
    dashcam-ai milestone-status --milestone 2

The Mac command runs only on macOS with MPS observed; the Linux command runs only on Linux with
CUDA observed. A stale report is diagnostic history, not acceptance. CPU validation may be added
but cannot substitute for either required GPU report.

## Validation and Acceptance

Automated acceptance requires focused tests for every new behavior, the complete pytest suite,
Ruff, and strict Mypy on a clean source commit. Synthetic tests must prove motor-vehicle filtering,
entering versus leaving semantics, a two-stage maneuver, parked and oncoming rejection,
ego-motion compensation, dynamic curved geometry, safe unknown geometry, duplicate suppression,
fragment bridging, ambiguous non-bridging, progress-based timeout, and compact collision-aware
annotation. Existing Milestone 1 perception-only artifacts and device resolution behavior must
remain compatible.

Human acceptance uses the reviewed scenarios as behavior, not raw event counts. Tracks `#3379`,
`#4910`, `#4984`, `#10295`, and `#11661` must remain entering lane changes and cut-ins. Track
`#7015` must produce one entering cut-in followed by one leaving non-cut-in. Track `#7769` must be
a leaving lane change and never a cut-in despite fragmentation. The duplicate pair
`#7683/#7684` may produce at most one event for each physical crossing. Tracks `#13021`, `#3864`,
and `#11211` must not confirm. Track `#10348` must remain negative during the earlier green-line
movement and still be able to confirm its later genuine entry. No person or bicycle may produce a
lane-change or cut-in event. The ambiguous `#21362/#21436` association must not be forced into a
confirmed event.

Scene acceptance requires the dynamic geometry to follow a representative curve without the
right boundary occupying an unrelated adjacent or opposing lane. During 31:04–31:15, where the
reviewed video turns into an alley without lane markings, geometry must become degraded or unknown
and confirmed events must be suppressed. During shadow or exposure transitions, an unknown motion
or geometry result is acceptable and safer than a false valid estimate.

Visualization acceptance requires every visible text label to contain only the track identifier
and approved class code. Dense distant traffic must not be covered by long confidence or state
strings. State remains recognizable through color, confirmed and candidate labels take placement
priority, and complete evidence remains available in structured artifacts.

Milestone acceptance additionally requires current authoritative `macos-mps` and `linux-cuda`
records for the exact final commit. Each record must have a clean worktree, observe the requested
accelerator, and pass all gates. Until both reports are current, cross-platform Milestone 2
hardening remains blocked even if local tests pass.

## Idempotence and Recovery

Tests and analysis commands are safe to rerun into a new ignored output directory. Validation
generation replaces only the selected platform's stable JSON and Markdown files; Git preserves
earlier evidence. Never hand-edit another platform's generated report. If a validation run is
dirty, interrupted, or stale, correct the environment and rerun it rather than changing the
verdict manually.

Dynamic geometry must be introduced behind the existing protocol so configured geometry remains
available during development. If a slice fails, keep the last passing implementation and tests,
record the discovery here, and adjust only the unaccepted slice. Do not use destructive Git reset
or checkout commands. Do not delete user outputs or large local videos. Duplicate association and
fragment bridging must be bounded by age, time, and confidence so rerunning a frame sequence gives
deterministic results and memory does not grow with video duration.

## Artifacts and Notes

The approved compact label examples are:

    #3379 C
    #6268 B
    #1234 BC

The intended event distinction is:

    left adjacent -> ego lane   = lane change, entering, possible cut-in
    ego lane -> right adjacent  = lane change, leaving, never a cut-in
    opposing lane -> apparent configured boundary crossing = reject

The validation state at plan creation is:

    current source: 2ddc5a9de09bd8ef71715a81f7c2df34d6835ef6
    macOS MPS: stale, source cef12052f6c636ca64631f997d5ef2d7bf5937fa
    Linux CUDA: stale, source c7d77e31d2b33a06e86cd19605fe8159f27ffdbe
    CPU: missing

## Interfaces and Dependencies

Use the existing Pydantic domain-model style and OpenCV dependency. Do not add an LLM/VLM or a
new tracking framework. Exact names may be refined during Slice 1 and must then be recorded in the
Decision Log, but the completed system must expose these concepts through serializable models:

    class ManeuverRelation(StrEnum):
        ENTERING_EGO = "entering_ego"
        LEAVING_EGO = "leaving_ego"
        NOT_EGO_RELATED = "not_ego_related"
        UNKNOWN = "unknown"

    class RelativeMotionEvidence(BaseModel):
        observed_displacement: Point2D
        predicted_background_displacement: Point2D
        compensated_displacement: Point2D
        direction_compatible: bool | None
        confidence: float

    class TrackContinuityEvidence(BaseModel):
        continuity_id: int
        raw_track_ids: list[int]
        duplicate_suppressed: bool
        association_confidence: float

`LaneGeometryStatus` in `src/dashcam_ai/domain/lane.py` must support `valid`, `degraded`, and
`unknown`, and every non-valid state must have explicit scene behavior. `LaneDetector` remains the
runtime boundary for configured and dynamic implementations. `LaneChangeEvent` and `CutInEvent`
must expose maneuver relation and continuity evidence; a cut-in builder must reject any relation
other than entering the ego lane. Configuration thresholds belong in
`src/dashcam_ai/config/models.py` and the YAML files, never as unexplained constants inside event
logic.

Revision note (2026-08-31): Initial approved Milestone 2 Hardening ExecPlan created from the
long-video CUDA artifact audit, the user's human labels, and the five-slice delivery decision.

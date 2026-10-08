# Add dynamic lane geometry to general lane-change analysis

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository follows `~/.codex/PLANS.md`. This document must be maintained in
accordance with that specification. It extends the completed configured general-lane foundation
described in `docs/EXEC_PLAN_MILESTONE2_GENERAL_LANE_CHANGE.md`; it does not restore the removed
ego-lane, forward-corridor, or cut-in scope.

## Purpose / Big Picture

After this work, the analyzer can derive lane boundaries and lane regions from each video frame
instead of assuming that one fixed three-lane drawing fits an entire trip. A two-lane road can
remain two lanes, a curve can move its boundaries through the image, and weak or missing road
evidence can become `degraded` or `unknown` instead of creating false lane-change candidates.
The user can compare `output/test5/annotated.mp4` with a new dynamic output and see boundaries
following the road rather than pointing into the sky or covering the opposing carriageway.

This plan changes lane geometry only. Opposing-traffic filtering, parked-vehicle classification,
cross-ID re-identification, turn-signal recognition, cut-in, distance, TTC, and collision risk are
not implemented here. They depend on trustworthy geometry or belong to later work.

## Progress

- [x] (2026-09-01 13:35Z) Audited the configured `test5` MPS artifact and recorded human labels.
- [x] (2026-09-01 13:35Z) Defined the dynamic-geometry scope and separated later traffic-direction,
  parked-object, and continuity work.
- [x] (2026-09-01 14:02Z) Slice 1: added degraded geometry, dynamic/hybrid provenance,
  per-boundary evidence source and confidence, configured/dynamic/hybrid validated settings,
  platform YAML defaults, and deterministic contract tests. Focused tests passed 25 tests; the
  full suite passed 114 tests, Ruff, strict Mypy across 53 source files, and `git diff --check`.
- [x] (2026-09-01 15:42Z) Slice 2: built an overlay-only single-frame evidence prototype and
  compared classical, lane-instance, and segmentation paths. OpenCV was exercised for 300
  `test5` frames. Official CLRNet was inspected but could not be run safely without a compatible
  checkpoint/runtime. Official YOLOP ONNX was exercised for 60 frames and produced evidence on
  every frame, but fragmented markings and road edges into 5--9 components. It therefore remains
  degraded diagnostic evidence and is not integrated with events. Final automated gates passed
  120 tests, Ruff, strict Mypy across 58 source files, and `git diff --check`.
- [ ] Slice 3: core temporal topology implementation is complete: curve-fragment merging, ordered
  cross-frame boundary
  association, exponential smoothing, abrupt-jump rejection, bounded inferred gaps, safe
  reacquisition, confirmed topology versions, and two/three-lane region construction. The
  overlay-only command now records both raw evidence and temporal geometry. Synthetic tests cover
  initial confirmation, fragmentation, topology changes, missing evidence, jump rejection, and
  degraded input. A 120-frame `test5` diagnostic produced 69 degraded and 51 unknown geometry
  frames with no valid geometry; this is safe but not yet sufficient for real-video acceptance.
  Automated gates pass 130 tests, Ruff, strict Mypy across 59 source files, and `git diff
  --check`. Remaining before Slice 3 acceptance: derive or validate same-carriageway road edges and
  reject median/roadside components; the configured ROI edges are not sufficient real road-edge
  evidence.
- [x] (2026-09-01 17:02Z) Slice 4: integrated configured/dynamic/hybrid detector construction,
  explicit OpenCV or local YOLOP backend settings, and fail-closed scene behavior. Degraded and
  unknown geometry now always produce unknown membership; configured hybrid fallback remains
  degraded; topology version changes reset temporal track state and reject active candidates.
  Fake-pipeline regression proves a complete but degraded polygon sequence cannot emit events.
  Final gates passed 137 tests, Ruff, strict Mypy across 60 source files, and `git diff --check`.
- [ ] Slice 5: visualization and configured regression are complete, but real-video dynamic and
  platform acceptance remain blocked. The annotator exposes geometry status, provenance,
  topology ID, confidence, and distinct observed/inferred/configured/fallback boundary colors.
  A current CPU configured run reproduced 2471 frames, 358 tracks, and 75 events (70 rejected,
  5 unknown), including the false Track `#20` candidate. The 120-frame dynamic diagnostic remains
  0 valid, 69 degraded, and 51 unknown because road-edge semantics are unresolved. Current macOS
  MPS and Linux CUDA reports are stale; MPS is unavailable in this execution environment.
  Platform-independent gates pass 139 tests, Ruff, strict Mypy across 60 source files, and
  `git diff --check`.

## Surprises & Discoveries

- Observation (2026-09-03): Reacquisition after topology expiry could emit a null topology ID
  with the retained internal version, violating the LaneGeometry contract. The old recovery test
  used immediate one-frame confirmation and missed this waiting state. New six-frame recovery
  tests reproduce both missing-evidence and abrupt-jump failures before the fix. Pending geometry
  now emits both identity fields as null while keeping the internal counter for the next version.
  Verification: 141 tests, Ruff, strict Mypy (60 source files), and diff checks pass. The test4
  YOLOP rerun completed all 300 frames: 0 valid, 299 degraded, 1 unknown. This fixes the crash,
  not the outstanding road-edge semantics or dynamic acceptance blocker.

- Observation: The `test5` configured run is conservative at the terminal event gate but extremely
  noisy before that gate.
  Evidence: 75 lifecycle events contain 70 rejected, 5 unknown, and 0 confirmed; 55 claim
  `lane_center -> lane_left` on a video with only two same-direction lanes.

- Observation: Ego-motion does not explain the `test5` event burst.
  Evidence: 2470 of 2471 frames have valid ego-motion; only frame 0 is unknown because no previous
  frame exists.

- Observation: Fixed polygons can describe internally valid geometry while being visually false.
  Evidence: geometry confidence is always 1.0 although both shared green boundaries point into the
  sky and the left yellow region covers the median and opposing traffic.

- Observation: Dynamic geometry alone will not recover every genuine maneuver.
  Evidence: the genuine right-to-left maneuver changes raw Track ID from `#355` to `#407` after
  occlusion. Cross-ID continuity is deliberately outside this plan.

- Observation: Dynamic contracts can be introduced without activating an unfinished runtime.
  Evidence: all three platform YAML files explicitly remain in `configured` mode while carrying
  validated dynamic thresholds, and the complete configured regression remains green.

- Observation: The conservative classical OpenCV prototype is not a viable production detector
  for the beginning of `test5`.
  Evidence: over frames 0–299 it returned 1 valid, 13 degraded, and 286 unknown results. The rare
  visible candidates can follow median or roadside structure rather than trustworthy lane
  boundaries. Lowering the gate would increase false geometry, so the prototype remains an
  overlay-only diagnostic and is not connected to scene analysis.

- Observation: The official CLRNet source is not directly runnable in this project's supported
  environment and does not include a checkpoint.
  Evidence: official commit `7269e9d1c1c650343b6c7febb8e764be538b1aed` targets a Python 3.8-era
  Torch 1.8 / torchvision 0.9 / mmcv 1.2.5 CUDA stack, while this project uses Python 3.12 and
  Torch 2.13. No repository dependency or environment was changed to force an unsafe comparison.

- Observation: Official YOLOP lane segmentation is materially denser than the classical
  prototype but its connected components are not lane instances.
  Evidence: official commit `8d8f68df318c71f01d6f813c024df646c7d1978f`, MIT-licensed
  `yolop-640-640.onnx` with SHA-256
  `cd66a3e0087a7258ae07768cc02cb742eed93865727ae4c9baf969b8fa190696` produced 60 degraded,
  0 valid, and 0 unknown frames on the start of `test5`. The overlay shows plausible painted
  markings, but also fragments one marking and follows median or roadside structure. A 60-frame
  diagnostic ran at about 4.5 frames/s on CPU; lowering confidence would falsely promote 5--9
  components into lane boundaries.

- Observation: Temporal consistency cannot repair semantically wrong single-frame components.
  Evidence: the Slice 3 120-frame `test5` run stabilized a three-region topology for 50 frames,
  but the visible boundaries still included median or roadside structure. Because all YOLOP input
  evidence was degraded, the tracker emitted 0 valid, 69 degraded, and 51 unknown frames. The
  result is correctly blocked from event use, but road-edge rejection remains necessary before
  dynamic geometry can pass acceptance.

- Observation: Fail-closed behavior prevents false events but cannot by itself satisfy acceptance.
  Evidence: the dynamic sample has zero valid frames and therefore cannot reproduce Track `#20`'s
  configured false candidate, but it also cannot demonstrate the genuine two-lane topology. Zero
  dynamic events is suppression, not successful perception.

## Decision Log

- Decision: Keep `ConfiguredLaneDetector` as a deterministic baseline and introduce dynamic
  behavior behind the existing `LaneDetector` protocol.
  Rationale: configured tests remain reproducible without model weights, and downstream code can
  compare both paths through the same `LaneGeometry` contract.
  Date/Author: 2026-09-01 / Codex and user

- Decision: Treat configured coordinates as optional initialization or explicit fallback, never
  as silently valid dynamic evidence.
  Rationale: `test5` proves a syntactically valid fixed polygon can be geometrically wrong. Any
  fallback must expose degraded provenance and must not confirm an event.
  Date/Author: 2026-09-01 / Codex and user

- Decision: Prototype lane perception before binding the production pipeline to one learned model.
  Rationale: model availability, weights, licensing, Mac/CUDA portability, curved-road accuracy,
  and inference cost are unresolved. An overlay-only prototype gives observable evidence before
  an architectural commitment.
  Date/Author: 2026-09-01 / Codex and user

- Decision: Dynamic topology may change lane count and must not assume permanent IDs named only
  `lane_left`, `lane_center`, and `lane_right`.
  Rationale: the target videos contain two- and three-lane sections. Stable per-segment IDs and
  explicit appearance, disappearance, merge, or split transitions are safer than preserving a
  false lane.
  Date/Author: 2026-09-01 / Codex and user

- Decision: A topology transition cannot itself confirm a vehicle lane change.
  Rationale: a road merge, split, junction, or temporary geometry failure moves boundaries without
  proving that a tracked vehicle crossed a stable shared boundary.
  Date/Author: 2026-09-01 / Codex and user

- Decision: Do not promote the OpenCV Canny/Hough prototype into `LaneDetector`.
  Rationale: its safe gate rejects most real frames, while its accepted candidates are not
  semantically constrained to road lanes. It is useful for deterministic tests and diagnostics,
  not for satisfying curved two/three-lane acceptance.
  Date/Author: 2026-09-01 / Codex

- Decision: Carry YOLOP ONNX forward only as the preferred Slice 3 evidence source, not as a
  complete dynamic `LaneDetector`.
  Rationale: it supplies temporally dense semantic lane pixels and is usable through the existing
  OpenCV runtime on Mac, but component fragments require temporal association, curve merging,
  road-edge rejection, and topology validation before they can define lane IDs. CLRNet remains a
  documented blocked comparison rather than adding an incompatible legacy stack.
  Date/Author: 2026-09-01 / Codex

- Decision: Scene integration is fail-closed on geometry quality and topology identity.
  Rationale: only `valid` geometry may produce membership. `degraded`, `unknown`, inferred, and
  configured fallback geometry cannot advance a lane-change state. A topology version change
  clears per-track temporal state and rejects any active candidate rather than treating renamed or
  moved lanes as a vehicle crossing.
  Date/Author: 2026-09-01 / Codex

## Outcomes & Retrospective

Planning and Slice 1 contracts are complete. The configured pipeline remains the passing
deterministic baseline, while domain and configuration models can now represent valid, degraded,
or unknown dynamic/hybrid evidence. No dynamic perception backend is active yet. The success
measure is not merely drawing different lines: the dynamic result must reduce geometry-driven
candidates on `test5`, safely become unknown where the road cannot be inferred, and preserve all
Milestone 1 behavior.

## Context and Orientation

This is a Python 3.12 offline dashcam analyzer. `src/dashcam_ai/cli.py` loads Pydantic settings
from `src/dashcam_ai/config/models.py`. `src/dashcam_ai/application/analyzer.py` streams decoded
frames and detections. `src/dashcam_ai/application/scene.py` asks a `LaneDetector` for one
`LaneGeometry` per frame, evaluates each Track bottom-center through
`src/dashcam_ai/lane/membership.py`, compensates camera motion, advances temporal state, and builds
general lane-change events.

`src/dashcam_ai/lane/base.py` contains the replaceable `LaneDetector` protocol. The current
`src/dashcam_ai/lane/configured.py` implementation maps normalized YAML polygons and shared
boundaries into original image pixels. `src/dashcam_ai/domain/lane.py` defines immutable lane
regions, boundaries, geometry status, provenance, confidence, and membership features.

A lane boundary is a polyline separating two adjacent lane regions. A lane region is a polygon in
which a vehicle bottom-center may be assigned. Dynamic topology means that the number and
adjacency of those regions can change over time. `degraded` means some recent or partial evidence
is usable for visualization or maintaining an existing state, but it is not trustworthy enough
to confirm a maneuver. `unknown` means no lane membership or event progress may be inferred.

The existing `configs/default.yaml`, `configs/mac.yaml`, and `configs/nvidia.yaml` configure three
fixed regions. The test artifact `output/test5` is intentionally untracked and must remain local.
Compact human findings are stored in `docs/MILESTONE2_ACCEPTANCE.md` without private output paths,
frames, model weights, or large artifacts.

## Plan of Work

### Slice 1: quality contracts and configuration

Extend `src/dashcam_ai/domain/lane.py` so `LaneGeometryStatus` supports `valid`, `degraded`, and
`unknown`, and provenance distinguishes `configured`, `dynamic`, `hybrid`, and `unknown`.
Represent per-boundary confidence and whether a boundary is directly observed, temporally inferred,
or configured fallback. Preserve JSON compatibility for configured valid geometry.

Extend `src/dashcam_ai/config/models.py` with a mode named `configured`, `dynamic`, or `hybrid` and
a nested dynamic section. It must validate a road region of interest, minimum confidence,
short-gap tolerance, temporal smoothing, maximum allowed boundary jump, curve-fit quality, and
topology confirmation duration. Put defaults in all three YAML files. No unexplained threshold may
be embedded in the detector or event logic.

Add focused tests to `tests/unit/test_lane_geometry.py`, `tests/unit/test_config.py`, and a new
`tests/unit/test_dynamic_lane_geometry.py`. Prove serialization, invalid configuration,
degraded/unknown invariants, and configured backward behavior. This slice must not load weights,
access the network, or change event results.

### Slice 2: single-frame evidence prototype

Add an implementation under `src/dashcam_ai/lane/dynamic.py` plus a narrow backend protocol so
lane perception is replaceable. The first runnable prototype operates only inside the configured
road region of interest and outputs image-space curve candidates with confidence. It must not
construct a confident three-lane topology merely because the old config contains three lanes.

Before choosing a production learned backend, compare at least one lane-instance approach and one
lane-segmentation/drivable-area approach on deterministic saved synthetic inputs and sampled local
frames. The prototype may require optional local weights, but automated tests use fake backends and
synthetic arrays. Record model name, version, local weight hash, runtime, and observed limitations;
never commit weights or video frames. Promote a backend only if it follows representative straight
and curved road boundaries, supports the target Mac/CUDA environments or has an explicit fallback,
and becomes low-confidence where evidence is absent.

Provide an overlay-only exercise that draws candidate curves, confidence, and detected lane count.
It must not call `TemporalLaneTracker` or emit `events.json`. This keeps perception feasibility
separate from event tuning.

### Slice 3: temporal boundaries and topology

Add bounded state under `src/dashcam_ai/lane/` that associates curves across frames, smooths curve
control points, rejects abrupt jumps, and bridges only a short configured evidence gap. Camera
motion from the existing homography may help project the previous boundary, but cannot turn stale
configured lines into valid observations.

Build lane regions from ordered neighboring boundaries and stable road edges. Assign stable IDs
within a topology segment. Require repeated evidence before adding or removing a lane, and record
topology changes such as appeared, disappeared, merged, or split. During an unconfirmed topology
change return degraded; after evidence expires return unknown. Add synthetic tests for straight
and curved roads, two-to-three and three-to-two transitions, occlusion, jump rejection, no-line
alleys, exposure loss, and recovery.

### Slice 4: safe scene integration

Update `src/dashcam_ai/cli.py` to construct configured, dynamic, or hybrid detectors from validated
configuration. Keep `src/dashcam_ai/application/scene.py` dependent only on `LaneDetector`.
Update membership and temporal state so valid stable geometry may advance a maneuver, degraded
geometry may only preserve explicitly allowed prior state for a bounded time, and unknown geometry
cannot approach, cross, enter, or confirm. A topology version change cannot be interpreted as a
vehicle crossing.

Update `src/dashcam_ai/visualization/annotator.py` to distinguish observed, inferred, fallback, and
unknown geometry without obscuring Track labels. Structured frame output must expose status,
provenance, confidence, topology identity, and reasons. General lane-change event schema remains
unchanged unless dynamic evidence requires an additive field.

### Slice 5: acceptance and platform evidence

Run the complete automated suite and compare configured and dynamic outputs on `samples/test5.mp4`.
Human review must verify that the dynamic overlay identifies the two same-direction lanes, follows
curves rather than the sky, and becomes degraded or unknown near junctions or absent markings.
Track `#20` must no longer become a geometry-driven candidate. The event burst should fall
substantially from the configured baseline of 75 without achieving the reduction by suppressing all
valid geometry. The genuine `#355` maneuver is evaluated only until its raw ID changes to `#407`.

The final clean commit requires separate authoritative macOS MPS and Linux CUDA reports. A report
for either platform cannot validate the other, and older reports are stale.

## Concrete Steps

Run every command from the `Traffic_AI_Detection` repository root. At each stopping point:

    git status --short --branch
    git log --oneline --decorate -8
    sed -n '1,320p' docs/EXEC_PLAN_DYNAMIC_LANE_GEOMETRY.md

For Slice 1, run:

    .venv/bin/pytest tests/unit/test_lane_geometry.py tests/unit/test_config.py \
      tests/unit/test_dynamic_lane_geometry.py
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

For later slices, add focused dynamic, temporal, scene-pipeline, OpenCV, and annotator tests before
running the same full gates. At every completed slice run:

    .venv/bin/pytest
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

The configured comparison remains:

    .venv/bin/dashcam-ai analyze --input ./samples/test5.mp4 \
      --output ./output/test5-configured-comparison --config ./configs/mac.yaml

The Slice 2 diagnostic command is:

    .venv/bin/dashcam-ai lane-overlay --input ./samples/test5.mp4 \
      --output ./output/test5-lane-evidence-prototype --config ./configs/mac.yaml \
      --maximum-frames 300

It writes only `lane-overlay.mp4`, `lane-evidence.jsonl`, and `summary.json`. It does not load YOLO,
create Tracks, invoke scene analysis, or emit lane-change events.

The optional official YOLOP comparison is:

    .venv/bin/dashcam-ai lane-overlay --input ./samples/test5.mp4 \
      --output ./output/test5-yolop-prototype --config ./configs/mac.yaml \
      --backend yolop-onnx --weights /local/path/to/yolop-640-640.onnx \
      --maximum-frames 60

The path is intentionally local, the weight is never committed, and its SHA-256 is included in
each evidence record so a result cannot be attributed to an unknown model artifact.

The dynamic command will use the same input and a dynamic-mode configuration, writing to a new
ignored output directory. Never overwrite `output/test5`, commit the MP4, JSONL, model weights, or
private absolute paths.

## Validation and Acceptance

Core automated tests require no GPU, weights, real video, or network. They must prove status and
provenance serialization, safe degraded/unknown behavior, boundary association, curved geometry,
two/three-lane topology, merge/split stability, missing evidence, abrupt jump rejection, and that
topology movement alone never confirms a lane change. Existing configured geometry and all
Milestone 1 regression tests must remain green.

Real-video acceptance compares the same `test5` frames. A human must see two same-direction lane
regions where two exist. Neither green shared boundary may point into the sky, and the left region
must not claim the opposing carriageway as a normal same-direction lane. Where those claims cannot
be made reliably, the artifact must say degraded or unknown. Zero events alone is not success;
the overlay and structured confidence must prove useful dynamic geometry.

Final acceptance requires `pytest`, Ruff, strict Mypy, a clean worktree, current macOS MPS evidence,
and current Linux CUDA evidence for the exact same source commit. Unavailable hardware is recorded
as blocked, never inferred from another platform.

## Idempotence and Recovery

Configured mode remains available throughout development. New analysis runs always use new ignored
output directories, so they are safe to repeat. If a dynamic backend fails or weights are absent,
the detector returns a documented unknown result rather than mutating config or downloading files
silently. If hybrid fallback is used, it is visibly degraded and cannot confirm an event.

Do not delete user outputs, model weights, or samples. Do not use destructive Git operations. If a
slice fails, keep the last passing code, record the discovery here, and narrow only the unfinished
slice. Validation files are generated only on their matching physical platform and are never
manually fabricated.

## Artifacts and Notes

The compact configured baseline is:

    input: samples/test5.mp4
    duration: 82.45 seconds
    frames: 2471
    actual same-direction lanes: 2
    configured regions: 3
    events: 75 = 70 rejected + 5 unknown + 0 confirmed
    dominant transition: lane_center -> lane_left, 55 events
    ego-motion: 2470 valid + 1 initial unknown

The large local files under `output/test5` are evidence for the user but are intentionally ignored.
Human labels and conclusions are preserved in `docs/MILESTONE2_ACCEPTANCE.md`.

## Interfaces and Dependencies

`src/dashcam_ai/lane/base.py` continues to define:

    class LaneDetector(Protocol):
        def detect(self, frame: Any, width: int, height: int) -> LaneGeometry: ...

Slice 1 extends `LaneGeometryStatus` with `DEGRADED` and `LaneGeometryProvenance` with `DYNAMIC`
and `HYBRID`. The exact per-boundary evidence model must be immutable, JSON serializable, bounded,
and carry confidence plus `observed`, `inferred`, or `configured_fallback` source state.

The runtime may use existing OpenCV and NumPy. A learned model integration must be optional and
isolated behind a backend protocol; its tests use fakes. Do not add an LLM/VLM, network dependency,
or mandatory GPU requirement. Any new package or weight requires a separate explicit decision,
license review, reproducible version, and user approval before installation or download.

Revision note (2026-09-01): Initial dynamic lane geometry plan created after the configured
`test5` two-lane audit proved that fixed three-lane polygons generate systematic false candidates.

Revision note (2026-09-01): Recorded Slice 1 quality contracts, explicit configured runtime mode,
new deterministic tests, and complete automated verification.

Revision note (2026-09-01): Recorded the partial Slice 2 evidence backend, standalone overlay,
OpenCV feasibility result, safe rejection decision, and the remaining learned-backend comparison.

Revision note (2026-09-01): Recorded Slice 3 temporal association and topology implementation,
including the safe but not yet acceptance-quality result of the 120-frame YOLOP diagnostic.

Revision note (2026-09-01): Recorded Slice 4 mode construction, hybrid fallback, topology reset,
fail-closed membership, fake-pipeline regression, and complete automated verification.

Revision note (2026-09-02): Recorded Slice 5 visualization, the full configured CPU comparison,
dynamic acceptance blocker, current platform-independent gates, stale platform evidence, and
unavailable MPS runtime.

# Refocus Milestone 2 on general configured-lane changes

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository follows the ExecPlan specification in `~/.codex/PLANS.md`. This document must
be maintained in accordance with that file. It supersedes the active scope of
`docs/EXEC_PLAN_MILESTONE2_HARDENING.md`; the older plan remains historical evidence of the
ego-lane and cut-in implementation that existed before this scope change.

## Purpose / Big Picture

Milestone 2 will determine whether a tracked road vehicle changed from one configured lane to an
adjacent configured lane. A resulting event identifies source lane, target lane, left or right
direction, approach time, boundary-crossing time, target-lane completion time, confidence, and
supporting frames, trajectory, geometry, membership, ego-motion, and relative-motion evidence.
Users can inspect these facts in `events.json` and `annotated.mp4`.

Milestone 2 no longer decides whether a vehicle entered the motorcycle's own lane and no longer
implements cut-in, a forward corridor, vehicle proximity, distance, time-to-collision, collision
risk, responsibility, turn signals, or LLM/VLM analysis. Turn signals and lane-change timeline
fusion belong to Milestone 3. Learned or dynamic lane detection and cross-ID re-identification
are also outside this plan; the accepted first version uses normalized configured geometry.

## Progress

- [x] (2026-08-31 14:30Z) Inspected the merged implementation, branch history, configuration,
  domain models, pipeline, tests, and stale platform reports.
- [x] (2026-08-31 14:30Z) Preserved and merged the old Hardening Slice 2 relative-motion work.
- [x] (2026-08-31 14:30Z) Approved and recorded this five-slice replacement plan.
- [x] (2026-08-31 15:05Z) Slice 1: added general multi-lane models, topology validation,
  normalized configured geometry for all platform YAML files, CLI construction, a bounded
  `reference_lane` compatibility adapter, legacy temporal aliases for the new boundary IDs, and
  focused geometry/configuration tests. Focused tests passed 26 tests; the full suite passed 111
  tests, Ruff, strict Mypy across 55 source files, and `git diff --check`.
- [ ] Slice 2: general membership, temporal smoothing, hysteresis, debounce, and missing tolerance.
- [ ] Slice 3: general lane-change timeline and source-to-target relative-motion safety gates.
- [ ] Slice 4: general event and pipeline integration; remove cut-in and forward corridor.
- [ ] Slice 5: visualization, documentation, regression, real-video review, and platform evidence.

## Surprises & Discoveries

- Observation: The preserved old Slice 2 mixes reusable relative motion with cut-in integration.
  Evidence: commits `4243d86` and `bfe1d34` add safe homography projection, stationary and scene
  motion gates, but also add cut-in confidence weights. The former must be retained and the latter
  removed through reviewable commits.

- Observation: The current geometry and event models are structurally ego-relative.
  Evidence: `src/dashcam_ai/domain/lane.py` exposes one `ego_lane`, while
  `src/dashcam_ai/domain/temporal.py` exposes `LanePosition.EGO` and entering/leaving relations.

- Observation: Existing platform reports cannot validate this migration.
  Evidence: checked-in macOS MPS and Linux CUDA reports reference older source commits; project
  rules require clean reports that observe the requested accelerator at the exact final commit.

- Observation: Migrating geometry before membership requires a temporary reference region for
  the still-ego-relative Slice 2 boundary.
  Evidence: existing membership and visualization consume one polygon. Slice 1 now selects the
  configured `lane_center`, or the median ordered lane when that ID is absent, through the
  non-serialized `LaneGeometry.reference_lane` property. Slice 2 must remove this adapter when it
  evaluates every lane.

## Decision Log

- Decision: Represent lanes with configured string IDs and unique image-left-to-image-right
  lateral orders, and represent explicit shared boundaries between adjacent lanes.
  Rationale: `lane_left`, `lane_center`, and `lane_right` express general transitions without
  assuming which lane contains the camera. A shared boundary proves adjacency, while order derives
  the semantic direction.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Use normalized configured lane polygons and boundaries; do not add dynamic or learned
  lane detection to this milestone.
  Rationale: this is the approved deterministic implementation and is testable without GPU
  models, weights, real videos, or network access. Its limitations on curves and camera-pose
  changes must remain explicit.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Derive left or right only from source and target lane order.
  Rationale: bounding-box movement includes perspective and camera motion. Compensated motion is
  supporting evidence, not the semantic source of direction.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Remove cut-in and forward-corridor APIs and artifact fields instead of leaving a
  disabled compatibility path.
  Rationale: keeping unsupported fields would leave the output contract ambiguous; Git history
  already preserves the former behavior.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Preserve bounded missing-observation and occlusion tolerance, but defer duplicate
  fusion and cross-ID re-identification.
  Rationale: temporal tolerance is needed within a track, while proving multiple IDs represent one
  physical vehicle is a separate tracking problem outside the revised acceptance criteria.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Deliver five sequential slices on `feature/milestone2-general-lane-change`, without an
  additional worktree or push unless explicitly requested.
  Rationale: each slice is independently reviewable, and commit `077311a` preserves the merged old
  scope as a clean starting point.
  Date/Author: 2026-08-31 / Codex and user

- Decision: Permit `LaneGeometry.reference_lane` only as an intermediate Slice 1 compatibility
  adapter and never serialize it as `ego_lane`.
  Rationale: geometry and configuration can migrate atomically while the full pre-existing suite
  remains usable; keeping the property out of artifacts prevents the new schema from claiming an
  ego lane. Slice 2 owns its removal.
  Date/Author: 2026-08-31 / Codex

## Outcomes & Retrospective

The replacement scope is approved and the old ego-lane/cut-in work is preserved in Git. Slice 1
now provides validated general configured lanes and shared boundaries without an `ego_lane`
artifact field. Membership and visualization still use a documented, non-serialized reference
adapter until Slice 2. Completion requires the remaining four slices, automated gates, a readable
general lane-change artifact, and authoritative macOS MPS and Linux CUDA reports for the final
commit. Update this section at each slice and replace it with the final outcome at acceptance.

## Context and Orientation

This is a Python 3.12 offline dashcam analyzer. `src/dashcam_ai/cli.py` loads Pydantic settings
from `src/dashcam_ai/config/models.py` and YAML under `configs/`.
`src/dashcam_ai/application/analyzer.py` processes video and delegates lane reasoning to
`src/dashcam_ai/application/scene.py`. Milestone 1 perception normalizes YOLO and BoT-SORT output
before this layer and must remain unchanged.

Original image pixels are canonical. Normalized coordinates range from zero to one and map to the
source resolution. A lane region is a polygon for one configured lane. A lane boundary is a
polyline shared by two configured lanes. Lateral order is an integer ordering lanes from image
left to image right. Lanes are adjacent only when they share a configured boundary.

Lane models are in `src/dashcam_ai/domain/lane.py`; `src/dashcam_ai/lane/base.py` defines the
replaceable `LaneDetector`; `src/dashcam_ai/lane/configured.py` maps normalized geometry;
`src/dashcam_ai/lane/membership.py` evaluates the track bottom-center; and
`src/dashcam_ai/lane/temporal.py` smooths observations and advances the state machine.

`src/dashcam_ai/motion/opencv.py` estimates a previous-to-current frame homography from background
features outside tracked boxes. A homography is a 3-by-3 transform predicting stationary
background motion. `src/dashcam_ai/motion/relative.py` subtracts that prediction from observed
bottom-center motion. Invalid transforms or unsafe projections produce unknown evidence and can
never confirm an event.

Event models and construction live in `src/dashcam_ai/domain/events.py` and
`src/dashcam_ai/events/lane_change.py`. Cut-in and corridor modules still exist and are removed in
Slice 4 after the general lane path works. `src/dashcam_ai/storage/artifacts.py` writes output and
`src/dashcam_ai/visualization/annotator.py` draws annotated video.

`inside_lane` means the anchor is reliably inside one region. `near_boundary` means it is within
the configured boundary margin. `outside_configured_lanes` means no configured polygon contains
it. `unknown` means geometry, motion, or observation quality is insufficient. Hysteresis uses
different entry and exit conditions to avoid threshold chatter. Debounce requires repeated
consistent observations. Target-lane dwell is the required stable time in the new lane.

## Plan of Work

### Slice 1: General multi-lane models and configured geometry

Replace the ego-only contract in `src/dashcam_ai/domain/lane.py` with collections of `LaneRegion`
and `LaneBoundary`. A region has `lane_id`, unique integer `lateral_order`, and polygon. A boundary
has `boundary_id`, a polyline, and left/right lane IDs. `LaneGeometry` contains immutable
collections, status, confidence, and provenance, and validates identifiers and references.

Change `LaneGeometryConfig` and all three YAML files to configure lanes and boundaries. Reject
duplicate IDs or orders, missing references, the same lane on both boundary sides, too few points,
and coordinates outside zero to one. Update `ConfiguredLaneDetector` to map all points to original
pixels. Add tests in `tests/unit/test_lane_geometry.py` and `tests/unit/test_config.py`. At the end,
three lanes map correctly at multiple resolutions without an ego lane. Temporal logic, artifacts,
cut-in, and visualization remain unchanged except for minimal compatibility needed for tests.

### Slice 2: General membership and temporal stabilization

Update membership to evaluate every lane polygon and relevant boundary. Output `inside_lane`,
`near_boundary`, `outside_configured_lanes`, or `unknown`, plus lane ID, nearest boundary,
signed or equivalent distance, anchor, confidence, and boundary-side candidates. Ambiguous
overlaps and weak geometry return unknown.

Refactor observation stabilization so a stable lane ID changes only after configurable smoothing,
hysteresis, and debounce. Preserve bounded history and missing-observation tolerance. Low-quality
observations cannot overwrite a stable lane. Test every membership state, overlap ambiguity,
jitter, temporary occlusion, missing observations, and low confidence. This slice produces stable
membership sequences but no final general lane-change event.

### Slice 3: General timeline and relative-motion safety

Replace the ego-relative temporal model with `stable_in_lane`, `approaching_boundary`,
`crossing_boundary`, `entered_new_lane`, and terminal outcomes. State retains source, target,
shared boundary, start/crossing/entry/completion times, bounded history, ego-motion quality, and
relative-motion summary.

A candidate requires a stable source lane and approach to a boundary shared with one adjacent
lane. Confirmation requires crossing that boundary, stable target-lane dwell, valid ego motion,
enough compensated lateral progress, direction consistency, and scene consistency. Returning,
timing out, or insufficient dwell rejects. Invalid geometry or motion and non-adjacent jumps are
unknown, never confirmed.

Refactor relative-motion expected direction to use source and target lateral order instead of ego
relations. Lower target order means left; higher means right. Test no-change, left, right, return,
insufficient dwell, camera-only translation, invalid motion, inconsistent direction, scene shift,
occlusion, and non-adjacent jumps.

### Slice 4: Event contract, pipeline, and cut-in removal

Refactor `LaneChangeEvent` to expose `source_lane`, `target_lane`, direction, start, crossing and
completion frames/times, status, confidence breakdown, and evidence. Evidence includes boundary,
supporting frames, anchors, membership/distances, ego motion, relative motion, and trajectory.
Enforce timeline ordering when fields exist; incomplete candidate, rejected, and unknown events
may have null later times.

Adapt the builder, scene pipeline, CLI, and artifacts. Then remove `CutInEvent`, `CutInDetector`,
`ConfiguredForwardCorridor`, their configuration, lifecycle, artifacts, visualization, and tests.
Search all references first and preserve Milestone 1. The artifact schema intentionally advances:
`events.json` contains only lane changes and frame analysis has no cut-in or corridor fields.

### Slice 5: Visualization, documentation, and acceptance

Draw all configured boundaries and lane IDs, current membership, bottom-center trajectory, and
candidate/confirmed lane-change styles. Display `LANE CHANGE LEFT` or `LANE CHANGE RIGHT`. Keep
compact labels: car `C`, truck `T`, bus `B`, motorcycle `M`, person `P`, bicycle `BC`; use box
color for event status so distant objects remain readable.

Update README, configuration examples, plans, schema, calibration instructions, and limitations.
Human review checks lane changes only and records source, target, direction, approach, crossing,
completion, system result, ID stability, and geometry suitability. Run full regression and final
exact-commit macOS MPS and Linux CUDA validation; CPU is supplementary and transfers to neither.

## Concrete Steps

Run from `/Users/robinsonliu/Desktop/Traffic_AI_Detection`. At every slice boundary inspect:

    git status --short --branch
    git log --oneline --decorate -8

Use focused tests for the files changed, then always run:

    .venv/bin/pytest
    .venv/bin/ruff check .
    .venv/bin/mypy src
    git diff --check

The plan-creation baseline is 111 passing tests, Ruff passing, and strict Mypy passing across 55
source files. Counts may change when cut-in tests are replaced; zero failures and complete intended
coverage matter more than preserving the count.

For final real-video evidence, use a config calibrated to that camera and a new ignored directory:

    .venv/bin/dashcam-ai analyze \
      --input ./samples/test1.mp4 \
      --output ./output/test1-general-lane-change \
      --config ./configs/mac.yaml

If MPS is unavailable, record it blocked and use CPU only as supplementary evidence. At the final
clean commit run matching commands on matching machines:

    .venv/bin/dashcam-ai validate --milestone 2 --platform macos-mps
    .venv/bin/dashcam-ai validate --milestone 2 --platform linux-cuda
    .venv/bin/dashcam-ai milestone-status --milestone 2

## Validation and Acceptance

Automated acceptance covers polygon and boundary geometry, normalized mapping, signed distance,
membership and stabilization, left/right/no-change, approach-return, insufficient dwell, jitter,
occlusion, missing observations, camera-only movement, invalid ego motion, inconsistent direction,
non-adjacent jumps, timeline ordering, JSON schema, fake-pipeline integration, and Milestone 1.

Artifact acceptance requires general source and target IDs, direction, timeline, confidence, and
evidence in `events.json`; no cut-in or forward corridor; and readable lanes, membership,
trajectories, compact labels, and event styles in `annotated.mp4`. One frame, camera-only motion,
invalid ego motion, non-adjacent jumps, or insufficient target dwell cannot confirm an event.

Milestone 2 is complete only when automated gates pass, Milestone 1 remains intact, human review
demonstrates no-change, left, right, and insufficient-evidence outcomes, and both final-commit
macOS MPS and Linux CUDA reports are authoritative.

## Idempotence and Recovery

Tests and new ignored output directories are safe to repeat. Migrate Pydantic models and all YAML
files atomically. Keep compatibility adapters only while required for a passing intermediate slice
and remove them in Slice 4. Never destructively reset, discard user changes, delete user media, or
hand-edit another platform's report. Verify all references before deleting cut-in files. If a
slice fails, keep the last passing slice, record the issue here, and retry only incomplete work.

## Artifacts and Notes

Configured topology and direction examples are:

    lane_left | boundary_left | lane_center | boundary_right | lane_right
    lane_left   -> lane_center = right
    lane_center -> lane_left   = left
    lane_center -> lane_right  = right
    lane_right  -> lane_center = left

A confirmed event should resemble:

    {
      "event_type": "lane_change",
      "status": "confirmed",
      "track_id": 17,
      "source_lane": "lane_right",
      "target_lane": "lane_center",
      "direction": "left",
      "started_at": 42.7,
      "lane_crossed_at": 43.2,
      "completed_at": 43.8,
      "confidence": {"overall": 0.86},
      "evidence": {"frame_ids": [1281, 1295, 1314], "boundary_id": "boundary_right"}
    }

## Interfaces and Dependencies

Use existing Pydantic v2, OpenCV, YAML loader, and enums. Add no learned lane model, tracker,
network service, LLM, or VLM. Exact signatures may be refined and logged, but the end state must
provide `LaneDirection`, `LaneRegion(lane_id, lateral_order, polygon)`,
`LaneBoundary(boundary_id, left_lane_id, right_lane_id, points)`, multi-lane `LaneGeometry`, a
membership feature with lane ID and boundary evidence, and a `LaneChangeEvent` with source,
target, direction, timeline, confidence, and evidence.

`LaneDetector` remains the geometry protocol, `EgoMotionEstimator` remains the camera-motion
protocol, and `RelativeMotionEvaluator` remains responsible for projection and relative movement.
Every threshold comes from configuration, never unexplained constants in event logic.

Revision note (2026-08-31): Created the approved general configured-lane-change plan after
preserving the former ego-lane/cut-in Hardening work. This defines five replacement slices and
makes cut-in, forward corridor, dynamic lane detection, and cross-ID re-identification out of
scope.

Revision note (2026-08-31): Recorded Slice 1 general lane topology, temporary reference-lane
compatibility decision, and focused/full validation evidence.

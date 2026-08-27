# Motorcycle Dashcam AI — Milestone 2 Execution Plan

This living plan extends the Milestone 1 perception pipeline with deterministic,
non-LLM lane-change and cut-in analysis:

```text
tracked video frames -> configured lane geometry -> ego-motion estimate
  -> temporal lane membership -> lane-change state -> cut-in evidence
  -> structured events + annotated video
```

## Scope and constraints

- Original-frame pixels remain the canonical coordinate system.
- The first lane implementation uses configured normalized geometry; learned lane
  models remain replaceable behind a `LaneDetector` protocol.
- Camera motion is estimated from background image features outside tracked vehicle
  boxes. An invalid estimate produces unknown evidence, never a positive event.
- Events require temporal evidence and expose candidate, confirmed, or rejected status.
- Cut-in confidence is based on explainable image-space geometry and motion. It does
  not claim real distance, legal TTC, liability, or enforcement suitability.
- Turn signals, LLM/VLM, FastAPI, GPS, and learned lane-model integration are excluded.

## Architecture decisions

- Lane geometry, ego motion, lane membership, and events use serializable domain models.
- Runtime implementations depend on protocols; OpenCV and learned models do not leak
  into application or domain layers.
- Normalized calibration is mapped to each source video's original resolution.
- A vehicle's bottom-center is the primary lane anchor; its center may be supporting
  crossing evidence.
- Temporal state is bounded per track and tolerates configurable missing observations.
- Thresholds and quality gates are injected from validated configuration.
- Existing Milestone 1 object, track, device, and inference behavior stays compatible.

## Implementation slices

### 1. Lane and geometry foundation

- Serializable lane boundary, region, geometry, provenance, and unknown status.
- Configured normalized ego-lane polygon and resolution mapping.
- Signed boundary distance and outside/boundary/inside/unknown membership.
- Geometry, mapping, margin-jitter, serialization, and configuration tests.

### 2. Ego-motion baseline

- `EgoMotionEstimator` protocol and serializable transform/quality result.
- OpenCV feature tracking, optical flow, RANSAC homography, and vehicle-box masks.
- Invalid/unknown quality paths and synthetic camera-motion tests.

Implemented as a previous-frame-to-current-frame homography over background features.
Feature count, tracked count, RANSAC inliers, inlier ratio, reprojection error, and a
bounded confidence are preserved as structured quality evidence. Vehicle boxes are
excluded before feature detection with configurable padding. Failed input, tracking,
homography, or quality gates return `unknown` without a transform.

### 3. Temporal membership and lane-change state machine

- Smoothing, hysteresis, debounce, minimum duration, and occlusion tolerance.
- Adjacent, approaching, crossing, entered, candidate, confirmed, and rejected states.
- Tests for jitter, stable adjacent traffic, crossing, missing frames, and timestamps.

Implemented as bounded per-track state with a median signed-distance window, stable-phase
debounce, separate approaching and entered thresholds, candidate timeout, and configurable
missing-observation tolerance. Ego-motion `unknown` observations cannot advance or confirm
a candidate. Confirmation requires both consecutive entered observations and elapsed-time
evidence; rejected tracks can re-arm after returning to a stable adjacent state.

### 4. Cut-in events and structured evidence

- Configured forward corridor and explainable interaction heuristics.
- `LaneChangeEvent`, `CutInEvent`, evidence frames, and confidence breakdown.
- Tests for true cut-in, non-corridor lane change, and insufficient evidence.

Implemented with a configured normalized forward-corridor polygon mapped to original-frame
coordinates. Bottom-center corridor interaction is primary; bbox center is supporting
evidence. Cut-in confidence exposes lane-change, corridor, bbox-expansion, and ego-motion
components. Confirmation requires a confirmed temporal lane change, bottom-center corridor
interaction, sufficient image-space bbox expansion, motion quality, and an overall threshold.
These heuristics do not represent physical distance or TTC.

### 5. Pipeline and visualization integration

- Per-frame scene analysis in the existing streaming analyzer.
- Real `events.json`, optional frame analysis fields, and bounded state.
- Lane, membership, trajectory, candidate, and confirmed-event annotation.
- Deterministic fake-pipeline and synthetic OpenCV integration tests.

Implemented as an optional `StreamingSceneAnalyzer` injected into the existing video
analyzer. It carries only bounded previous-frame, per-track, and latest-event state. Frame
JSONL records can include lane geometry, ego motion, corridor, membership, temporal state,
and event snapshots; `events.json` stores deduplicated latest event states. Missing-track
rejection cascades to linked cut-in candidates. Annotation now draws lane geometry, forward
corridor, membership/status labels, trajectories, and event banners while the legacy
perception-only analyzer path remains supported.

Event lifecycle finalization rejects candidates that remain incomplete at end-of-video with
an explicit reason. Confirmed and rejected events are terminal snapshots: their end frame,
timestamp, confidence, and evidence are frozen instead of being overwritten by later frames.
Track overlays use compact ID/class labels, reserve a second line for actionable states, and
place text on dark collision-aware backgrounds so crowded distant detections remain legible.

### 6. Regression and documentation

- Full pytest, Ruff, and strict Mypy verification.
- Milestone 1 CPU/MPS/CUDA behavior and artifact compatibility checks.
- README usage, calibration, limitations, and acceptance notes.

Completed with a full automated regression, explicit Milestone 1 perception-only artifact
coverage, and validation that default, macOS, and NVIDIA configurations contain the complete
Milestone 2 sections. README now documents event states, calibration order, visualization,
limitations, and device-specific validation status.

The Apple Silicon MPS acceptance run processed all 625 frames of `samples/test1.mp4`. After
ego-lane calibration, Track 6 no longer produced a confirmed lane change, and end-of-video
candidates were finalized as rejected. The resulting event artifact contained 24 rejected,
zero candidate, and zero confirmed events. Synthetic integration coverage verifies the
positive confirmation path; a real positive lane-change/cut-in clip remains an explicit
acceptance gap. CUDA configuration and unavailable-device behavior are covered automatically,
but Milestone 2 has not yet been run on the target NVIDIA hardware.

## Verification gates

Each slice must pass its focused tests plus the existing test suite. The final gate is:

```text
pytest
ruff check .
mypy src
```

Core automated tests require no GPU, model weights, or network connection.

Cross-platform evidence for this milestone is governed by the repository-root `AGENTS.md` and
stored under `validation/`. Run `dashcam-ai milestone-status --milestone 2` to compare current
Mac MPS and Windows CUDA reports with the checked-out source commit. A result from either platform
does not prove the other platform, and missing or stale evidence keeps the cross-platform verdict
blocked.

## Progress

- [x] Milestone 2 architecture inspected and approved.
- [x] Slice 1: lane and geometry foundation.
- [x] Slice 2: ego-motion baseline.
- [x] Slice 3: temporal membership and lane-change state machine.
- [x] Slice 4: cut-in events and evidence.
- [x] Slice 5: pipeline and visualization integration.
- [x] Slice 6: regression and documentation.

## Risks

- Fixed calibration is intentionally limited on curves, hills, and changing camera pose;
  provenance, confidence, and unknown status make that limitation explicit.
- Homography can fail in low texture, darkness, rain, or highly dynamic scenes; quality
  gates must prevent confirmation when background evidence is insufficient.
- Track ID switches can fragment temporal evidence; Milestone 2 mitigates rather than
  attempts full re-identification.
- Image-space corridor interaction is not physical distance or legal TTC.

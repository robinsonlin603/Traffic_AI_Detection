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

### 5. Pipeline and visualization integration

- Per-frame scene analysis in the existing streaming analyzer.
- Real `events.json`, optional frame analysis fields, and bounded state.
- Lane, membership, trajectory, candidate, and confirmed-event annotation.
- Deterministic fake-pipeline and synthetic OpenCV integration tests.

### 6. Regression and documentation

- Full pytest, Ruff, and strict Mypy verification.
- Milestone 1 CPU/MPS/CUDA behavior and artifact compatibility checks.
- README usage, calibration, limitations, and acceptance notes.

## Verification gates

Each slice must pass its focused tests plus the existing test suite. The final gate is:

```text
pytest
ruff check .
mypy src
```

Core automated tests require no GPU, model weights, or network connection.

## Progress

- [x] Milestone 2 architecture inspected and approved.
- [x] Slice 1: lane and geometry foundation.
- [x] Slice 2: ego-motion baseline.
- [x] Slice 3: temporal membership and lane-change state machine.
- [ ] Slice 4: cut-in events and evidence.
- [ ] Slice 5: pipeline and visualization integration.
- [ ] Slice 6: regression and documentation.

## Risks

- Fixed calibration is intentionally limited on curves, hills, and changing camera pose;
  provenance, confidence, and unknown status make that limitation explicit.
- Homography can fail in low texture, darkness, rain, or highly dynamic scenes; quality
  gates must prevent confirmation when background evidence is insufficient.
- Track ID switches can fragment temporal evidence; Milestone 2 mitigates rather than
  attempts full re-identification.
- Image-space corridor interaction is not physical distance or legal TTC.

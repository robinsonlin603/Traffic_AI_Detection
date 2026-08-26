# Motorcycle Dashcam AI — Execution Plan

This living plan implements the approved Milestone 0 and the thin vertical slice of
Milestone 1. The acceptance target is:

```text
MP4 -> normalized detections -> persistent BoT-SORT tracks
    -> track histories -> structured artifacts -> annotated MP4
```

## Scope

- Python package, configuration, domain models, protocols, and structured logging.
- Original-resolution video reading and reversible inference coordinate mapping.
- Deterministic fake adapters plus Ultralytics YOLO/BoT-SORT integration.
- Metadata, JSONL frame records, track summaries, and optional annotated video.
- CLI and GPU-independent tests.

Lane geometry, cut-in detection, turn-signal analysis, FastAPI, GPS, and LLM/VLM
features are explicitly deferred.

## Milestones

1. Establish package and library-independent domain boundaries.
2. Build a deterministic pipeline using fake adapters.
3. Add OpenCV video I/O and Ultralytics adapter behind lazy imports.
4. Add artifact storage, annotation, and CLI.
5. Verify unit tests, lint, typing, and available integration behavior.

## Decisions

- Original-frame coordinates are canonical.
- Letterbox transforms are explicit and reversible.
- Frame records stream to JSONL; summaries remain JSON.
- Heavy CV dependencies are optional so the core test suite stays CPU-only.
- The Ultralytics adapter uses `model.track(..., tracker="botsort.yaml")`, then
  immediately converts results into domain objects.

## Verification

- Geometry and coordinate mapping unit tests.
- Model serialization and track-history tests.
- Deterministic fake-pipeline integration test.
- Optional real-video smoke test when OpenCV and a source video are available.

## Progress

- [x] Requirements and architecture approved.
- [x] Package and domain foundation.
- [x] Deterministic pipeline.
- [x] CV adapters, artifacts, annotation, and CLI.
- [x] Automated verification.

## Outcome

Milestone 0 and the approved Milestone 1 vertical slice are implemented. Core tests
run without model weights or CUDA. An OpenCV integration test generates a temporary
MP4 and verifies annotated-video output. Real YOLO inference remains dependent on a
user-supplied or downloaded model and a representative source video.

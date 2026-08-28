# Cross-platform validation evidence

These files are durable project context shared through Git. JSON is canonical; Markdown is a
generated human-readable view. Each platform owns its stable filename, and Git history retains
earlier runs.

- `macos-mps` proves only macOS MPS behavior for its exact `source_commit`.
- `linux-cuda` proves only Linux CUDA behavior for its exact `source_commit`.
- `cpu` proves only CPU behavior for its exact `source_commit`.

Never infer one platform from another. A report is stale after source changes, and a dirty run
cannot pass authoritatively. Do not hand-edit generated reports or commit videos, model weights,
private paths, credentials, or large logs.

Generate and inspect evidence from the repository root:

    dashcam-ai validate --milestone 2 --platform linux-cuda
    dashcam-ai validation-status validation/milestone-2/linux-cuda.json
    dashcam-ai milestone-status --milestone 2

The validation command does not commit or push. Review `git diff -- validation/` before publishing
the platform-owned JSON and Markdown pair.

"""建立可重現影片分析所需的執行環境資訊。"""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dashcam_ai.domain.runtime import RuntimeMetadata
from dashcam_ai.runtime.device import DeviceResolution


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _model_sha256(model: str) -> str | None:
    path = Path(model)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_runtime_metadata(
    resolution: DeviceResolution,
    model: str,
    imgsz: int,
    confidence: float,
    torch_module: Any,
) -> RuntimeMetadata:
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    return RuntimeMetadata(
        requested_device=resolution.requested,
        resolved_device=resolution.resolved,
        device_name=resolution.name,
        python_version=platform.python_version(),
        torch_version=str(torch_module.__version__),
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        ultralytics_version=_version("ultralytics"),
        opencv_version=_version("opencv-python"),
        model=model,
        model_sha256=_model_sha256(model),
        imgsz=imgsz,
        confidence=confidence,
    )

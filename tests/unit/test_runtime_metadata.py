from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from dashcam_ai.runtime.device import DeviceResolution
from dashcam_ai.runtime.metadata import build_runtime_metadata


def test_runtime_metadata_includes_device_versions_and_model_hash(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"test-model")
    torch = SimpleNamespace(__version__="2.7.0", version=SimpleNamespace(cuda="12.8"))

    metadata = build_runtime_metadata(
        resolution=DeviceResolution(requested="cuda", resolved="cuda:0", name="RTX 4070 SUPER"),
        model=str(model),
        imgsz=1280,
        confidence=0.35,
        torch_module=torch,
    )

    assert metadata.resolved_device == "cuda:0"
    assert metadata.device_name == "RTX 4070 SUPER"
    assert metadata.torch_version == "2.7.0"
    assert metadata.cuda_version == "12.8"
    assert metadata.model_sha256 == sha256(b"test-model").hexdigest()


def test_named_model_without_local_file_has_no_hash() -> None:
    torch = SimpleNamespace(__version__="2.7.0", version=SimpleNamespace(cuda=None))
    metadata = build_runtime_metadata(
        DeviceResolution(requested="cpu", resolved="cpu", name="CPU"),
        "named-model.pt",
        640,
        0.5,
        torch,
    )
    assert metadata.model_sha256 is None

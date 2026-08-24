from __future__ import annotations

from types import SimpleNamespace

import pytest

from dashcam_ai.runtime.device import DeviceUnavailableError, inspect_devices, resolve_device


class FakeCuda:
    def __init__(self, available: bool, names: list[str] | None = None) -> None:
        self.available = available
        self.names = names or []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return len(self.names)

    def get_device_name(self, index: int) -> str:
        return self.names[index]


class FakeMps:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def fake_torch(cuda: bool = False, mps: bool = False, names: list[str] | None = None):
    return SimpleNamespace(
        cuda=FakeCuda(cuda, names),
        backends=SimpleNamespace(mps=FakeMps(mps)),
    )


def test_auto_prefers_cuda_over_mps() -> None:
    result = resolve_device("auto", fake_torch(cuda=True, mps=True, names=["RTX 4070 SUPER"]))
    assert result.resolved == "cuda:0"
    assert result.name == "RTX 4070 SUPER"


def test_auto_uses_mps_when_cuda_is_unavailable() -> None:
    result = resolve_device("auto", fake_torch(mps=True))
    assert result.resolved == "mps"


def test_auto_falls_back_to_cpu() -> None:
    assert resolve_device("auto", fake_torch()).resolved == "cpu"


def test_cuda_alias_resolves_first_device() -> None:
    result = resolve_device("cuda", fake_torch(cuda=True, names=["RTX 4070 SUPER"]))
    assert result.resolved == "cuda:0"


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:abc", "gpu", "mps:0"])
def test_invalid_device_is_rejected(device: str) -> None:
    with pytest.raises(DeviceUnavailableError):
        resolve_device(device, fake_torch())


def test_unavailable_mps_has_clear_error() -> None:
    with pytest.raises(DeviceUnavailableError, match="MPS"):
        resolve_device("mps", fake_torch())


def test_unavailable_cuda_has_clear_error() -> None:
    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        resolve_device("cuda:0", fake_torch())


def test_device_inspection_lists_all_backends() -> None:
    devices = inspect_devices(fake_torch(cuda=True, mps=True, names=["RTX 4070 SUPER"]))
    assert [item.device for item in devices] == ["cpu", "mps", "cuda:0"]
    assert all(item.available for item in devices)

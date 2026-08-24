"""解析 CPU、Apple MPS 與 NVIDIA CUDA 推論裝置。"""

from __future__ import annotations

import platform
from typing import Any

from pydantic import BaseModel, ConfigDict


class DeviceUnavailableError(ValueError):
    """指定的推論裝置不存在或目前無法使用。"""


class DeviceResolution(BaseModel):
    model_config = ConfigDict(frozen=True)
    requested: str
    resolved: str
    name: str


class DeviceAvailability(BaseModel):
    model_config = ConfigDict(frozen=True)
    device: str
    available: bool
    name: str | None = None
    reason: str | None = None


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DeviceUnavailableError("找不到 PyTorch；請安裝專案的 cv 相依套件。") from error
    return torch


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "CPU"


def resolve_device(
    requested: str | None = "auto", torch_module: Any | None = None
) -> DeviceResolution:
    """依照 CUDA、MPS、CPU 優先順序解析並驗證裝置。"""
    value = (requested or "auto").strip().lower()
    torch = torch_module or _torch()
    if value == "auto":
        if torch.cuda.is_available():
            return DeviceResolution(
                requested="auto", resolved="cuda:0", name=str(torch.cuda.get_device_name(0))
            )
        if torch.backends.mps.is_available():
            return DeviceResolution(
                requested="auto", resolved="mps", name="Apple Metal Performance Shaders"
            )
        return DeviceResolution(requested="auto", resolved="cpu", name=_cpu_name())
    if value == "cpu":
        return DeviceResolution(requested=value, resolved="cpu", name=_cpu_name())
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceUnavailableError("此電腦目前無法使用 MPS；請改用 auto 或 cpu。")
        return DeviceResolution(
            requested=value, resolved="mps", name="Apple Metal Performance Shaders"
        )
    original = value
    if value == "cuda":
        value = "cuda:0"
    if value.startswith("cuda:"):
        suffix = value.removeprefix("cuda:")
        if not suffix.isdigit():
            raise DeviceUnavailableError("CUDA 裝置格式必須是 cuda 或 cuda:<非負整數>。")
        index = int(suffix)
        if not torch.cuda.is_available():
            raise DeviceUnavailableError("此電腦目前無法使用 CUDA；Mac 請使用 mps 或 cpu。")
        count = int(torch.cuda.device_count())
        if index >= count:
            raise DeviceUnavailableError(f"找不到 cuda:{index}；目前只有 {count} 個 CUDA 裝置。")
        return DeviceResolution(
            requested=original,
            resolved=f"cuda:{index}",
            name=str(torch.cuda.get_device_name(index)),
        )
    raise DeviceUnavailableError("不支援的裝置；請使用 auto、cpu、mps、cuda 或 cuda:<索引>。")


def inspect_devices(torch_module: Any | None = None) -> list[DeviceAvailability]:
    """列出目前主機可供推論使用的裝置。"""
    try:
        torch = torch_module or _torch()
    except DeviceUnavailableError as error:
        return [
            DeviceAvailability(device="cpu", available=True, name=_cpu_name()),
            DeviceAvailability(device="mps", available=False, reason=str(error)),
            DeviceAvailability(device="cuda:0", available=False, reason=str(error)),
        ]
    devices = [DeviceAvailability(device="cpu", available=True, name=_cpu_name())]
    mps_available = bool(torch.backends.mps.is_available())
    devices.append(
        DeviceAvailability(
            device="mps",
            available=mps_available,
            name="Apple Metal Performance Shaders" if mps_available else None,
            reason=None if mps_available else "MPS 不可用",
        )
    )
    if torch.cuda.is_available():
        devices.extend(
            DeviceAvailability(
                device=f"cuda:{index}", available=True, name=str(torch.cuda.get_device_name(index))
            )
            for index in range(int(torch.cuda.device_count()))
        )
    else:
        devices.append(DeviceAvailability(device="cuda:0", available=False, reason="CUDA 不可用"))
    return devices

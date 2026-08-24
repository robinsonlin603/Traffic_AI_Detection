"""定義分析環境與模型的可重現性資訊。"""

from pydantic import BaseModel, ConfigDict, Field


class RuntimeMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    requested_device: str
    resolved_device: str
    device_name: str
    python_version: str
    torch_version: str | None = None
    cuda_version: str | None = None
    ultralytics_version: str | None = None
    opencv_version: str | None = None
    model: str
    model_sha256: str | None = None
    imgsz: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

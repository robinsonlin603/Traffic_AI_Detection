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
    minimum_vehicle_area_ratio: float = Field(default=0, ge=0, le=1)
    duplicate_vehicle_iou_threshold: float = Field(default=0.85, ge=0, le=1)
    duplicate_vehicle_containment_threshold: float = Field(default=0.9, ge=0, le=1)
    duplicate_vehicle_center_distance_ratio: float = Field(default=0.22, ge=0, le=1)

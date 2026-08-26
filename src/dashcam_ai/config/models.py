"""定義應用程式設定模型，並從 YAML 載入及驗證設定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from dashcam_ai.domain.lane import NormalizedPoint2D


class DetectionConfig(BaseModel):
    """物件偵測模型、推論門檻與目標類別設定。"""
    model: str = "yolo26m.pt"
    confidence: float = Field(default=0.35, ge=0, le=1)
    imgsz: int = Field(default=1280, gt=0)
    device: str = "auto"
    classes: list[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck", "person", "bicycle"]
    )


class TrackingConfig(BaseModel):
    """物件追蹤器及有效軌跡長度設定。"""
    tracker: str = "botsort.yaml"
    minimum_track_length: int = Field(default=2, gt=0)


class LaneGeometryConfig(BaseModel):
    """人工校正的 normalized 自車車道與其可信度。"""

    enabled: bool = True
    ego_lane_polygon: list[NormalizedPoint2D] = Field(
        default_factory=lambda: [
            NormalizedPoint2D(x=0.44, y=0.45),
            NormalizedPoint2D(x=0.56, y=0.45),
            NormalizedPoint2D(x=0.90, y=1.00),
            NormalizedPoint2D(x=0.10, y=1.00),
        ],
        min_length=4,
        max_length=4,
    )
    confidence: float = Field(default=1.0, ge=0, le=1)


class LaneMembershipConfig(BaseModel):
    """車道邊界帶設定；時間平滑參數於後續 slice 加入。"""

    boundary_margin_pixels: float = Field(default=12.0, ge=0)


class OutputConfig(BaseModel):
    """分析結果與標註影片的輸出設定。"""
    save_video: bool = True
    save_frames: bool = True
    codec: str = Field(default="mp4v", min_length=4, max_length=4)


class LoggingConfig(BaseModel):
    """應用程式日誌等級設定。"""
    level: str = "INFO"


class AppConfig(BaseModel):
    """彙整所有設定區段的頂層模型。"""
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    lane_geometry: LaneGeometryConfig = Field(default_factory=LaneGeometryConfig)
    lane_membership: LaneMembershipConfig = Field(default_factory=LaneMembershipConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> AppConfig:
    """讀取 YAML 設定檔，並轉成經 Pydantic 驗證的設定物件。"""
    with path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)

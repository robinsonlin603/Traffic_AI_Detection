"""定義應用程式設定模型，並從 YAML 載入及驗證設定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

from dashcam_ai.detection.ego_mask import validate_ego_polygon


class DetectionConfig(BaseModel):
    """物件偵測模型、推論門檻與目標類別設定。"""

    model: str = "yolo26m.pt"
    confidence: float = Field(default=0.35, ge=0, le=1)
    imgsz: int = Field(default=1280, gt=0)
    device: str = "auto"
    minimum_vehicle_area_ratio: float = Field(default=0, ge=0, le=1)
    duplicate_vehicle_iou_threshold: float = Field(default=0.85, ge=0, le=1)
    duplicate_vehicle_containment_threshold: float = Field(default=0.9, ge=0, le=1)
    duplicate_vehicle_center_distance_ratio: float = Field(default=0.22, ge=0, le=1)
    ego_vehicle_polygon: list[tuple[float, float]] = Field(default_factory=list)
    ego_vehicle_overlap_threshold: float = Field(default=0.8, gt=0, le=1)
    classes: list[str] = Field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck"]
    )

    @field_validator("ego_vehicle_polygon")
    @classmethod
    def validate_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return validate_ego_polygon(value)


class TrackingConfig(BaseModel):
    """物件追蹤器及有效軌跡長度設定。"""

    tracker: str = "botsort.yaml"
    minimum_track_length: int = Field(default=2, gt=0)


class LaneDetectionConfig(BaseModel):
    """Milestone 2 模型式道路白線分割與時間平滑設定。"""

    enabled: bool = True
    backend: str = "yolop_onnx"
    model: str = "models/yolop-640-640.onnx"
    model_sha256: str = ""
    input_size: int = Field(default=640, gt=0)
    probability_threshold: float = Field(default=0.5, gt=0, lt=1)
    minimum_drivable_probability: float = Field(default=0.5, ge=0, le=1)
    white_lightness_threshold: int = Field(default=90, ge=0, le=255)
    white_saturation_threshold: int = Field(default=125, ge=0, le=255)
    local_contrast_threshold: int = Field(default=12, ge=0, le=255)
    strong_probability_threshold: float = Field(default=0.6, gt=0, lt=1)
    roi_top_ratio: float = Field(default=0.4, gt=0, lt=1)
    minimum_component_area_ratio: float = Field(default=0.00004, gt=0, lt=1)
    minimum_vertical_span_ratio: float = Field(default=0.1, gt=0, lt=1)
    minimum_fragment_vertical_span_ratio: float = Field(default=0.05, gt=0, lt=1)
    maximum_horizontal_to_vertical_ratio: float = Field(default=5.0, gt=0)
    maximum_fit_error_ratio: float = Field(default=0.025, gt=0, lt=1)
    maximum_arrow_fit_error_ratio: float = Field(default=0.008, gt=0, lt=1)
    maximum_row_width_ratio: float = Field(default=0.08, gt=0, lt=1)
    maximum_row_width_variation_ratio: float = Field(default=3.5, gt=1)
    maximum_fragment_gap_ratio: float = Field(default=0.12, gt=0, lt=1)
    sample_count: int = Field(default=24, ge=2)
    smoothing_alpha: float = Field(default=0.35, gt=0, lt=1)
    maximum_missing_frames: int = Field(default=1, ge=0)
    minimum_curve_confidence: float = Field(default=0.5, ge=0, le=1)
    maximum_boundaries: int = Field(default=8, gt=0)
    temporal_association_distance_ratio: float = Field(default=0.08, gt=0, lt=1)
    excluded_bbox_margin_ratio: float = Field(default=0.02, gt=0, lt=1)
    minimum_confirmation_frames: int = Field(default=2, gt=0)


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
    lane_detection: LaneDetectionConfig = Field(default_factory=LaneDetectionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> AppConfig:
    """讀取 YAML 設定檔，並轉成經 Pydantic 驗證的設定物件。"""
    with path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)

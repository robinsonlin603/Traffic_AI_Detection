"""定義應用程式設定模型，並從 YAML 載入及驗證設定。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

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


class EgoMotionConfig(BaseModel):
    """OpenCV 背景特徵追蹤及 homography 品質門檻。"""

    max_features: int = Field(default=500, gt=0)
    feature_quality_level: float = Field(default=0.01, gt=0, le=1)
    feature_min_distance: float = Field(default=8.0, ge=0)
    optical_flow_window_size: int = Field(default=21, gt=0)
    optical_flow_max_level: int = Field(default=3, ge=0)
    ransac_reprojection_threshold: float = Field(default=3.0, gt=0)
    minimum_tracked_features: int = Field(default=12, ge=4)
    minimum_inliers: int = Field(default=8, ge=4)
    minimum_inlier_ratio: float = Field(default=0.5, ge=0, le=1)
    maximum_mean_reprojection_error: float = Field(default=2.5, ge=0)
    mask_padding_pixels: int = Field(default=8, ge=0)

    @field_validator("optical_flow_window_size")
    @classmethod
    def validate_odd_window_size(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("optical_flow_window_size must be odd")
        return value


class TemporalLaneConfig(BaseModel):
    """車道歸屬時間平滑、遲滯、缺失容忍與確認門檻。"""

    smoothing_window_frames: int = Field(default=3, gt=0)
    approaching_distance_pixels: float = Field(default=40.0, gt=0)
    entered_distance_pixels: float = Field(default=20.0, gt=0)
    debounce_frames: int = Field(default=2, gt=0)
    minimum_confirmation_frames: int = Field(default=3, gt=0)
    minimum_confirmation_duration_seconds: float = Field(default=0.1, ge=0)
    maximum_missing_frames: int = Field(default=2, ge=0)
    candidate_timeout_seconds: float = Field(default=2.0, gt=0)
    history_size: int = Field(default=30, gt=0)


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
    ego_motion: EgoMotionConfig = Field(default_factory=EgoMotionConfig)
    temporal_lane: TemporalLaneConfig = Field(default_factory=TemporalLaneConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: Path) -> AppConfig:
    """讀取 YAML 設定檔，並轉成經 Pydantic 驗證的設定物件。"""
    with path.open("r", encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)

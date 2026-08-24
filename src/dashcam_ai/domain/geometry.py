"""定義影像座標、邊界框及推論尺寸轉換模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Point2D(BaseModel):
    """影像座標系中的二維點。"""
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class BBox(BaseModel):
    """以左上角與右下角座標表示的矩形邊界框。"""
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_extents(self) -> BBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bbox maximum coordinates must not be smaller than minimums")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> Point2D:
        return Point2D(x=(self.x1 + self.x2) / 2, y=self.y2)

    def clip(self, width: int, height: int) -> BBox:
        """將邊界框限制在指定影像尺寸內。"""
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        return BBox(
            x1=min(max(self.x1, 0.0), float(width)),
            y1=min(max(self.y1, 0.0), float(height)),
            x2=min(max(self.x2, 0.0), float(width)),
            y2=min(max(self.y2, 0.0), float(height)),
        )

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


class FrameTransform(BaseModel):
    """在原始影像與 letterbox 推論影像之間進行可逆座標轉換。"""

    model_config = ConfigDict(frozen=True)

    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    inference_width: int = Field(gt=0)
    inference_height: int = Field(gt=0)
    scale: float = Field(gt=0)
    pad_x: float = Field(ge=0)
    pad_y: float = Field(ge=0)

    @classmethod
    def letterbox(
        cls, original_width: int, original_height: int, target_width: int, target_height: int
    ) -> FrameTransform:
        """計算保持長寬比縮放時的比例與兩側補邊量。"""
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = original_width * scale
        resized_height = original_height * scale
        return cls(
            original_width=original_width,
            original_height=original_height,
            inference_width=target_width,
            inference_height=target_height,
            scale=scale,
            pad_x=(target_width - resized_width) / 2,
            pad_y=(target_height - resized_height) / 2,
        )

    def to_inference_bbox(self, bbox: BBox) -> BBox:
        """將原始影像的邊界框轉換至推論影像座標。"""
        return BBox(
            x1=bbox.x1 * self.scale + self.pad_x,
            y1=bbox.y1 * self.scale + self.pad_y,
            x2=bbox.x2 * self.scale + self.pad_x,
            y2=bbox.y2 * self.scale + self.pad_y,
        )

    def to_original_bbox(self, bbox: BBox) -> BBox:
        """將推論座標還原至原始影像，並裁切超出範圍的部分。"""
        return BBox(
            x1=(bbox.x1 - self.pad_x) / self.scale,
            y1=(bbox.y1 - self.pad_y) / self.scale,
            x2=(bbox.x2 - self.pad_x) / self.scale,
            y2=(bbox.y2 - self.pad_y) / self.scale,
        ).clip(self.original_width, self.original_height)

"""整合 Ultralytics YOLO 偵測與持續性物件追蹤。"""

from __future__ import annotations

from typing import Any

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.runtime.device import resolve_device
from dashcam_ai.runtime.metadata import build_runtime_metadata


class UltralyticsDetectorTracker:
    """執行 YOLO 偵測及持續性 BoT-SORT 追蹤的轉接器。"""

    def __init__(
        self,
        model: str,
        confidence: float,
        imgsz: int,
        class_names: list[str],
        tracker: str = "botsort.yaml",
        device: str | None = "auto",
    ) -> None:
        try:
            import torch
            from ultralytics import YOLO  # type: ignore[attr-defined]
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for YOLO/BoT-SORT. Install the 'cv' dependency group."
            ) from error
        resolution = resolve_device(device, torch)
        self._model = YOLO(model)
        names = self._model.names
        # 設定檔以易讀的類別名稱表示，推論 API 則需要數字 class ID。
        self._allowed_class_ids = [
            int(class_id) for class_id, name in names.items() if name in set(class_names)
        ]
        self._confidence = confidence
        self._imgsz = imgsz
        self._tracker = tracker
        self._device = resolution.resolved
        self.runtime_metadata = build_runtime_metadata(
            resolution=resolution,
            model=model,
            imgsz=imgsz,
            confidence=confidence,
            torch_module=torch,
        )

    def process(self, frame: Any) -> list[TrackedObject]:
        """分析單一影格，並將 Ultralytics 結果轉為專案領域模型。"""
        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self._tracker,
            conf=self._confidence,
            imgsz=self._imgsz,
            classes=self._allowed_class_ids,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        ids_tensor: Any = boxes.id
        classes_tensor: Any = boxes.cls
        confidence_tensor: Any = boxes.conf
        coordinates_tensor: Any = boxes.xyxy
        track_ids = ids_tensor.int().cpu().tolist()
        class_ids = classes_tensor.int().cpu().tolist()
        confidences = confidence_tensor.cpu().tolist()
        coordinates = coordinates_tensor.cpu().tolist()
        names = results[0].names
        # 在邊界完成格式轉換，避免應用層依賴 Ultralytics 的 tensor 型別。
        return [
            TrackedObject(
                track_id=track_id,
                class_id=class_id,
                class_name=str(names[class_id]),
                confidence=float(confidence),
                bbox=BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
            )
            for track_id, class_id, confidence, xyxy in zip(
                track_ids, class_ids, confidences, coordinates, strict=True
            )
        ]

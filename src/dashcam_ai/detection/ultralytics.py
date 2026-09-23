"""整合 Ultralytics YOLO 偵測與持續性物件追蹤。"""

from __future__ import annotations

from typing import Any

from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import TrackedObject
from dashcam_ai.runtime.device import resolve_device
from dashcam_ai.runtime.metadata import build_runtime_metadata
from dashcam_ai.tracking.identity import VehicleIdentityResolver

POWERED_VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}
VEHICLE_CLASS_NAME = "vehicle"


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
        minimum_vehicle_area_ratio: float = 0,
        duplicate_vehicle_iou_threshold: float = 0.85,
        duplicate_vehicle_containment_threshold: float = 0.9,
        duplicate_vehicle_center_distance_ratio: float = 0.22,
    ) -> None:
        if not 0 <= minimum_vehicle_area_ratio <= 1:
            raise ValueError("minimum_vehicle_area_ratio must be between 0 and 1")
        if not 0 <= duplicate_vehicle_iou_threshold <= 1:
            raise ValueError("duplicate_vehicle_iou_threshold must be between 0 and 1")
        if not 0 <= duplicate_vehicle_containment_threshold <= 1:
            raise ValueError("duplicate_vehicle_containment_threshold must be between 0 and 1")
        if not 0 <= duplicate_vehicle_center_distance_ratio <= 1:
            raise ValueError("duplicate_vehicle_center_distance_ratio must be between 0 and 1")
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
        self._minimum_vehicle_area_ratio = minimum_vehicle_area_ratio
        self._duplicate_vehicle_iou_threshold = duplicate_vehicle_iou_threshold
        self._duplicate_vehicle_containment_threshold = duplicate_vehicle_containment_threshold
        self._duplicate_vehicle_center_distance_ratio = duplicate_vehicle_center_distance_ratio
        self._vehicle_class_ids = {
            int(class_id) for class_id, name in names.items() if name in POWERED_VEHICLE_NAMES
        }
        self._vehicle_class_id = max(int(class_id) for class_id in names) + 1
        # Register before track() appends its tracker callback. Detections are filtered,
        # deduplicated and normalized before they can consume track IDs.
        self._model.add_callback("on_predict_postprocess_end", self._prepare_vehicle_detections)
        self._allowed_class_ids = [
            int(class_id) for class_id, name in names.items() if name in set(class_names)
        ]
        self._confidence = confidence
        self._imgsz = imgsz
        self._tracker = tracker
        self._identity_resolver = VehicleIdentityResolver(
            duplicate_iou_threshold=duplicate_vehicle_iou_threshold,
            duplicate_containment_threshold=duplicate_vehicle_containment_threshold,
            duplicate_center_distance_ratio=duplicate_vehicle_center_distance_ratio,
        )
        self._device = resolution.resolved
        self.runtime_metadata = build_runtime_metadata(
            resolution=resolution,
            model=model,
            imgsz=imgsz,
            confidence=confidence,
            torch_module=torch,
        ).model_copy(
            update={
                "minimum_vehicle_area_ratio": minimum_vehicle_area_ratio,
                "duplicate_vehicle_iou_threshold": duplicate_vehicle_iou_threshold,
                "duplicate_vehicle_containment_threshold": (
                    duplicate_vehicle_containment_threshold
                ),
                "duplicate_vehicle_center_distance_ratio": (
                    duplicate_vehicle_center_distance_ratio
                ),
            }
        )

    def _prepare_vehicle_detections(self, predictor: Any) -> None:
        """Filter, class-agnostically deduplicate, then normalize powered vehicles."""
        for index, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            vehicle = boxes.cls == -1
            for class_id in self._vehicle_class_ids:
                vehicle |= boxes.cls == class_id
            keep = boxes.conf < 0
            if self._minimum_vehicle_area_ratio > 0:
                height, width = result.orig_shape
                xyxy = boxes.xyxy
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                vehicle &= areas >= self._minimum_vehicle_area_ratio * width * height
            selected: list[int] = []
            for candidate in boxes.conf.argsort(descending=True).cpu().tolist():
                if not bool(vehicle[candidate]):
                    continue
                candidate_box = boxes.xyxy[candidate]
                if all(
                    not self._boxes_are_duplicates(candidate_box, boxes.xyxy[accepted])
                    for accepted in selected
                ):
                    selected.append(candidate)
                    keep[candidate] = True
            prepared = result[keep]
            if prepared.boxes is not None and len(prepared.boxes) > 0:
                prepared.boxes.data[:, 5] = self._vehicle_class_id
                prepared.names = {**prepared.names, self._vehicle_class_id: VEHICLE_CLASS_NAME}
            predictor.results[index] = prepared

    def _boxes_are_duplicates(self, first: Any, second: Any) -> bool:
        intersection_width = max(0.0, float(min(first[2], second[2]) - max(first[0], second[0])))
        intersection_height = max(0.0, float(min(first[3], second[3]) - max(first[1], second[1])))
        intersection = intersection_width * intersection_height
        first_width = float(first[2] - first[0])
        first_height = float(first[3] - first[1])
        second_width = float(second[2] - second[0])
        second_height = float(second[3] - second[1])
        first_area = first_width * first_height
        second_area = second_width * second_height
        union = first_area + second_area - intersection
        iou = intersection / union if union > 0 else 0.0
        if iou >= self._duplicate_vehicle_iou_threshold:
            return True
        smaller_area = min(first_area, second_area)
        containment = intersection / smaller_area if smaller_area > 0 else 0.0
        maximum_width = max(first_width, second_width)
        maximum_height = max(first_height, second_height)
        horizontal_center_distance = abs(float(first[0] + first[2] - second[0] - second[2]) / 2)
        vertical_center_distance = abs(float(first[1] + first[3] - second[1] - second[3]) / 2)
        return (
            containment >= self._duplicate_vehicle_containment_threshold
            and maximum_width > 0
            and maximum_height > 0
            and horizontal_center_distance / maximum_width
            < self._duplicate_vehicle_center_distance_ratio
            and vertical_center_distance / maximum_height
            < self._duplicate_vehicle_center_distance_ratio
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
            return self._identity_resolver.update([])
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return self._identity_resolver.update([])
        ids_tensor: Any = boxes.id
        classes_tensor: Any = boxes.cls
        confidence_tensor: Any = boxes.conf
        coordinates_tensor: Any = boxes.xyxy
        track_ids = ids_tensor.int().cpu().tolist()
        class_ids = classes_tensor.int().cpu().tolist()
        confidences = confidence_tensor.cpu().tolist()
        coordinates = coordinates_tensor.cpu().tolist()
        names = results[0].names
        tracked = [
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
        return self._identity_resolver.update(tracked)

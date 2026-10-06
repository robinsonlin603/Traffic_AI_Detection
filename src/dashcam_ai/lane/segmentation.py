"""使用 YOLOP 語意分割遮罩辨識道路白線。"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane_lines import (
    LaneCandidateDecision,
    LaneCurve,
    LaneLineDiagnostics,
    LaneLineFrame,
    LaneLineStatus,
)
from dashcam_ai.video.reader import _cv2


class YoloPLaneLineDetector:
    """將 YOLOP 車道線機率圖轉成可追蹤的影像座標曲線。"""

    def __init__(
        self,
        *,
        model: str,
        expected_sha256: str | None = None,
        input_size: int = 640,
        probability_threshold: float = 0.5,
        minimum_drivable_probability: float = 0.5,
        white_lightness_threshold: int = 90,
        white_saturation_threshold: int = 125,
        local_contrast_threshold: int = 12,
        strong_probability_threshold: float = 0.6,
        roi_top_ratio: float = 0.4,
        minimum_component_area_ratio: float = 0.00004,
        minimum_vertical_span_ratio: float = 0.1,
        minimum_fragment_vertical_span_ratio: float = 0.05,
        maximum_horizontal_to_vertical_ratio: float = 5.0,
        maximum_fit_error_ratio: float = 0.025,
        maximum_arrow_fit_error_ratio: float = 0.008,
        maximum_row_width_ratio: float = 0.08,
        maximum_row_width_variation_ratio: float = 3.5,
        maximum_fragment_gap_ratio: float = 0.12,
        sample_count: int = 24,
        smoothing_alpha: float = 0.35,
        maximum_missing_frames: int = 1,
        minimum_curve_confidence: float = 0.5,
        maximum_boundaries: int = 8,
        temporal_association_distance_ratio: float = 0.08,
        excluded_bbox_margin_ratio: float = 0.02,
        minimum_confirmation_frames: int = 2,
    ) -> None:
        model_path = Path(model)
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 YOLOP ONNX 模型：{model_path}")
        if expected_sha256:
            actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if actual_sha256.lower() != expected_sha256.lower():
                raise ValueError(
                    "YOLOP ONNX 模型 SHA-256 不符："
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        ratios = (
            probability_threshold,
            strong_probability_threshold,
            roi_top_ratio,
            minimum_component_area_ratio,
            minimum_vertical_span_ratio,
            minimum_fragment_vertical_span_ratio,
            maximum_fit_error_ratio,
            maximum_arrow_fit_error_ratio,
            maximum_row_width_ratio,
            maximum_fragment_gap_ratio,
            smoothing_alpha,
            temporal_association_distance_ratio,
            excluded_bbox_margin_ratio,
        )
        if any(not 0 < value < 1 for value in ratios):
            raise ValueError("lane segmentation ratios must be between zero and one")
        if (
            input_size <= 0
            or sample_count < 2
            or maximum_boundaries <= 0
            or maximum_horizontal_to_vertical_ratio <= 0
            or maximum_row_width_variation_ratio <= 1
        ):
            raise ValueError("invalid lane segmentation dimensions")
        if maximum_missing_frames < 0 or minimum_confirmation_frames <= 0:
            raise ValueError("invalid lane temporal settings")
        if not 0 <= white_lightness_threshold <= 255:
            raise ValueError("white lightness threshold must be between zero and 255")
        if not 0 <= white_saturation_threshold <= 255:
            raise ValueError("white saturation threshold must be between zero and 255")
        if not 0 <= local_contrast_threshold <= 255:
            raise ValueError("local contrast threshold must be between zero and 255")
        if minimum_fragment_vertical_span_ratio > minimum_vertical_span_ratio:
            raise ValueError("minimum fragment span cannot exceed minimum full span")
        if maximum_arrow_fit_error_ratio > maximum_fit_error_ratio:
            raise ValueError("arrow fit error cannot exceed general fit error")
        if not 0 <= minimum_curve_confidence <= 1:
            raise ValueError("minimum curve confidence must be between zero and one")
        if not 0 <= minimum_drivable_probability <= 1:
            raise ValueError("minimum drivable probability must be between zero and one")

        cv2 = _cv2()
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._input_size = input_size
        self._threshold = probability_threshold
        self._minimum_drivable = minimum_drivable_probability
        self._white_lightness = white_lightness_threshold
        self._white_saturation = white_saturation_threshold
        self._local_contrast = local_contrast_threshold
        self._strong_probability = strong_probability_threshold
        self._roi_top = roi_top_ratio
        self._minimum_area = minimum_component_area_ratio
        self._minimum_span = minimum_vertical_span_ratio
        self._minimum_fragment_span = minimum_fragment_vertical_span_ratio
        self._maximum_horizontal_to_vertical = maximum_horizontal_to_vertical_ratio
        self._maximum_fit_error = maximum_fit_error_ratio
        self._maximum_arrow_fit_error = maximum_arrow_fit_error_ratio
        self._maximum_row_width = maximum_row_width_ratio
        self._maximum_row_width_variation = maximum_row_width_variation_ratio
        self._maximum_fragment_gap = maximum_fragment_gap_ratio
        self._sample_count = sample_count
        self._alpha = smoothing_alpha
        self._maximum_missing = maximum_missing_frames
        self._minimum_confidence = minimum_curve_confidence
        self._maximum_boundaries = maximum_boundaries
        self._association_distance = temporal_association_distance_ratio
        self._bbox_margin = excluded_bbox_margin_ratio
        self._minimum_confirmation = minimum_confirmation_frames
        self._previous: dict[str, LaneCurve] = {}
        self._missing: dict[str, int] = {}
        self._ages: dict[str, int] = {}
        self._next_boundary_id = 1
        self.occlusion_mask: np.ndarray[Any, Any] | None = None
        self._marking_occlusion_mask: np.ndarray[Any, Any] | None = None
        self._parent_lane_mask: np.ndarray[Any, Any] | None = None
        self.marking_mask: np.ndarray[Any, Any] | None = None
        self._marking_masks: dict[str, np.ndarray[Any, Any]] = {}
        self._contaminated_paint: np.ndarray[Any, Any] | None = None
        self._bay_history: tuple[
            np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], int,
        ] | None = None

    def reset(self) -> None:
        """清除不同影片之間的時間狀態。"""
        self._previous.clear()
        self._missing.clear()
        self._ages.clear()
        self._next_boundary_id = 1
        self.occlusion_mask = None
        self._marking_occlusion_mask = None
        self._parent_lane_mask = None
        self.marking_mask = None
        self._marking_masks = {}
        self._contaminated_paint = None
        self._bay_history = None

    def detect(
        self,
        frame: Any,
        excluded_boxes: list[BBox] | None = None,
    ) -> LaneLineFrame:
        """推論車道線遮罩並輸出經跨幀配對的曲線。"""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
            self.reset()
            return LaneLineFrame(
                status=LaneLineStatus.UNKNOWN,
                reason="lane segmentation requires a BGR numpy frame",
            )
        height, width = frame.shape[:2]
        if height < 2 or width < 2:
            self.reset()
            return LaneLineFrame(status=LaneLineStatus.UNKNOWN, reason="empty frame")

        probability, drivable_probability = self._segmentation_probabilities(frame)
        mask, mask_diagnostics = self._candidate_mask(
            frame,
            probability,
            excluded_boxes or [],
            drivable_probability,
        )
        paint, rejected_paint = self._paint_context(frame)
        bay_mask = self._labelled_bay_markings(frame, drivable_probability)
        self._marking_masks["labelled_bay_context"] = bay_mask
        rejected_paint |= bay_mask
        self.marking_mask = rejected_paint.copy()
        repeated = self._marking_masks.get("repeated_marking_context")
        if repeated is not None:
            self.marking_mask |= repeated
        candidates, rejection_reasons, component_count = self._component_curves(
            mask,
            probability,
            drivable_probability,
            use_context=True,
        )
        # Reject non-lane fragments before merging: a transverse marking must
        # not borrow a neighbouring fragment's direction or road support.
        supported: list[LaneCurve] = []
        occlusion_supported: set[int] = set()
        parent_labels: np.ndarray[Any, Any] | None = None
        parent_support: set[int] = set()
        decisions: list[LaneCandidateDecision] = []
        for curve in candidates:
            was_occluded = (
                (curve.component_height_ratio or 0) < self._minimum_fragment_span
                and self._occluded_fragment(curve, width, frame)
            )
            if was_occluded:
                if parent_labels is None:
                    parent_labels, parent_support = self._occluded_parent_components(
                        frame, probability, drivable_probability, paint, rejected_paint,
                    )
                was_occluded = self._curve_component_label(curve, parent_labels) in parent_support
            refined = self._refine_paint_geometry(curve, frame)
            if refined is not curve:
                decisions.append(LaneCandidateDecision(
                    stage="context", reason="aligned_to_visible_paint", curve=curve,
                ))
                curve = refined
            reason = self._context_rejection(
                curve, paint, rejected_paint, drivable_probability, frame
            )
            if reason == "low_drivable_context":
                trimmed = self._painted_tail(curve, frame)
                if trimmed is not None:
                    decisions.append(LaneCandidateDecision(
                        stage="context", reason="trimmed_unpainted_end", curve=curve,
                    ))
                    curve = trimmed
                    reason = self._context_rejection(
                        curve, paint, rejected_paint, drivable_probability, frame
                    )
            if reason:
                rejection_reasons[reason] += 1
                decisions.append(LaneCandidateDecision(stage="context", reason=reason, curve=curve))
            else:
                supported.append(curve)
                if was_occluded:
                    occlusion_supported.add(id(curve))
        merged = self._merge_fragments(supported, width, height, frame)
        contextual: list[LaneCurve] = []
        for curve in merged:
            reason = (
                None if any(curve is original for original in supported)
                else self._context_rejection(
                    curve, paint, rejected_paint, drivable_probability, frame
                )
            )
            if reason:
                rejection_reasons[reason] += 1
                decisions.append(LaneCandidateDecision(stage="merge", reason=reason, curve=curve))
            elif (
                (curve.component_height_ratio or 0) < self._minimum_span
                and np.hypot(
                    curve.points[-1].x - curve.points[0].x,
                    curve.points[-1].y - curve.points[0].y,
                ) < width * self._minimum_span
                and not (
                    id(curve) in occlusion_supported
                    and self._occluded_fragment(curve, width, frame)
                )
                and not (
                    (curve.component_height_ratio or 0) >= self._minimum_fragment_span
                    and self._narrow_stripe_support(curve, frame) >= 0.9
                )
                and not (
                    self._narrow_stripe_support(curve, frame) >= 0.8
                    and np.hypot(
                        curve.points[-1].x - curve.points[0].x,
                        curve.points[-1].y - curve.points[0].y,
                    ) >= width * self._minimum_fragment_span
                )
            ):
                rejection_reasons["unmerged_short_fragment"] += 1
                decisions.append(LaneCandidateDecision(
                    stage="length", reason="unmerged_short_fragment", curve=curve,
                ))
            else:
                contextual.append(curve)
        diagnostics = LaneLineDiagnostics(
            **mask_diagnostics,
            component_count=component_count,
            accepted_component_count=len(candidates),
            merged_candidate_count=len(merged),
            rejection_reasons=dict(rejection_reasons),
            candidate_decisions=tuple(decisions),
        )
        return self._stabilize(contextual, width, diagnostics, frame)

    @staticmethod
    def _curve_component_label(curve: LaneCurve, labels: np.ndarray[Any, Any]) -> int:
        height, width = labels.shape
        xs = np.clip(np.rint([p.x for p in curve.points]).astype(int), 0, width - 1)
        ys = np.clip(np.rint([p.y for p in curve.points]).astype(int), 0, height - 1)
        sampled = labels[ys, xs]
        sampled = sampled[sampled > 0]
        return int(np.bincount(sampled).argmax()) if sampled.size else 0

    def _occluded_parent_components(
        self, frame: np.ndarray[Any, Any], probability: np.ndarray[Any, Any],
        road: np.ndarray[Any, Any], paint: np.ndarray[Any, Any], bad: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], set[int]]:
        """Contrast-separated pieces cannot invent occlusion in a rejected parent."""
        parent = getattr(self, "_parent_lane_mask", None)
        if parent is None or self.occlusion_mask is None or not np.any(self.occlusion_mask):
            return np.zeros(probability.shape, np.int32), set()
        _, labels = _cv2().connectedComponents(parent, 8)
        curves, _, _ = self._component_curves(parent, probability, road, use_context=True)
        supported = {
            self._curve_component_label(curve, labels) for curve in curves
            if self._occluded_fragment(curve, frame.shape[1], frame)
            and self._context_rejection(curve, paint, bad, road, frame) is None
        }
        supported.discard(0)
        return labels, supported

    def _occluded_fragment(
        self, curve: LaneCurve, width: int, frame: np.ndarray[Any, Any] | None = None,
    ) -> bool:
        """Rescue a short visible stripe only when masking cut an endpoint."""
        if self.occlusion_mask is None:
            return False
        # Mere proximity to a car also occurs at pedestrian crossings. A
        # shortened fragment must still follow visible narrow paint throughout.
        if frame is not None and self._narrow_stripe_support(curve, frame) < 0.9:
            return False
        first, last = curve.points[0], curve.points[-1]
        if np.hypot(last.x - first.x, last.y - first.y) < (
            width * self._minimum_fragment_span * 0.5
        ):
            return False
        height = self.occlusion_mask.shape[0]
        radius = max(1, round(width * 0.006))
        for point in (first, last):
            x, y = round(point.x), round(point.y)
            if not (0 <= x < width and 0 <= y < height):
                continue
            nearby = self.occlusion_mask[
                max(0, y-radius):min(height, y+radius+1),
                max(0, x-radius):min(width, x+radius+1),
            ]
            if np.any(nearby):
                return True
        return False

    def _stabilize(
        self,
        candidates: list[LaneCurve],
        width: int,
        diagnostics: LaneLineDiagnostics | None = None,
        frame: np.ndarray[Any, Any] | None = None,
    ) -> LaneLineFrame:
        """配對、確認及有限延續目前影格的候選曲線。"""
        deduplicated = self._deduplicate(candidates, width, frame)
        diagnostics = (diagnostics or LaneLineDiagnostics()).model_copy(
            update={"deduplicated_candidate_count": len(deduplicated)}
        )
        associated = self._associate(deduplicated, width)
        decisions = list(diagnostics.candidate_decisions)
        for curve in associated:
            if self._ages.get(curve.boundary_id, 0) < self._minimum_confirmation:
                decisions.append(LaneCandidateDecision(
                    stage="confirmation", reason="awaiting_confirmation", curve=curve,
                ))
        associated_ids = {curve.boundary_id for curve in associated}
        fresh = [
            curve
            for curve in associated
            if self._ages.get(curve.boundary_id, 0) >= self._minimum_confirmation
        ]
        output = list(fresh)
        for boundary_id, previous in list(self._previous.items()):
            if boundary_id in associated_ids:
                continue
            missing = self._missing.get(boundary_id, 0) + 1
            self._missing[boundary_id] = missing
            confirmed = self._ages.get(boundary_id, 0) >= self._minimum_confirmation
            confidence = previous.confidence * (0.9**missing)
            conflicts_with_fresh = any(
                self._curve_distance(previous, current, width)
                <= self._association_distance
                for current in fresh
            )
            marking_mask = getattr(self, "marking_mask", None)
            rejected_now = (
                marking_mask is not None and self._mask_overlap(previous, marking_mask) > 0.2
            )
            if rejected_now:
                decisions.append(LaneCandidateDecision(
                    stage="carry", reason="current_marking_evidence", curve=previous,
                ))
            if (
                confirmed
                and missing <= self._maximum_missing
                and confidence >= self._minimum_confidence
                and not conflicts_with_fresh
                and not rejected_now
            ):
                output.append(
                    previous.model_copy(
                        update={
                            "confidence": confidence,
                            "carried_frames": missing,
                            "confirmed_frames": self._ages[boundary_id],
                        }
                    )
                )
                continue
            self._remove_track(boundary_id)

        diagnostics = diagnostics.model_copy(update={"candidate_decisions": tuple(decisions)})
        curves = tuple(sorted(output, key=lambda curve: curve.points[-1].x))
        fresh_count = sum(curve.carried_frames == 0 for curve in curves)
        if fresh_count >= 2:
            return LaneLineFrame(
                status=LaneLineStatus.VALID,
                curves=curves,
                diagnostics=diagnostics,
            )
        if curves:
            return LaneLineFrame(
                status=LaneLineStatus.DEGRADED,
                curves=curves,
                reason="fewer than two confirmed lane boundaries",
                diagnostics=diagnostics,
            )
        return LaneLineFrame(
            status=LaneLineStatus.UNKNOWN,
            reason="YOLOP found no confirmed lane-line pixels",
            diagnostics=diagnostics,
        )

    def _segmentation_probabilities(
        self, frame: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        cv2 = _cv2()
        height, width = frame.shape[:2]
        scale = min(self._input_size / width, self._input_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        left = (self._input_size - resized_width) // 2
        top = (self._input_size - resized_height) // 2
        rgb = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        normalized = canvas.astype(np.float32) / 255.0
        normalized = np.asarray((
            normalized - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32), dtype=np.float32)
        blob = normalized.transpose(2, 0, 1)[None]
        self._net.setInput(blob)
        outputs = self._net.forward(["lane_line_seg", "drive_area_seg"])
        probabilities: list[np.ndarray[Any, Any]] = []
        for name, logits_value in zip(
            ("lane_line_seg", "drive_area_seg"), outputs, strict=True
        ):
            logits = np.asarray(logits_value, dtype=np.float32)
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
                raise RuntimeError(f"unexpected YOLOP {name} output shape: {logits.shape}")
            # This ONNX export ends in Sigmoid, not logits. Normalize its two
            # scores without applying another sigmoid/softmax; 0.5 is argmax.
            probability = self._foreground_score(logits)
            cropped = probability[top : top + resized_height, left : left + resized_width]
            probabilities.append(
                cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            )
        return probabilities[0], probabilities[1]

    @staticmethod
    def _foreground_score(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise RuntimeError("YOLOP expects finite post-Sigmoid scores in [0, 1]")
        return np.asarray(
            scores[0, 1] / np.maximum(scores[0].sum(axis=0), np.finfo(np.float32).eps),
            dtype=np.float32,
        )

    def _candidate_mask(
        self,
        frame: np.ndarray[Any, Any],
        probability: np.ndarray[Any, Any],
        excluded_boxes: list[BBox],
        drivable_probability: np.ndarray[Any, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, int]]:
        cv2 = _cv2()
        height, width = probability.shape
        lane_mask = np.where(probability >= self._threshold, 255, 0).astype(np.uint8)
        lane_mask[: round(height * self._roi_top)] = 0
        hls = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HLS)
        strict_white = cv2.inRange(
            hls,
            np.asarray((0, self._white_lightness, 0), dtype=np.uint8),
            np.asarray((179, 255, self._white_saturation), dtype=np.uint8),
        )
        original_white = strict_white.copy()
        lightness = hls[:, :, 1]
        local_mean = cv2.GaussianBlur(lightness, (0, 0), sigmaX=9)
        contrast = lightness.astype(np.int16) - local_mean.astype(np.int16)
        strict_white[contrast < self._local_contrast] = 0
        adaptive_white = np.where(
            (probability >= self._strong_probability)
            & (lightness >= max(35, self._white_lightness // 2))
            & (hls[:, :, 2] <= min(255, self._white_saturation + 35))
            & (contrast >= self._local_contrast),
            255,
            0,
        ).astype(np.uint8)
        white = cv2.bitwise_or(strict_white, adaptive_white)
        white = cv2.dilate(white, np.ones((5, 5), dtype=np.uint8))
        mask = cv2.bitwise_and(lane_mask, white)
        parent_mask = cv2.bitwise_and(lane_mask, cv2.dilate(
            cv2.bitwise_or(original_white, adaptive_white), np.ones((5, 5), np.uint8),
        ))
        white_supported_pixel_count = int(np.count_nonzero(mask))
        margin_x = round(width * self._bbox_margin)
        margin_y = round(height * self._bbox_margin)
        occlusion = np.zeros((height, width), dtype=np.uint8)
        protected = np.zeros_like(mask)
        for bbox in excluded_boxes:
            cv2.rectangle(
                protected, (max(0, round(bbox.x1)), max(0, round(bbox.y1))),
                (min(width - 1, round(bbox.x2)), min(height - 1, round(bbox.y2))),
                255, -1,
            )
            x1 = max(0, round(bbox.x1) - margin_x)
            y1 = max(0, round(bbox.y1) - margin_y)
            x2 = min(width - 1, round(bbox.x2) + margin_x)
            y2 = min(height - 1, round(bbox.y2) + margin_y)
            box_mask = np.zeros_like(mask)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            corner_foreground = (
                bbox.area >= width * height * 0.12
                and bbox.y2 >= height * 0.97
                and (bbox.x1 <= width * 0.02 or bbox.x2 >= width * 0.98)
            )
            if corner_foreground and drivable_probability is not None:
                # Refine oversized corner boxes with road evidence, never simply
                # discard them: non-road ego body and other vehicle boxes stay masked.
                road = (drivable_probability >= 0.6).astype(np.uint8)
                radius = max(1, round(width * 0.006))
                road = cv2.dilate(road, np.ones((radius * 2 + 1,) * 2, np.uint8))
                box_mask[road > 0] = 0
            occlusion = cv2.bitwise_or(occlusion, box_mask)
        self._marking_occlusion_mask = occlusion.copy()
        if excluded_boxes and drivable_probability is not None:
            # Geometry refinement must use this frame's box interiors, not the
            # previous frame's expanded occlusion mask.
            self.occlusion_mask = protected
            visible_paint = self._visible_margin_paint(
                frame, mask, probability, drivable_probability, protected, occlusion,
            )
            occlusion[visible_paint > 0] = 0
        self.occlusion_mask = occlusion
        parent_mask[occlusion > 0] = 0
        parent_mask = cv2.morphologyEx(
            parent_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8),
        )
        parent_mask[occlusion > 0] = 0
        self._parent_lane_mask = parent_mask
        mask[occlusion > 0] = 0
        vehicle_excluded_pixel_count = white_supported_pixel_count - int(
            np.count_nonzero(mask)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        mask[occlusion > 0] = 0
        return mask, {
            "lane_pixel_count": int(np.count_nonzero(lane_mask)),
            "white_supported_pixel_count": white_supported_pixel_count,
            "vehicle_excluded_pixel_count": vehicle_excluded_pixel_count,
        }

    def _visible_margin_paint(
        self, frame: np.ndarray[Any, Any], mask: np.ndarray[Any, Any],
        probability: np.ndarray[Any, Any], road: np.ndarray[Any, Any],
        protected: np.ndarray[Any, Any], expanded: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Release evidenced lane paint in padding, preserving every box interior."""
        visible = mask.copy()
        visible[protected > 0] = 0
        curves, _, _ = self._component_curves(visible, probability, road, use_context=True)
        result = np.zeros_like(mask)
        height, width = mask.shape
        for curve in curves:
            # Padding can hide most of a dash; still require a visible consensus
            # spanning at least 2% of image width before releasing any pixels.
            curve = self._refine_paint_geometry(curve, frame, minimum_length=width * 0.02)
            points = np.asarray([(p.x, p.y) for p in curve.points])
            xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
            # Rescue predominantly hidden dashes. Extending already visible
            # components can alter their geometry or expose nearby markings.
            if np.mean(expanded[ys, xs] > 0) < 0.5:
                continue
            dx, dy = points[-1] - points[0]
            if dy <= 0:
                continue
            horizon_x = points[0, 0] + dx / dy * (height * self._roi_top - points[0, 1])
            if not width * 0.25 <= horizon_x <= width * 0.75:
                continue
            if self._narrow_stripe_support(curve, frame) < 0.9:
                continue
            tangent = np.gradient(points, axis=0)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.maximum(
                np.linalg.norm(tangent, axis=1), 1.0,
            )[:, None]
            if self._paint_ridge_support(points, normal, frame) < 0.9:
                continue
            cv2 = _cv2()
            cv2.polylines(result, [np.rint(points).astype(np.int32)], False, 255,
                          max(3, round(width * 0.006)))
        result[(protected > 0) | (mask == 0)] = 0
        return result

    def _paint_context(
        self, frame: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        # Margin rescue must not change full-shape marking classification or
        # turn a formerly vehicle-contaminated symbol into a new crosswalk group.
        effective = self.occlusion_mask
        marking_mask = getattr(self, "_marking_occlusion_mask", None)
        if marking_mask is not None:
            self.occlusion_mask = marking_mask
        try:
            return self._classify_paint_context(frame)
        finally:
            self.occlusion_mask = effective

    def _classify_paint_context(
        self, frame: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Keep full paint shapes, including arrow heads absent from lane segmentation."""
        cv2 = _cv2()
        height, width = frame.shape[:2]
        hls = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HLS)
        lightness = hls[:, :, 1].astype(np.float32)
        background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(3, width / 60))
        paint = (
            (lightness > background + max(25, self._local_contrast))
            & (lightness >= self._white_lightness)
            & (hls[:, :, 2] <= self._white_saturation)
        ).astype(np.uint8)
        paint[: round(height * self._roi_top)] = 0
        self._marking_masks = self._grouped_markings(hls)
        # Classify the visible paint BEFORE cutting it at detection-box edges.
        # A box can truncate a stripe into a blob or hide an arrow's head from
        # this classifier. Candidate extraction and rendering still mask cars.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(paint, 8)
        rejected = np.zeros_like(paint)
        contaminated = np.zeros_like(paint)
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            # Distant arrow heads can be smaller than the normal whole-symbol
            # gate. They require width expansion; a small rectangular paint chip
            # alone must not label an otherwise continuous stripe as an arrow.
            if (
                area < max(3, height * width * self._minimum_area * 3)
                or area > height * width * 0.04
            ):
                continue
            region_labels = labels[y : y + h, x : x + w] == label
            if self.occlusion_mask is not None:
                covered = self.occlusion_mask[y : y + h, x : x + w][region_labels]
                if np.mean(covered > 0) > 0.1:
                    contaminated[y:y+h, x:x+w][region_labels] = 1
                if np.mean(covered > 0) > 0.5:
                    # Mostly vehicle/occlusion: its shape is not reliable road
                    # symbol evidence. Do not classify its clipped remainder.
                    continue
            ys, xs = np.where(labels[y : y + h, x : x + w] == label)
            coordinates = np.column_stack((xs, ys)).astype(np.float64)
            coordinates -= coordinates.mean(axis=0)
            values, axes = np.linalg.eigh(np.cov(coordinates.T))
            elongation = float(np.sqrt(values[1] / max(values[0], 1.0)))
            along = coordinates @ axes[:, 1]
            across = coordinates @ axes[:, 0]
            bins = np.linspace(float(along.min()), float(along.max()) + 1, 13)
            widths = []
            for start, end in zip(bins[:-1], bins[1:], strict=True):
                section = across[(along >= start) & (along < end)]
                if section.size >= 3:
                    widths.append(float(np.ptp(section)) + 1)
            expansion = (
                # Compare with the narrow quarter, not the median: a detached
                # triangular head can occupy most of the component's length.
                float(np.percentile(widths, 90) / max(np.percentile(widths, 25), 1))
                if widths else 1.0
            )
            # An arrow shoulder is an abrupt interior width change even when
            # its long, thick shaft makes percentile/median expansion small.
            interior = np.asarray(widths[1:-1])
            shoulder = bool(
                interior.size >= 4
                and np.median(interior) >= width * 0.008
                and np.any(
                    np.maximum(interior[1:], interior[:-1])
                    > 1.8 * np.maximum(np.minimum(interior[1:], interior[:-1]), 1)
                )
            )
            # Rotation-independent width expansion catches arrows pointing sideways.
            small_symbol = area < height * width * self._minimum_area * 6
            arrow_shape = (
                elongation < 12 and expansion > 2.2 and len(widths) >= 6
                if small_symbol else
                elongation < 4 or (elongation < 12 and (expansion > 2.2 or shoulder))
            )
            if arrow_shape:
                region = rejected[y : y + h, x : x + w]
                region[labels[y : y + h, x : x + w] == label] = 1
        radius = max(1, round(width * 0.004))
        kernel = np.ones((radius * 2 + 1,) * 2, dtype=np.uint8)
        for reason, marking in self._marking_masks.items():
            if reason == "repeated_marking_context":
                continue
            rejected = cv2.bitwise_or(rejected, marking)
        self._contaminated_paint = cv2.dilate(contaminated, kernel)
        return cv2.dilate(paint, kernel), cv2.dilate(rejected, kernel)

    def _grouped_markings(self, hls: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
        """Identify connected parking corners and groups of broad crosswalk bars."""
        cv2 = _cv2()
        height, width = hls.shape[:2]
        lightness = hls[:, :, 1].astype(np.float32)
        background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=width / 40)
        paint = (
            (lightness >= self._white_lightness)
            & (lightness > background + self._local_contrast)
            & (hls[:, :, 2] <= self._white_saturation)
        ).astype(np.uint8)
        paint[:round(height * self._roi_top)] = 0
        # The recording's text footer is not road evidence.
        paint[round(height * 0.95):] = 0
        radius = max(1, round(width * 0.002))
        paint = cv2.morphologyEx(
            paint, cv2.MORPH_CLOSE, np.ones((radius * 2 + 1,) * 2, np.uint8)
        )
        contours, _ = cv2.findContours(paint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars: list[tuple[Any, tuple[int, int, int, int], float, float]] = []
        parking = np.zeros_like(paint)
        crosswalk = np.zeros_like(paint)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not width * height * self._minimum_area * 3 <= area <= width * height * 0.04:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            (_, _), lengths, _ = cv2.minAreaRect(contour)
            minor, major = sorted(lengths)
            # Paint shrinks with distance from the road horizon. Fixed near-
            # camera pixel gates missed the far end of pedestrian crossings.
            depth = max(0.15, (y - height * self._roi_top) / (height * (1 - self._roi_top)))
            if major < width * 0.045 * depth:
                continue
            if self.occlusion_mask is not None:
                local = np.zeros((h, w), np.uint8)
                cv2.drawContours(local, [contour - (x, y)], -1, 1, cv2.FILLED)
                if np.mean(self.occlusion_mask[y:y+h, x:x+w][local > 0] > 0) > 0.5:
                    continue
            solidity = area / max(1, cv2.contourArea(cv2.convexHull(contour)))
            bars.append((contour, (x, y, w, h), minor, solidity))
            if solidity >= 0.55 or minor < width * 0.018 or major < width * 0.08:
                continue
            polygon = cv2.approxPolyDP(contour, width * 0.004, True)[:, 0, :]
            edges = np.roll(polygon, -1, axis=0) - polygon
            lengths_array = np.linalg.norm(edges, axis=1)
            long_edges = edges[lengths_array > width * 0.05]
            # An L/T corner contains two long straight strokes meeting at a
            # sharp angle. A thin curved lane does not form this paint topology.
            corner = any(
                abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b)) < 0.85
                for index, a in enumerate(long_edges)
                for b in long_edges[index + 1:]
            )
            if (
                corner
                and area / max(cv2.arcLength(contour, True), 1) < width * 0.006
            ):
                cv2.drawContours(parking, [contour], -1, 1, cv2.FILLED)
        # Recover a grid side separated from a recognised corner by worn paint.
        nearby_parking = cv2.dilate(
            parking, np.ones((max(3, round(width * 0.04)) | 1,) * 2, np.uint8)
        )
        for contour, _, minor, solidity in bars:
            if minor > width * 0.025 or solidity < 0.6:
                continue
            if np.any(nearby_parking[contour[:, 0, 1], contour[:, 0, 0]]):
                cv2.drawContours(parking, [contour], -1, 1, cv2.FILLED)
        broad = [
            item for item in bars
            if item[3] >= 0.7 and (
                # Preserve the established near-bar rule. Perspective support
                # supplements it; a far-bar aspect gate must not remove long bars.
                (item[2] >= width * 0.008 and item[1][3] >= height * 0.04)
                or (
                    item[2] >= max(2, width * 0.008 * max(
                        0.15, (item[1][1] - height * self._roi_top)
                        / (height * (1 - self._roi_top)),
                    ))
                    and min(item[1][2], item[1][3]) >= 3
                    and max(cv2.minAreaRect(item[0])[1]) / max(item[2], 1) <= 7
                )
            )
        ]
        for contour, (x, y, w, h), _, _ in broad:
            neighbours = [
                other for other in broad
                if abs(other[1][1] - y) <= height * 0.06
                and min(y + h, other[1][1] + other[1][3])
                - max(y, other[1][1]) >= min(h, other[1][3]) * 0.5
                and abs((other[1][0] + other[1][2] / 2) - (x + w / 2)) >= width * 0.035
                and abs((other[1][0] + other[1][2] / 2) - (x + w / 2)) <= width * 0.5
            ]
            if len(neighbours) >= 2:
                cv2.drawContours(crosswalk, [contour], -1, 1, cv2.FILLED)
        radius = max(1, round(width * 0.004))
        kernel = np.ones((radius * 2 + 1,) * 2, np.uint8)
        repeated = self._parallel_markings(paint)
        return {
            "parking_marking_context": cv2.dilate(parking, kernel),
            "crosswalk_context": cv2.dilate(crosswalk, kernel),
            "repeated_marking_context": cv2.dilate(repeated, kernel),
        }

    def _labelled_bay_markings(
        self, frame: np.ndarray[Any, Any], road: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Use complex road-paint groups beside a stripe, not a bus or image side.

        Several enclosed paint cavities supply label evidence even when
        the bay's end is outside the image. This is topology, not text reading.
        Dim paint may supply context, but cannot create a lane candidate.
        """
        cv2 = _cv2()
        height, width = frame.shape[:2]
        hls = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HLS)
        lightness = hls[:, :, 1].astype(np.float32)
        background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=width / 40)
        road_neighbourhood = cv2.dilate(
            (road > 0.5).astype(np.uint8), np.ones((11, 11), np.uint8),
        )
        paint = (
            (lightness >= max(40, self._white_lightness * 0.55))
            & (lightness > background + self._local_contrast)
            & (hls[:, :, 2] <= self._white_saturation)
            & (road_neighbourhood > 0)
        ).astype(np.uint8)
        paint[:round(height * self._roi_top)] = 0
        occlusion = getattr(self, "_marking_occlusion_mask", self.occlusion_mask)
        if occlusion is not None:
            paint[occlusion > 0] = 0
        radius = max(1, round(width * 0.0015))
        paint = cv2.morphologyEx(
            paint, cv2.MORPH_CLOSE, np.ones((radius * 2 + 1,) * 2, np.uint8),
        )
        current = self._labelled_bay_sides(paint, frame)
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        features = (road > 0.5).astype(np.uint8)
        features[:round(height * self._roi_top)] = 0
        features[round(height * 0.95):] = 0
        if occlusion is not None:
            features[occlusion > 0] = 0
        history = getattr(self, "_bay_history", None)
        transported = np.zeros_like(paint)
        age = 0
        if history is not None and history[3] < 45:
            previous_gray, previous_features, previous_mask, age = history
            transported = self._transport_bay_mask(
                previous_gray, gray, previous_features, features, previous_mask,
            )
            visible = cv2.dilate(paint, np.ones((9, 9), np.uint8))
            transported = self._visible_bay_evidence(transported, visible)
            if not np.any(transported):
                transported = self._follow_bay_paint(
                    previous_gray, gray, previous_mask, paint, frame,
                )
                transported = self._visible_bay_evidence(transported, visible)
            age += 1
        history_mask = current
        if np.any(current):
            if np.any(transported):
                if np.count_nonzero(current & transported) >= transported.sum() * 0.2:
                    current |= transported
                    age = 0
                    history_mask = current
                else:
                    # A different fresh marking cannot replace or renew this
                    # still-observed bay. Its current evidence is usable, but
                    # only the existing bay retains temporal state.
                    history_mask = transported
                    current |= transported
            else:
                age = 0
        else:
            current = transported
            history_mask = current
        self._bay_history = (
            (gray, features, history_mask.copy(), age) if np.any(history_mask) else None
        )
        return current

    @staticmethod
    def _visible_bay_evidence(
        mask: np.ndarray[Any, Any], visible: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Expire disappeared glyphs separately from a still-visible border."""
        cv2 = _cv2()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        retained = np.zeros_like(mask)
        for index in range(1, count):
            component = labels == index
            supported = component & (visible > 0)
            if np.count_nonzero(supported) >= stats[index, cv2.CC_STAT_AREA] * 0.55:
                retained[supported] = 1
        return retained

    @staticmethod
    def _transport_bay_mask(
        previous_gray: np.ndarray[Any, Any], gray: np.ndarray[Any, Any],
        previous_features: np.ndarray[Any, Any], features: np.ndarray[Any, Any],
        previous_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Require consistent forward/backward road flow and a bounded mapping."""
        cv2 = _cv2()
        height, width = gray.shape
        empty = np.zeros_like(gray)
        if previous_gray.shape != gray.shape:
            return empty
        # The mapping describes this road patch. Distant vehicles and the
        # camera's own body cannot vote for a competing global motion.
        radius = max(3, round(width * 0.12)) | 1
        local_features = previous_features & cv2.dilate(
            previous_mask, np.ones((radius, radius), np.uint8),
        )
        points = cv2.goodFeaturesToTrack(
            previous_gray, 250, 0.001, 8, mask=local_features, blockSize=7,
        )
        if points is None or len(points) < 16:
            return empty
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3,
        )
        if moved is None or forward is None:
            return empty
        returned, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, previous_gray, moved, None, winSize=(21, 21), maxLevel=3,
        )
        if returned is None or backward is None:
            return empty
        before, after = points[:, 0], moved[:, 0]
        xs = np.clip(np.rint(after[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(after[:, 1]).astype(int), 0, height - 1)
        valid = (
            (forward[:, 0] > 0) & (backward[:, 0] > 0)
            & (np.linalg.norm(points[:, 0] - returned[:, 0], axis=1) <= 1.5)
            & (after[:, 0] >= 0) & (after[:, 0] < width)
            & (after[:, 1] >= 0) & (after[:, 1] < height)
            & (features[ys, xs] > 0)
        )
        before, after = before[valid], after[valid]
        if len(before) < 16:
            return empty
        transform, inliers = cv2.estimateAffine2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        )
        if (
            transform is None or inliers is None
            or np.mean(inliers) < 0.4 or np.count_nonzero(inliers) < 12
        ):
            return empty
        support = before[inliers[:, 0] > 0]
        if np.ptp(support[:, 0]) < width * 0.04 or np.ptp(support[:, 1]) < height * 0.08:
            return empty
        ys, xs = np.where(previous_mask > 0)
        if not xs.size:
            return empty
        corners = np.asarray([[[xs.min(), ys.min()], [xs.max(), ys.min()],
                               [xs.max(), ys.max()], [xs.min(), ys.max()]]], dtype=np.float32)
        projected = cv2.transform(corners, transform)
        if not np.all(np.isfinite(projected)) or np.max(
            np.linalg.norm(projected - corners, axis=2)
        ) > width * 0.04:
            return empty
        scale = np.linalg.svd(transform[:, :2], compute_uv=False)
        if np.any(scale < 0.8) or np.any(scale > 1.25):
            return empty
        return np.asarray(cv2.warpAffine(
            previous_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
        ), dtype=np.uint8)

    def _follow_bay_paint(
        self, previous_gray: np.ndarray[Any, Any], gray: np.ndarray[Any, Any],
        previous_mask: np.ndarray[Any, Any], paint: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Associate the observed stripe when texture flow is underconstrained.

        A highly similar scene, nearby previous evidence and a sustained narrow
        stripe are all required. Only the CURRENT observed stroke is returned;
        this does not renew the label's age or draw a historical screen mask.
        """
        cv2 = _cv2()
        height, width = gray.shape
        result = np.zeros_like(paint)
        if previous_gray.shape != gray.shape:
            return result
        images = [cv2.resize(img, (64, 36)).ravel().astype(float)
                  for img in (previous_gray, gray)]
        if min(float(np.std(img)) for img in images) < 5:
            return result
        if float(np.corrcoef(images)[0, 1]) < 0.85:
            return result
        radius = max(3, round(width * 0.016)) | 1
        nearby = cv2.dilate(previous_mask, np.ones((radius, radius), np.uint8))
        lines = cv2.HoughLinesP(
            paint * 255, 1, np.pi / 180, 20,
            minLineLength=max(20, round(width * 0.025)), maxLineGap=8,
        )
        if lines is None:
            return result
        for line in lines[:, 0]:
            samples = np.linspace(line[:2], line[2:], 24)
            xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
            if np.mean(nearby[ys, xs] > 0) < 0.7:
                continue
            stripe = LaneCurve(
                boundary_id="bay-follow", confidence=1.0,
                points=tuple(Point2D(x=float(x), y=float(y)) for x, y in samples),
            )
            if self._narrow_stripe_support(
                stripe, frame, gray=gray, maximum_width_ratio=0.025,
            ) >= 0.8:
                cv2.line(result, tuple(line[:2]), tuple(line[2:]),
                         1, max(5, round(width * 0.012)))
        return result

    def _bay_corner_context(
        self, paint: np.ndarray[Any, Any], frame: np.ndarray[Any, Any],
        contours: Any,
    ) -> np.ndarray[Any, Any]:
        """A visible sparse elbow supplies bay evidence before its label is legible.

        Both long strokes need paint ridges. A continued crossbar forms a T,
        so it cannot make a real lane at a stop line into a parking boundary.
        """
        cv2 = _cv2()
        height, width = paint.shape
        result = np.zeros_like(paint)
        visible = cv2.dilate(paint, np.ones((9, 9), np.uint8))
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        anchors = []
        for contour in contours:
            area = cv2.contourArea(contour)
            major = max(cv2.minAreaRect(contour)[1])
            if not width * height * self._minimum_area * 3 <= area <= width * height * 0.04:
                continue
            if major < width * 0.2 or area / max(cv2.arcLength(contour, True), 1) >= width * 0.006:
                continue
            if area / max(cv2.contourArea(cv2.convexHull(contour)), 1) >= 0.55:
                continue
            polygon = cv2.approxPolyDP(contour, width * 0.004, True)[:, 0].astype(float)
            for index in range(len(polygon)):
                corner = polygon[(index + 1) % len(polygon)]
                ends = [polygon[index], polygon[(index + 2) % len(polygon)]]
                directions = [end - corner for end in ends]
                lengths = [float(np.linalg.norm(direction)) for direction in directions]
                if min(lengths) < width * 0.08:
                    continue
                axes = [
                    direction / length
                    for direction, length in zip(directions, lengths, strict=True)
                ]
                if abs(float(axes[0] @ axes[1])) >= 0.85:
                    continue
                supported = True
                for end, axis in zip(ends, axes, strict=True):
                    beyond = corner - np.asarray([0.025, 0.05, 0.08])[:, None] * width * axis
                    bx = np.clip(np.rint(beyond[:, 0]).astype(int), 0, width - 1)
                    by = np.clip(np.rint(beyond[:, 1]).astype(int), 0, height - 1)
                    stripe = LaneCurve(
                        boundary_id="bay-corner", confidence=1.0,
                        points=tuple(Point2D(x=float(x), y=float(y))
                                     for x, y in np.linspace(corner, end, 16)),
                    )
                    if (np.mean(visible[by, bx] > 0) >= 0.7
                        or self._narrow_stripe_support(
                            stripe, frame, gray=gray, maximum_width_ratio=0.025,
                        ) < 0.8):
                        supported = False
                        break
                if not supported:
                    continue
                cv2.drawContours(result, [contour], -1, 1, cv2.FILLED)
                for end, axis in zip(ends, axes, strict=True):
                    if abs(end[1] - corner[1]) >= height * 0.08:
                        anchors.append((corner, end, axis))
        lines = cv2.HoughLinesP(
            paint * 255, 1, np.pi / 180, 20,
            minLineLength=max(20, round(width * 0.025)), maxLineGap=8,
        ) if anchors else None
        if lines is not None:
            for line in lines[:, 0]:
                points = line.reshape(2, 2).astype(float)
                axis = points[1] - points[0]
                axis /= np.linalg.norm(axis)
                for first, last, parent_axis in anchors:
                    normal = np.asarray([-parent_axis[1], parent_axis[0]])
                    if abs(float(axis @ parent_axis)) < 0.97:
                        continue
                    if np.max(np.abs((points - first) @ normal)) > width * 0.012:
                        continue
                    along = (points - first) @ parent_axis
                    length = float(np.linalg.norm(last - first))
                    if along.max() < -width * 0.12 or along.min() > length + width * 0.12:
                        continue
                    cv2.line(result, tuple(line[:2]), tuple(line[2:]),
                             1, max(5, round(width * 0.012)))
        return np.asarray(cv2.dilate(result, np.ones((9, 9), np.uint8)), dtype=np.uint8)

    def _labelled_bay_sides(
        self, paint: np.ndarray[Any, Any], frame: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Mark only paint segments adjacent to a cluster of enclosed cavities."""
        cv2 = _cv2()
        height, width = paint.shape
        contours, hierarchy = cv2.findContours(
            paint, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        result = self._bay_corner_context(paint, frame, contours)
        if hierarchy is None:
            return result
        label_parents = set()
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] >= 0 or hierarchy[0, index, 2] < 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            interior = np.zeros((h, w), np.uint8)
            cv2.drawContours(interior, [contour - (x, y)], -1, 1, cv2.FILLED)
            # A sparse lattice encloses several holes too. Its empty cells
            # cannot label a separate lane beside the junction as a bay edge.
            # Measure the actual paint, without filling those cells as glyphs.
            if np.mean(paint[y:y+h, x:x+w][interior > 0] > 0) >= 0.55:
                label_parents.add(index)
        cavities = []
        cavity_edges = []

        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] not in label_parents:
                continue
            area = cv2.contourArea(contour)
            if not max(9, width * height * 0.00001) <= area <= width * height * 0.005:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if y + h / 2 >= height * 0.95:
                # Camera captions at the footer are not road-label evidence.
                continue
            cavities.append((x + w / 2, y + h / 2))
            cavity_edges.append(contour[:, 0].astype(float))
        if len(cavities) < 3:
            return result
        centres = np.asarray(cavities)
        groups = []
        remaining = set(range(len(centres)))
        while remaining:
            group = {remaining.pop()}
            frontier = set(group)
            while frontier:
                index = frontier.pop()
                nearby = {
                    other for other in remaining
                    if np.linalg.norm(centres[other] - centres[index]) <= width * 0.08
                }
                remaining -= nearby
                group |= nearby
                frontier |= nearby
            points = centres[sorted(group)]
            if len(group) >= 3 and np.ptp(points[:, 0]) >= width * 0.025:
                edges = np.vstack([cavity_edges[index] for index in sorted(group)])
                groups.append((points.mean(axis=0), points.min(axis=0),
                               points.max(axis=0), edges))
        if not groups:
            return result
        lines = cv2.HoughLinesP(
            paint * 255, 1, np.pi / 180, 20,
            minLineLength=max(20, round(width * 0.025)), maxLineGap=8,
        )
        if lines is None:
            return result
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        strokes = []
        for line in lines[:, 0]:
            points = line.reshape(2, 2).astype(float)
            if points[0, 1] > points[1, 1]:
                points = points[::-1]
            dx, dy = points[1] - points[0]
            if dy <= 0 or abs(dx) > 3 * dy:
                continue
            direction = points[1] - points[0]
            length = float(np.linalg.norm(direction))
            direction /= length
            normal = np.asarray([-direction[1], direction[0]])
            stripe = LaneCurve(
                boundary_id="bay-context", confidence=1.0,
                points=tuple(Point2D(x=float(x), y=float(y))
                             for x, y in np.linspace(points[0], points[1], 16)),
            )
            sustained = self._narrow_stripe_support(
                stripe, frame, gray=gray, maximum_width_ratio=0.025,
            ) >= 0.8
            strokes.append((points, direction, normal, length, sustained))
        glyph_paint = np.zeros_like(paint)
        for index, contour in enumerate(contours):
            if index not in label_parents:
                continue
            solidity = cv2.contourArea(contour) / max(
                cv2.contourArea(cv2.convexHull(contour)), 1,
            )
            if solidity >= 0.3:
                cv2.drawContours(glyph_paint, [contour], -1, 1, cv2.FILLED)
        for centre, lower, upper, edges in groups:
            eligible = []
            for points, direction, normal, length, sustained in strokes:
                relative = centre - points[0]
                signed = float(relative @ normal)
                across = abs(signed)
                label_offsets = (edges - points[0]) @ normal
                label_along = (edges - points[0]) @ direction
                label_corners = np.asarray([
                    lower, upper, [lower[0], upper[1]], [upper[0], lower[1]],
                ])
                corner_offsets = (label_corners - points[0]) @ normal
                outside_box = (
                    np.all(corner_offsets > width * 0.012)
                    or np.all(corner_offsets < -width * 0.012)
                )
                outside_label = (
                    np.all(label_offsets > width * 0.012)
                    or np.all(label_offsets < -width * 0.012)
                )
                samples = np.linspace(points[0], points[1], 16)
                sx = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
                sy = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
                inside_label = np.all(
                    (samples >= lower - width * 0.02)
                    & (samples <= upper + width * 0.02), axis=1,
                )
                if (
                    width * 0.025 <= across <= width * 0.16
                    # The label must lie beside the observed border. Remote
                    # paint ahead of a lane dash cannot supply its semantics.
                    and label_along.max() >= -width * 0.025
                    and label_along.min() <= length + width * 0.025
                    and length >= width * 0.1
                    and sustained
                    and outside_label
                    and (across > width * 0.06 or outside_box)
                    # A connected manhole elsewhere cannot veto this border.
                    # Compact outlines are text evidence only near this label.
                    and (across > width * 0.06
                         or np.mean(glyph_paint[sy, sx] > 0) < 0.5)
                    and np.mean(inside_label) < 0.5
                ):
                    eligible.append((signed, points, direction, normal))
            if eligible:
                label_region = np.zeros_like(paint)
                margin = round(width * 0.08)
                cv2.rectangle(label_region, tuple(np.maximum(0, lower - margin).astype(int)),
                              tuple(np.minimum([width - 1, height - 1],
                                               upper + margin).astype(int)), 1, cv2.FILLED)
                # Labels inside a verified bay are markings too. Do not let
                # their detached thin edges become fresh lane candidates.
                label_paint = glyph_paint & paint & label_region
                result |= cv2.dilate(label_paint, np.ones((9, 9), np.uint8))
            # The nearest border on each side owns this label. A farther
            # parallel lane beside the bay must not inherit its semantics.
            for sign in (-1, 1):
                side = [item for item in eligible if item[0] * sign > 0]
                if not side:
                    continue
                _, parent, axis, normal = min(side, key=lambda item: abs(item[0]))
                for points, direction, _, _, _ in strokes:
                    if abs(float(direction @ axis)) < 0.97:
                        continue
                    if np.max(np.abs((points - parent[0]) @ normal)) > width * 0.012:
                        continue
                    cv2.line(result, tuple(points[0].astype(int)),
                             tuple(points[1].astype(int)), 1, max(5, round(width * 0.012)))
        return result

    @staticmethod
    def _near_bay_extension(
        curve: LaneCurve, marking: np.ndarray[Any, Any],
    ) -> bool:
        """Associate a visible fragment across a bounded worn-paint gap.

        The verified border supplies direction; its mask is never extended
        into missing paint. A separate parallel stripe cannot inherit it.
        """
        cv2 = _cv2()
        width = marking.shape[1]
        points = np.asarray([(p.x, p.y) for p in curve.points])
        direction = points[-1] - points[0]
        length = float(np.linalg.norm(direction))
        if length < 1:
            return False
        direction /= length
        lines = cv2.HoughLinesP(
            marking * 255, 1, np.pi / 180, 50,
            minLineLength=round(width * 0.1), maxLineGap=8,
        )
        if lines is None:
            return False
        for line in lines[:, 0]:
            first, last = line.reshape(2, 2).astype(float)
            axis = last - first
            span = float(np.linalg.norm(axis))
            axis /= span
            if abs(float(direction @ axis)) < 0.97:
                continue
            normal = np.asarray([-axis[1], axis[0]])
            if np.max(np.abs((points - first) @ normal)) > width * 0.012:
                continue
            along = (points - first) @ axis
            if along.max() >= -width * 0.12 and along.min() <= span + width * 0.12:
                return True
        return False

    def _parallel_markings(self, paint: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Find repeated local paint strokes even when a grid connects them.

        At least three separated parallel strokes must overlap along their
        direction. One lane ribbon or separated dashes cannot form this group.
        """
        cv2 = _cv2()
        width = paint.shape[1]
        result = np.zeros_like(paint)
        paint = paint.copy()
        if self.occlusion_mask is not None:
            paint[self.occlusion_mask > 0] = 0
        lines = cv2.HoughLinesP(
            paint * 255, 1, np.pi / 180, 18,
            minLineLength=max(12, round(width * 0.025)), maxLineGap=3,
        )
        if lines is None:
            return result
        strokes: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]] = []
        for line in lines[:, 0]:
            points = line.reshape(2, 2).astype(float)
            vector = points[1] - points[0]
            length = float(np.linalg.norm(vector))
            if not length:
                continue
            if self.occlusion_mask is not None:
                samples = np.rint(np.linspace(
                    points[0], points[1], max(2, round(length)),
                )).astype(int)
                if np.mean(self.occlusion_mask[samples[:, 1], samples[:, 0]] > 0) > 0.5:
                    continue
            strokes.append((points, vector / length, length))
        if not strokes:
            return result
        middles = np.asarray([p.mean(axis=0) for p, _, _ in strokes])
        axes = np.asarray([axis for _, axis, _ in strokes])
        lengths = np.asarray([length for _, _, length in strokes])
        for points, direction, length in strokes:
            normal = np.asarray([-direction[1], direction[0]])
            offsets = middles - points.mean(axis=0)
            across = offsets @ normal
            eligible = np.flatnonzero(
                (np.abs(axes @ direction) >= 0.97)
                & (np.maximum(lengths, length) <= np.minimum(lengths, length) * 3)
                & (np.abs(offsets @ direction) <= np.maximum(lengths, length) * 0.7)
                & (np.abs(across) >= width * 0.008)
                & (np.abs(across) <= width * 0.06)
            )
            neighbours: list[float] = []
            projection = points @ direction
            for index in eligible:
                other, _, other_length = strokes[index]
                other_projection = other @ direction
                start = max(projection.min(), other_projection.min())
                stop = min(projection.max(), other_projection.max())
                if stop - start < min(length, other_length) * 0.5:
                    continue
                middle = (start + stop) / 2
                first = points[0] + (points[1] - points[0]) * (
                    (middle - projection[0]) / (projection[1] - projection[0])
                )
                second = other[0] + (other[1] - other[0]) * (
                    (middle - other_projection[0])
                    / (other_projection[1] - other_projection[0])
                )
                gap = float((second - first) @ normal)
                if min(length, other_length) < abs(gap) * 1.5:
                    continue
                samples = np.rint(np.linspace(first, second, 11)[3:8]).astype(int)
                # Separate paint bands have pavement between them. Hough edges
                # of ONE broad/tapered ribbon must not count as multiple bars.
                if np.mean(paint[samples[:, 1], samples[:, 0]] > 0) >= 0.4:
                    continue
                if all(abs(gap - existing) >= width * 0.008 for existing in neighbours):
                    neighbours.append(gap)
                    # Only the existence of two distinct neighbours is used.
                    # Later additions cannot change this stroke's decision.
                    if len(neighbours) >= 2:
                        break
            if len(neighbours) >= 2:
                cv2.line(result, tuple(np.rint(points[0]).astype(int)),
                         tuple(np.rint(points[1]).astype(int)), 1,
                         max(3, round(width * 0.012)))
        return result

    @staticmethod
    def _mask_overlap(curve: LaneCurve, mask: np.ndarray[Any, Any]) -> float:
        """Sample the whole polyline, including between its stored points."""
        height, width = mask.shape
        points = np.asarray([(p.x, p.y) for p in curve.points])
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distances = np.r_[0.0, np.cumsum(lengths)]
        samples = np.linspace(0, distances[-1], max(2, round(distances[-1] / 2) + 1))
        xs = np.rint(np.interp(samples, distances, points[:, 0])).astype(int)
        ys = np.rint(np.interp(samples, distances, points[:, 1])).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        hits = np.zeros(len(samples), dtype=bool)
        hits[inside] = mask[ys[inside], xs[inside]] > 0
        return float(np.mean(hits))

    def _context_rejection(
        self,
        curve: LaneCurve,
        paint: np.ndarray[Any, Any],
        rejected_paint: np.ndarray[Any, Any],
        drivable: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any] | None = None,
    ) -> str | None:
        height, width = paint.shape
        markings: dict[str, np.ndarray[Any, Any]] = getattr(self, "_marking_masks", {})
        for reason, marking in markings.items():
            if reason == "repeated_marking_context":
                continue
            if self._mask_overlap(curve, marking) > 0.2:
                return reason
            if (
                reason == "labelled_bay_context" and frame is not None
                and self._narrow_stripe_support(curve, frame) >= 0.75
                and self._near_bay_extension(curve, marking)
            ):
                return reason
        points = np.asarray([(p.x, p.y) for p in curve.points])
        dx, dy = points[-1] - points[0]
        # Test convergence into the scene, not "vertical vs horizontal" in the
        # image. Outer longitudinal lanes can have very shallow image slopes.
        if dy <= 0:
            return "transverse_marking"
        horizon_x = points[0, 0] + dx / dy * (height * self._roi_top - points[0, 1])
        if abs(dx) > 2 * dy and not -0.1 * width <= horizon_x <= 1.1 * width:
            return "transverse_marking"
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        if np.mean(rejected_paint[ys, xs] > 0) > 0.35:
            contaminated = getattr(self, "_contaminated_paint", None)
            if not (
                frame is not None and contaminated is not None
                and self._mask_overlap(curve, contaminated) > 0.5
                and self._narrow_stripe_support(curve, frame) >= 0.75
            ):
                return "arrow_paint_context"
        repeated = markings.get("repeated_marking_context")
        if (
            repeated is not None
            and (curve.component_height_ratio or 0) < self._minimum_span
            and self._mask_overlap(curve, repeated) > 0.2
        ):
            return "repeated_marking_context"
        tangent = np.gradient(points, axis=0)
        length = np.maximum(np.linalg.norm(tangent, axis=1), 1.0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / length[:, None]
        sides = []
        for sign in (-1, 1):
            samples = points[:, None, :] + sign * normal[:, None, :] * np.asarray(
                [width * 0.012, width * 0.024, width * 0.036]
            )[None, :, None]
            side_x = np.clip(np.rint(samples[:, :, 0]).astype(int), 0, width - 1)
            side_y = np.clip(np.rint(samples[:, :, 1]).astype(int), 0, height - 1)
            sides.append(float(np.mean(drivable[side_y, side_x] > 0.5)))
        paint_ridge = self._paint_ridge_support(points, normal, frame) if frame is not None else 0
        if (
            frame is not None
            and min(sides) < 0.3
            and self._colored_curb_support(points, normal, frame) >= 0.4
        ):
            return "colored_curb_context"
        # A shallow bright edge can also be a curb. Only a sustained painted
        # stripe extending into the scene may override a semantic road boundary.
        painted_boundary = (
            paint_ridge >= 0.8
            and abs(dx) <= 2 * dy
        ) or (
            frame is not None and paint_ridge >= 0.9
            and self._narrow_stripe_support(curve, frame) >= 0.9
            and width * 0.25 <= horizon_x <= width * 0.75
        ) or (
            frame is not None and max(sides) < 0.3
            and self._narrow_stripe_support(curve, frame) >= 0.75
        )
        if min(sides) < 0.3 and max(sides) > 0.7 and not painted_boundary:
            return "road_edge_context"
        if (
            (curve.drivable_probability or 0) < self._minimum_drivable
            and min(sides) < 0.6
            and not painted_boundary
        ):
            return "low_drivable_context"
        if (
            frame is not None
            and dy < height * 0.15
            and abs(dx) > 1.5 * dy
            and paint_ridge < 0.7
        ):
            # A fitted arrow shaft may sit just outside its classified paint.
            # Weak contrast alone also describes a real, dim dashed lane: require
            # nearby symbol evidence unless there is almost no stripe support.
            radius = max(1, round(width * 0.016))
            nearby_symbols = _cv2().dilate(
                rejected_paint, np.ones((radius * 2 + 1,) * 2, np.uint8)
            )
            if paint_ridge < 0.2 or np.mean(nearby_symbols[ys, xs] > 0) >= 0.3:
                return "weak_transverse_paint"
        if (
            (curve.component_width_ratio or 0) * width
            > (curve.component_height_ratio or 0) * height
            * self._maximum_horizontal_to_vertical
            and float(np.mean(paint[ys, xs] > 0)) < 0.7
        ):
            return "wide_fragment_without_paint"
        return None

    @staticmethod
    def _narrow_stripe_support(
        curve: LaneCurve, frame: np.ndarray[Any, Any],
        *, gray: np.ndarray[Any, Any] | None = None,
        maximum_width_ratio: float = 0.012,
    ) -> float:
        """Require a narrow bright ribbon with dark pavement on both sides.

        Inspect perpendicular widths rather than image-row widths, which grow
        for shallow real lanes. A brightness step and a broad arrow shaft fail.
        """
        cv2 = _cv2()
        height, width = frame.shape[:2]
        if gray is None:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        points = np.asarray([(p.x, p.y) for p in curve.points])
        tangent = np.gradient(points, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.maximum(
            np.linalg.norm(tangent, axis=1), 1.0
        )[:, None]
        radius = max(6, round(width * 0.025))
        offsets = np.arange(-radius, radius + 1)
        samples = points[:, None, :] + normal[:, None, :] * offsets[None, :, None]
        xs = np.clip(np.rint(samples[:, :, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(samples[:, :, 1]).astype(int), 0, height - 1)
        values = gray[ys, xs]
        # Compare against nearby pavement. Distant dark pixels can turn a local
        # glare patch plus a real stripe into one apparently broad white ribbon.
        near_ratio = 0.006 if maximum_width_ratio <= 0.012 else maximum_width_ratio * 0.65
        near = max(2, round(width * near_ratio))
        far = min(radius, max(near + 1, near * 2))
        flank = np.maximum(
            np.median(values[:, (offsets >= -far) & (offsets <= -near)], axis=1),
            np.median(values[:, (offsets >= near) & (offsets <= far)], axis=1),
        )
        bright = values > flank[:, None] + 18
        support = []
        for row in bright:
            runs = np.flatnonzero(np.diff(np.r_[False, row, False]))
            support.append(any(
                1 <= stop - start <= max(3, width * maximum_width_ratio)
                and start <= radius + width * 0.006
                and stop >= radius - width * 0.006
                for start, stop in zip(runs[::2], runs[1::2], strict=True)
            ))
        return float(np.mean(support))

    @staticmethod
    def _colored_curb_support(
        points: np.ndarray[Any, Any],
        normal: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any],
    ) -> float:
        """Measure sustained red/yellow curb paint next to a semantic road edge."""
        cv2 = _cv2()
        height, width = frame.shape[:2]
        hls = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2HLS)
        offsets = np.linspace(-0.03, 0.03, 25) * width
        samples = points[:, None, :] + normal[:, None, :] * offsets[None, :, None]
        xs = np.clip(np.rint(samples[..., 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(samples[..., 1]).astype(int), 0, height - 1)
        colors = hls[ys, xs]
        colored = (
            ((colors[..., 0] <= 45) | (colors[..., 0] >= 170))
            & (colors[..., 2] >= 45)
            & (colors[..., 1] >= 35)
            & (colors[..., 1] <= 220)
        )
        return float(np.mean(np.sum(colored, axis=1) >= 2))

    @staticmethod
    def _paint_ridge_support(
        points: np.ndarray[Any, Any],
        normal: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any],
    ) -> float:
        """A painted stripe has a bright center with darker pavement on BOTH sides."""
        return float(np.mean(YoloPLaneLineDetector._paint_ridge_profile(points, normal, frame)))

    @staticmethod
    def _paint_ridge_profile(
        points: np.ndarray[Any, Any],
        normal: np.ndarray[Any, Any],
        frame: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        cv2 = _cv2()
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        height, width = gray.shape
        peaks = np.linspace(-width * 0.012, width * 0.012, 11)
        flanks = np.asarray([-0.022, -0.016, -0.009, 0.009, 0.016, 0.022]) * width
        centers = points[:, None, :] + normal[:, None, :] * peaks[None, :, None]

        def sample(coordinates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            xs = np.clip(np.rint(coordinates[..., 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(coordinates[..., 1]).astype(int), 0, height - 1)
            return np.asarray(gray[ys, xs])

        center = sample(centers)
        side_values = sample(
            centers[:, :, None, :] + normal[:, None, None, :] * flanks[None, None, :, None]
        )
        darker_sides = np.maximum(side_values[:, :, :3].mean(axis=2),
                                 side_values[:, :, 3:].mean(axis=2))
        return np.asarray(np.max(center - darker_sides, axis=1) >= 35)

    def _painted_tail(self, curve: LaneCurve, frame: np.ndarray[Any, Any]) -> LaneCurve | None:
        """Trim an unsupported end, never bridge separate bright patches."""
        points = np.asarray([(p.x, p.y) for p in curve.points])
        tangent = np.gradient(points, axis=0)
        length = np.maximum(np.linalg.norm(tangent, axis=1), 1.0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / length[:, None]
        support = self._paint_ridge_profile(points, normal, frame)
        runs = np.flatnonzero(np.diff(np.r_[False, support, False]))
        for start, stop in zip(runs[::2], runs[1::2], strict=True):
            if start > 1 and stop < len(points) - 1:
                continue
            if stop - start < max(4, len(points) * 0.4):
                continue
            if np.linalg.norm(points[stop - 1] - points[start]) < frame.shape[1] * 0.025:
                continue
            # Component measurements still describe the original model pixels.
            return curve.model_copy(update={"points": curve.points[start:stop]})
        return None

    def _refine_paint_geometry(
        self, curve: LaneCurve, frame: np.ndarray[Any, Any],
        *, minimum_length: float | None = None,
    ) -> LaneCurve:
        """Fit visible narrow paint, rather than a bent segmentation halo.

        Nearby pavement can join separate dashes in the model mask. Sample
        perpendicular bright ribbons and require a consensus before replacing
        geometry. No visible paint means no extrapolation or invented line.
        Curved paint without a straight consensus keeps its original geometry.
        """
        cv2 = _cv2()
        height, width = frame.shape[:2]
        if minimum_length is None:
            minimum_length = width * 0.025
        points = np.asarray([(p.x, p.y) for p in curve.points])
        if len(points) < 6:
            return curve
        tangent = np.gradient(points, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.maximum(
            np.linalg.norm(tangent, axis=1), 1.0,
        )[:, None]
        radius = max(8, round(width * 0.025))
        offsets = np.arange(-radius, radius + 1)
        samples = points[:, None, :] + normal[:, None, :] * offsets[None, :, None]
        xs = np.clip(np.rint(samples[..., 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(samples[..., 1]).astype(int), 0, height - 1)
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        values = gray[ys, xs].astype(np.float32)
        near = max(3, round(width * 0.012))
        flanks = np.maximum(
            np.median(values[:, offsets <= -near], axis=1),
            np.median(values[:, offsets >= near], axis=1),
        )
        bright = (values >= self._white_lightness) & (values > flanks[:, None] + 25)
        occlusion = getattr(self, "occlusion_mask", None)
        if occlusion is not None:
            bright[occlusion[ys, xs] > 0] = False
        observed = []
        for index, row in enumerate(bright):
            transitions = np.flatnonzero(np.diff(np.r_[False, row, False]))
            centers = [
                (start + stop - 1) / 2 - radius
                for start, stop in zip(transitions[::2], transitions[1::2], strict=True)
                # A nearby lane dash is wider than its distant end. Do not
                # truncate real paint merely because it exceeds the far width.
                if 2 <= stop - start <= max(4, width * 0.035)
                and start > 0 and stop < len(row)
            ]
            if centers:
                offset = min(centers, key=abs)
                observed.append(points[index] + normal[index] * offset)
        if len(observed) < max(6, len(points) * 0.4):
            return curve
        paint_points = np.asarray(observed)
        tolerance = max(1.5, width * 0.0025)
        best = np.zeros(len(paint_points), dtype=bool)
        # A deterministic small consensus fit avoids sensitivity to one bright
        # vehicle edge and does not use video names, positions, or frame IDs.
        for index, first in enumerate(paint_points):
            for last in paint_points[index + 1:]:
                direction = last - first
                length = float(np.linalg.norm(direction))
                if length < minimum_length:
                    continue
                relative = paint_points - first
                distances = np.abs(
                    relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
                ) / length
                inliers = distances <= tolerance
                if inliers.sum() > best.sum():
                    best = inliers
        if best.sum() < max(6, len(paint_points) * 0.75):
            return curve
        visible = paint_points[best]
        visible = visible[np.argsort(visible[:, 1])]
        if (
            np.ptp(visible[:, 1]) < 4
            or np.linalg.norm(visible[-1] - visible[0]) < minimum_length
        ):
            return curve
        coefficients = np.polyfit(visible[:, 1], visible[:, 0], 1)
        sample_ys = np.linspace(visible[0, 1], visible[-1, 1], self._sample_count)
        fitted_xs = np.polyval(coefficients, sample_ys)
        return curve.model_copy(update={
            "points": tuple(Point2D(x=float(x), y=float(y))
                            for x, y in zip(fitted_xs, sample_ys, strict=True)),
        })

    def _component_curves(
        self,
        mask: np.ndarray[Any, Any],
        probability: np.ndarray[Any, Any],
        drivable_probability: np.ndarray[Any, Any],
        *,
        use_context: bool = False,
    ) -> tuple[list[LaneCurve], Counter[str], int]:
        cv2 = _cv2()
        height, width = mask.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[LaneCurve] = []
        rejection_reasons: Counter[str] = Counter()
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            top = int(stats[label, cv2.CC_STAT_TOP])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            if area < height * width * self._minimum_area:
                rejection_reasons["too_small"] += 1
                continue
            if component_height < height * self._minimum_fragment_span * 0.4:
                rejection_reasons["too_short"] += 1
                continue
            if component_width > component_height * (
                max(12.0, self._maximum_horizontal_to_vertical)
                if use_context else self._maximum_horizontal_to_vertical
            ):
                rejection_reasons["too_wide"] += 1
                continue
            bottom = top + component_height - 1
            points: list[Point2D] = []
            row_widths: list[float] = []
            for y_value in np.linspace(top, bottom, self._sample_count):
                y = int(round(float(y_value)))
                band_top = max(top, y - 2)
                band_bottom = min(bottom + 1, y + 3)
                _, xs = np.where(labels[band_top:band_bottom] == label)
                if xs.size:
                    points.append(Point2D(x=float(np.median(xs)), y=float(y)))
                row_xs = np.flatnonzero(labels[y] == label)
                if row_xs.size:
                    row_widths.append(float(row_xs[-1] - row_xs[0] + 1))
            if len(points) < max(4, self._sample_count // 3) or not row_widths:
                rejection_reasons["insufficient_points"] += 1
                continue
            median_row_width = float(np.median(row_widths))
            maximum_row_width = float(np.percentile(row_widths, 90))
            row_width_variation = maximum_row_width / max(median_row_width, 1.0)
            if (
                maximum_row_width / width > self._maximum_row_width
                and row_width_variation > self._maximum_row_width_variation
            ):
                rejection_reasons["arrow_like_width"] += 1
                continue
            point_ys = np.asarray([point.y for point in points], dtype=np.float64)
            point_xs = np.asarray([point.x for point in points], dtype=np.float64)
            coefficients = np.polyfit(point_ys, point_xs, deg=min(2, len(points) - 1))
            fitted_xs = np.polyval(coefficients, point_ys)
            fit_error = float(np.sqrt(np.mean(np.square(point_xs - fitted_xs))))
            if fit_error > width * self._maximum_fit_error:
                rejection_reasons["poor_fit"] += 1
                continue
            aspect_ratio = component_width / component_height
            if (
                aspect_ratio > 2.0
                and fit_error > width * self._maximum_arrow_fit_error
            ):
                rejection_reasons["arrow_like_fit"] += 1
                continue
            if component_height < height * self._minimum_span:
                stable_short_fragment = (
                    maximum_row_width / width <= self._maximum_row_width * (
                        1.0 if use_context else 0.5
                    )
                    and row_width_variation <= self._maximum_row_width_variation
                    and fit_error <= width * self._maximum_arrow_fit_error
                )
                if not stable_short_fragment:
                    rejection_reasons["unstable_short_fragment"] += 1
                    continue
            confidence = float(probability[labels == label].mean())
            if confidence < self._minimum_confidence:
                rejection_reasons["low_lane_probability"] += 1
                continue
            drivable_support = float(drivable_probability[labels == label].mean())
            if drivable_support < self._minimum_drivable and not use_context:
                rejection_reasons["low_drivable_probability"] += 1
                continue
            candidates.append(
                LaneCurve(
                    boundary_id="candidate",
                    points=tuple(
                        Point2D(x=float(x), y=float(y))
                        for x, y in zip(fitted_xs, point_ys, strict=True)
                    ),
                    confidence=confidence,
                    lane_probability=confidence,
                    drivable_probability=drivable_support,
                    fit_error_ratio=fit_error / width,
                    component_area_ratio=area / (height * width),
                    component_width_ratio=component_width / width,
                    component_height_ratio=component_height / height,
                    median_row_width_ratio=median_row_width / width,
                    maximum_row_width_ratio=maximum_row_width / width,
                    row_width_variation_ratio=row_width_variation,
                )
            )
        candidates.sort(key=lambda curve: curve.confidence, reverse=True)
        return candidates, rejection_reasons, count - 1

    def _merge_fragments(
        self,
        candidates: list[LaneCurve],
        width: int,
        height: int,
        frame: np.ndarray[Any, Any] | None = None,
    ) -> list[LaneCurve]:
        """合併短距離遮擋前後、方向一致的白線片段。"""
        merged = list(candidates)
        while True:
            best: tuple[float, int, int, LaneCurve] | None = None
            for first_index, first in enumerate(merged):
                for second_index in range(first_index + 1, len(merged)):
                    combined = self._try_merge_fragments(
                        first,
                        merged[second_index],
                        width,
                        height,
                        frame,
                    )
                    if combined is None:
                        continue
                    score = combined.fit_error_ratio or 0.0
                    if best is None or score < best[0]:
                        best = (score, first_index, second_index, combined)
            if best is None:
                return merged
            _, first_index, second_index, combined = best
            merged.pop(second_index)
            merged.pop(first_index)
            merged.append(combined)

    def _try_merge_fragments(
        self,
        first: LaneCurve,
        second: LaneCurve,
        width: int,
        height: int,
        frame: np.ndarray[Any, Any] | None = None,
    ) -> LaneCurve | None:
        first_top = first.points[0].y
        second_top = second.points[0].y
        upper, lower = (first, second) if first_top <= second_top else (second, first)
        gap = lower.points[0].y - upper.points[-1].y
        if gap < 0 or gap > height * self._maximum_fragment_gap:
            return None
        upper_is_fragment = (upper.component_height_ratio or 1.0) < self._minimum_span
        lower_is_fragment = (lower.component_height_ratio or 1.0) < self._minimum_span
        if not upper_is_fragment and not lower_is_fragment:
            return None

        upper_slope = self._endpoint_slope(upper.points[-2], upper.points[-1])
        lower_slope = self._endpoint_slope(lower.points[0], lower.points[1])
        if (
            frame is not None and gap > 0
            and lower.points[-1].y - upper.points[0].y >= height * self._minimum_span
        ):
            # Contrast can remove the centre of wide paint. Only an actually
            # painted gap may ignore a small fragment's noisy endpoint tangent.
            points = np.linspace(
                [upper.points[-1].x, upper.points[-1].y],
                [lower.points[0].x, lower.points[0].y],
                max(8, round(gap / 2)),
            )
            direction = points[-1] - points[0]
            normal = np.broadcast_to(
                [-direction[1], direction[0]] / np.linalg.norm(direction), points.shape,
            )
            xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
            visible = bool(np.all(
                (points[:, 0] >= 0) & (points[:, 0] < width)
                & (points[:, 1] >= 0) & (points[:, 1] < height)
            ))
            if self.occlusion_mask is not None:
                visible = visible and not np.any(self.occlusion_mask[ys, xs])
            gray = _cv2().cvtColor(frame[:, :, :3], _cv2().COLOR_BGR2GRAY).astype(np.float32)
            flanks = points[:, None, :] + normal[:, None, :] * (
                np.asarray([-0.03, -0.022, -0.016, 0.016, 0.022, 0.03]) * width
            )[None, :, None]
            side_x = np.clip(np.rint(flanks[:, :, 0]).astype(int), 0, width - 1)
            side_y = np.clip(np.rint(flanks[:, :, 1]).astype(int), 0, height - 1)
            side = gray[side_y, side_x]
            painted = (
                (gray[ys, xs] >= self._white_lightness)
                & (gray[ys, xs] > np.maximum(side[:, :3].mean(1), side[:, 3:].mean(1)) + 35)
            )
            if visible and np.mean(painted) >= 0.9:
                upper_slope = float(np.median([
                    self._endpoint_slope(a, b)
                    for a, b in zip(upper.points[:-1], upper.points[1:], strict=True)
                ]))
                lower_slope = float(np.median([
                    self._endpoint_slope(a, b)
                    for a, b in zip(lower.points[:-1], lower.points[1:], strict=True)
                ]))
        predicted_x = upper.points[-1].x + upper_slope * gap
        # Measure perpendicular alignment in BOTH directions. The temporal
        # association radius is far too permissive for joining paint fragments.
        tolerance = width * 0.012
        if abs(predicted_x - lower.points[0].x) / np.hypot(upper_slope, 1) > tolerance:
            return None
        predicted_upper_x = lower.points[0].x - lower_slope * gap
        if abs(predicted_upper_x - upper.points[-1].x) / np.hypot(lower_slope, 1) > tolerance:
            return None
        if abs(upper_slope - lower_slope) > 1.0:
            return None

        source_points = tuple(sorted((*upper.points, *lower.points), key=lambda point: point.y))
        source_ys = np.asarray([point.y for point in source_points], dtype=np.float64)
        source_xs = np.asarray([point.x for point in source_points], dtype=np.float64)
        coefficients = np.polyfit(source_ys, source_xs, deg=2)
        fitted_source_xs = np.polyval(coefficients, source_ys)
        fit_error = float(np.sqrt(np.mean(np.square(source_xs - fitted_source_xs))))
        if fit_error > width * self._maximum_arrow_fit_error:
            return None
        sample_ys = np.linspace(source_ys[0], source_ys[-1], self._sample_count)
        fitted_xs = np.polyval(coefficients, sample_ys)
        upper_area = upper.component_area_ratio or 0.0
        lower_area = lower.component_area_ratio or 0.0
        total_area = max(upper_area + lower_area, np.finfo(float).eps)

        def weighted(attribute: str) -> float | None:
            upper_value = getattr(upper, attribute)
            lower_value = getattr(lower, attribute)
            if upper_value is None or lower_value is None:
                return None
            return float(
                (upper_value * upper_area + lower_value * lower_area) / total_area
            )

        median_row_width = weighted("median_row_width_ratio")
        maximum_row_width = max(
            upper.maximum_row_width_ratio or 0.0,
            lower.maximum_row_width_ratio or 0.0,
        )
        return LaneCurve(
            boundary_id="candidate",
            points=tuple(
                Point2D(x=float(x), y=float(y))
                for x, y in zip(fitted_xs, sample_ys, strict=True)
            ),
            confidence=weighted("confidence") or min(upper.confidence, lower.confidence),
            lane_probability=weighted("lane_probability"),
            drivable_probability=weighted("drivable_probability"),
            fit_error_ratio=fit_error / width,
            component_area_ratio=min(1.0, upper_area + lower_area),
            component_width_ratio=float((fitted_xs.max() - fitted_xs.min()) / width),
            component_height_ratio=float((source_ys[-1] - source_ys[0]) / height),
            median_row_width_ratio=median_row_width,
            maximum_row_width_ratio=maximum_row_width,
            row_width_variation_ratio=(
                maximum_row_width / max(median_row_width or 0.0, np.finfo(float).eps)
            ),
        )

    @staticmethod
    def _endpoint_slope(first: Point2D, second: Point2D) -> float:
        delta_y = second.y - first.y
        if delta_y == 0:
            return 0.0
        return (second.x - first.x) / delta_y

    def _deduplicate(
        self, candidates: list[LaneCurve], width: int,
        frame: np.ndarray[Any, Any] | None = None,
    ) -> list[LaneCurve]:
        """Prefer paint-aligned candidates over a high-confidence pavement halo."""
        selected: list[LaneCurve] = []
        duplicate_distance = self._association_distance * 0.5
        gray = background = None
        if frame is not None:
            gray = _cv2().cvtColor(frame[:, :, :3], _cv2().COLOR_BGR2GRAY)
            background = _cv2().GaussianBlur(gray, (0, 0), 9).astype(np.float32)

        def priority(curve: LaneCurve) -> tuple[float, float]:
            if gray is None or background is None:
                return 0.0, curve.confidence
            xs = np.clip(np.rint([p.x for p in curve.points]).astype(int), 0, width - 1)
            ys = np.clip(np.rint([p.y for p in curve.points]).astype(int), 0, gray.shape[0] - 1)
            supported = (gray[ys, xs] >= self._white_lightness) & (
                gray[ys, xs] > background[ys, xs] + 20
            )
            return float(supported.mean()), curve.confidence

        for candidate in sorted(candidates, key=priority, reverse=True):
            if any(
                self._curve_distance(candidate, existing, width) <= duplicate_distance
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= self._maximum_boundaries:
                break
        return selected

    def _associate(self, candidates: list[LaneCurve], width: int) -> list[LaneCurve]:
        unmatched_previous = set(self._previous)
        unmatched_candidates = set(range(len(candidates)))
        matches: dict[int, str] = {}
        possible_matches = sorted(
            (
                (self._association_cost(candidate, previous, width), index, previous_id)
                for index, candidate in enumerate(candidates)
                for previous_id, previous in self._previous.items()
            ),
            key=lambda match: match[0],
        )
        for distance, index, previous_id in possible_matches:
            if distance > self._association_distance:
                break
            if index not in unmatched_candidates or previous_id not in unmatched_previous:
                continue
            matches[index] = previous_id
            unmatched_candidates.remove(index)
            unmatched_previous.remove(previous_id)

        output: list[LaneCurve] = []
        for index, candidate in enumerate(candidates):
            boundary_id = matches.get(index)
            if boundary_id is None:
                boundary_id = f"lane-{self._next_boundary_id}"
                self._next_boundary_id += 1
                self._ages[boundary_id] = 1
                curve = candidate.model_copy(
                    update={
                        "boundary_id": boundary_id,
                        "confirmed_frames": 1,
                    }
                )
            else:
                self._ages[boundary_id] = self._ages.get(boundary_id, 1) + 1
                curve = self._smooth(
                    self._previous[boundary_id],
                    candidate,
                    boundary_id,
                    self._ages[boundary_id],
                    width,
                )
            self._previous[boundary_id] = curve
            self._missing[boundary_id] = 0
            output.append(curve)
        return output

    def _association_cost(self, first: LaneCurve, second: LaneCurve, width: int) -> float:
        distance = self._curve_distance(first, second, width)
        if not np.isfinite(distance):
            return distance
        top = max(first.points[0].y, second.points[0].y)
        bottom = min(first.points[-1].y, second.points[-1].y)
        ys = np.linspace(top, bottom, min(self._sample_count, 12))
        directions = []
        for curve in (first, second):
            xs = np.interp(ys, [p.y for p in curve.points], [p.x for p in curve.points])
            directions.append(np.arctan2(np.diff(xs), np.diff(ys)))
        # Small bounded smoothing steps can produce a brief sharp tangent on
        # a short dash. Its predominant direction must remain consistent; the
        # shared-curve distance still compares all sampled positions.
        if np.median(np.abs(directions[0] - directions[1])) > np.deg2rad(20):
            return float("inf")
        return distance

    def _curve_distance(
        self,
        first: LaneCurve,
        second: LaneCurve,
        width: int,
    ) -> float:
        """以共同垂直範圍內的曲線均方距離回傳正規化差異。"""
        first_points = np.asarray([(point.y, point.x) for point in first.points])
        second_points = np.asarray([(point.y, point.x) for point in second.points])
        top = max(float(first_points[0, 0]), float(second_points[0, 0]))
        bottom = min(float(first_points[-1, 0]), float(second_points[-1, 0]))
        first_span = float(first_points[-1, 0] - first_points[0, 0])
        second_span = float(second_points[-1, 0] - second_points[0, 0])
        minimum_overlap = min(first_span, second_span) * 0.35
        if bottom <= top or bottom - top < minimum_overlap:
            return float("inf")
        sample_ys = np.linspace(top, bottom, min(self._sample_count, 12))
        first_xs = np.interp(sample_ys, first_points[:, 0], first_points[:, 1])
        second_xs = np.interp(sample_ys, second_points[:, 0], second_points[:, 1])
        return float(np.sqrt(np.mean(np.square(first_xs - second_xs))) / width)

    def _remove_track(self, boundary_id: str) -> None:
        self._previous.pop(boundary_id, None)
        self._missing.pop(boundary_id, None)
        self._ages.pop(boundary_id, None)

    def _smooth(
        self,
        previous: LaneCurve,
        current: LaneCurve,
        boundary_id: str,
        confirmed_frames: int,
        width: int,
    ) -> LaneCurve:
        previous_points = np.asarray([(point.y, point.x) for point in previous.points])
        # Do not let temporal averaging drag a thin stripe onto nearby asphalt.
        maximum_shift = max(1.0, (current.median_row_width_ratio or 0.0) * width / 2)
        points = tuple(
            Point2D(
                x=(
                    point.x + float(np.clip(
                        (float(np.interp(point.y, previous_points[:, 0], previous_points[:, 1]))
                         - point.x) * (1 - self._alpha),
                        -maximum_shift,
                        maximum_shift,
                    ))
                    if previous_points[0, 0] <= point.y <= previous_points[-1, 0]
                    else point.x
                ),
                y=point.y,
            )
            for point in current.points
        )
        return LaneCurve(
            boundary_id=boundary_id,
            points=points,
            confidence=(
                previous.confidence * (1 - self._alpha)
                + current.confidence * self._alpha
            ),
            confirmed_frames=confirmed_frames,
            lane_probability=current.lane_probability,
            drivable_probability=current.drivable_probability,
            fit_error_ratio=current.fit_error_ratio,
            component_area_ratio=current.component_area_ratio,
            component_width_ratio=current.component_width_ratio,
            component_height_ratio=current.component_height_ratio,
            median_row_width_ratio=current.median_row_width_ratio,
            maximum_row_width_ratio=current.maximum_row_width_ratio,
            row_width_variation_ratio=current.row_width_variation_ratio,
        )

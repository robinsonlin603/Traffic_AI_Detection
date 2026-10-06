from collections import Counter

import numpy as np
import pytest

from dashcam_ai.domain.geometry import BBox, Point2D
from dashcam_ai.domain.lane_lines import LaneCurve, LaneLineStatus
from dashcam_ai.lane.segmentation import YoloPLaneLineDetector


def _detector() -> YoloPLaneLineDetector:
    detector = object.__new__(YoloPLaneLineDetector)
    detector._sample_count = 24
    detector._alpha = 0.35
    detector._minimum_area = 0.00004
    detector._minimum_span = 0.1
    detector._minimum_fragment_span = 0.05
    detector._maximum_horizontal_to_vertical = 5.0
    detector._maximum_fit_error = 0.025
    detector._maximum_arrow_fit_error = 0.008
    detector._maximum_row_width = 0.08
    detector._maximum_row_width_variation = 3.5
    detector._maximum_fragment_gap = 0.12
    detector._minimum_drivable = 0.5
    detector._maximum_missing = 1
    detector._minimum_confidence = 0.5
    detector._maximum_boundaries = 8
    detector._association_distance = 0.08
    detector._minimum_confirmation = 2
    detector._previous = {}
    detector._missing = {}
    detector._ages = {}
    detector._next_boundary_id = 1
    detector._threshold = 0.5
    detector._strong_probability = 0.6
    detector._white_lightness = 90
    detector._white_saturation = 125
    detector._local_contrast = 12
    detector._roi_top = 0.4
    detector._bbox_margin = 0.02
    detector.occlusion_mask = None
    return detector


def _curve(
    start_x: float,
    end_x: float,
    *,
    confidence: float = 0.6,
    top: float = 300,
    bottom: float = 700,
) -> LaneCurve:
    return LaneCurve(
        boundary_id="candidate",
        points=(
            Point2D(x=start_x, y=top),
            Point2D(x=(start_x + end_x) / 2, y=(top + bottom) / 2),
            Point2D(x=end_x, y=bottom),
        ),
        confidence=confidence,
        lane_probability=confidence,
        drivable_probability=0.8,
        fit_error_ratio=0.01,
        component_area_ratio=0.001,
        component_width_ratio=abs(end_x - start_x) / 1280,
        component_height_ratio=(bottom - top) / 720,
        median_row_width_ratio=0.01,
        maximum_row_width_ratio=0.02,
        row_width_variation_ratio=2.0,
    )


def test_unconfirmed_curve_is_not_carried() -> None:
    detector = _detector()

    first = detector._stabilize([_curve(500, 600)], 1280)
    second = detector._stabilize([], 1280)

    assert first.status is LaneLineStatus.UNKNOWN
    assert second.status is LaneLineStatus.UNKNOWN
    assert detector._previous == {}


def test_confirmed_curve_is_carried_once_with_diagnostics() -> None:
    detector = _detector()

    detector._stabilize([_curve(500, 600)], 1280)
    confirmed = detector._stabilize([_curve(502, 602)], 1280)
    carried = detector._stabilize([], 1280)
    expired = detector._stabilize([], 1280)

    assert confirmed.curves[0].confirmed_frames == 2
    assert confirmed.curves[0].lane_probability == 0.6
    assert carried.curves[0].carried_frames == 1
    assert carried.curves[0].confidence == pytest.approx(0.54)
    assert expired.status is LaneLineStatus.UNKNOWN


def test_carried_curve_must_pass_decayed_confidence_floor() -> None:
    detector = _detector()

    detector._stabilize([_curve(500, 600, confidence=0.54)], 1280)
    detector._stabilize([_curve(502, 602, confidence=0.54)], 1280)

    result = detector._stabilize([], 1280)

    assert result.status is LaneLineStatus.UNKNOWN


def test_association_compares_the_whole_shared_curve() -> None:
    detector = _detector()
    first = detector._associate([_curve(100, 500)], 1280)[0]

    second = detector._associate([_curve(500, 500)], 1280)[0]

    assert first.boundary_id != second.boundary_id


def test_near_duplicate_candidates_are_suppressed() -> None:
    detector = _detector()

    selected = detector._deduplicate(
        [
            _curve(500, 600, confidence=0.61),
            _curve(505, 610, confidence=0.59),
            _curve(800, 1000, confidence=0.6),
        ],
        1280,
    )

    assert [curve.confidence for curve in selected] == [0.61, 0.6]


def test_fresh_curve_removes_overlapping_carried_track() -> None:
    detector = _detector()
    stale = _curve(500, 600).model_copy(
        update={"boundary_id": "lane-1", "confirmed_frames": 3}
    )
    active = _curve(510, 610).model_copy(
        update={"boundary_id": "lane-2", "confirmed_frames": 3}
    )
    detector._previous = {"lane-1": stale, "lane-2": active}
    detector._missing = {"lane-1": 0, "lane-2": 0}
    detector._ages = {"lane-1": 3, "lane-2": 3}
    detector._next_boundary_id = 3

    result = detector._stabilize([_curve(512, 612)], 1280)

    assert len(result.curves) == 1
    assert result.curves[0].carried_frames == 0
    assert len(detector._previous) == 1


def test_arrow_width_expansion_is_rejected() -> None:
    detector = _detector()
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[15:90, 96:104] = 255
    for y in range(45, 70):
        half_width = 5 + (y - 45) * 2
        mask[y, 100 - half_width : 100 + half_width] = 255
    probability = np.full(mask.shape, 0.9, dtype=np.float32)
    drivable = np.full(mask.shape, 0.9, dtype=np.float32)

    candidates, rejection_reasons, _ = detector._component_curves(
        mask,
        probability,
        drivable,
    )

    assert candidates == []
    assert rejection_reasons["arrow_like_width"] == 1


def test_aligned_short_fragments_can_merge_across_occlusion() -> None:
    detector = _detector()
    upper = _curve(500, 520, top=300, bottom=360)
    lower = _curve(540, 570, top=420, bottom=500)

    merged = detector._try_merge_fragments(upper, lower, 1280, 720)

    assert merged is not None
    assert merged.points[0].y == 300
    assert merged.points[-1].y == 500


def test_post_sigmoid_background_is_below_class_boundary() -> None:
    scores = np.asarray([[[[1.0, 0.4]], [[0.4, 1.0]]]], dtype=np.float32)
    result = YoloPLaneLineDetector._foreground_score(scores)
    assert result[0, 0] == pytest.approx(0.4 / 1.4)
    assert result[0, 1] == pytest.approx(1 / 1.4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2])
def test_invalid_model_scores_fail_explicitly(value: float) -> None:
    with pytest.raises(RuntimeError, match="post-Sigmoid"):
        YoloPLaneLineDetector._foreground_score(np.full((1, 2, 2, 2), value))


@pytest.mark.parametrize("mirror", [False, True])
def test_corner_box_preserves_road_but_masks_body_and_real_vehicle(mirror: bool) -> None:
    detector = _detector()
    frame = np.full((100, 200, 3), 60, np.uint8)
    frame[62:69, 20:90] = 200
    probability = np.full((100, 200), 0.9, np.float32)
    road = np.full_like(probability, 0.3)
    road[55:75, 20:90] = 0.8
    boxes = [BBox(x1=0, y1=50, x2=100, y2=100), BBox(x1=55, y1=55, x2=65, y2=75)]
    if mirror:
        frame = frame[:, ::-1].copy()
        road = road[:, ::-1].copy()
        boxes = [BBox(x1=199-b.x2, y1=b.y1, x2=199-b.x1, y2=b.y2) for b in boxes]
    mask, _ = detector._candidate_mask(frame, probability, boxes, road)
    road_x, body_x, car_x = (30, 30, 60) if not mirror else (169, 169, 139)
    assert mask[65, road_x] > 0
    assert mask[90, body_x] == 0
    assert mask[65, car_x] == 0
    assert detector.occlusion_mask is not None
    assert detector.occlusion_mask[90, body_x] > 0


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("thickness,value,released", [(5, 230, True), (50, 230, False),
                                                     (5, 80, False)])
def test_box_padding_requires_narrow_strong_paint_and_preserves_interior(
    mirror: bool, thickness: int, value: int, released: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    probability = np.zeros((720, 1280), np.float32)
    road = np.full_like(probability, 0.8)
    cv2.line(frame, (500, 400), (300, 460), (value,) * 3, thickness)
    cv2.line(probability, (500, 400), (300, 460), 0.9, thickness)
    box = BBox(x1=280, y1=380, x2=540, y2=434)
    if mirror:
        frame = frame[:, ::-1].copy()
        probability = probability[:, ::-1].copy()
        box = BBox(x1=1279-box.x2, y1=box.y1, x2=1279-box.x1, y2=box.y2)
    padding_x, core_x = (370, 420) if not mirror else (909, 859)

    mask, _ = detector._candidate_mask(frame, probability, [box], road)

    assert bool(mask[439, padding_x]) is released
    assert detector.occlusion_mask is not None
    assert bool(detector.occlusion_mask[439, padding_x]) is not released
    assert mask[424, core_x] == 0
    assert detector.occlusion_mask[424, core_x] == 255
    # Repeated frames, or a different previous frame, cannot change which paint
    # is released. Refinement must never read yesterday's vehicle mask.
    expected = mask.copy()
    detector.occlusion_mask[:] = 255
    repeated, _ = detector._candidate_mask(frame, probability, [box], road)
    np.testing.assert_array_equal(repeated, expected)


@pytest.mark.parametrize("mirror", [False, True])
def test_shallow_paint_overrides_road_edge_only_with_narrow_converging_ridge(
    mirror: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    curve = _curve(500, 300, top=400, bottom=460)
    frame = np.full((720, 1280, 3), 60, np.uint8)
    road = np.full((720, 1280), 0.8, np.float32)
    for y in range(720):
        road[y, :max(0, min(1280, round(500-(y-400)*200/60)))] = 0.1
    cv2.line(frame, (500, 400), (300, 460), (230,) * 3, 5)
    if mirror:
        frame = frame[:, ::-1].copy()
        road = road[:, ::-1].copy()
        curve = curve.model_copy(update={"points": tuple(
            Point2D(x=1279-p.x, y=p.y) for p in curve.points
        )})
    paint = np.ones(road.shape, np.uint8)
    bad = np.zeros_like(paint)
    assert detector._context_rejection(curve, paint, bad, road, frame) is None

    # A bright pavement step shares the slope, but has no dark flank on both sides.
    frame[road < 0.5] = 230
    assert detector._context_rejection(curve, paint, bad, road, frame) == "road_edge_context"


def test_padding_does_not_extend_an_already_visible_stripe() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    probability = np.zeros((720, 1280), np.float32)
    cv2.line(frame, (500, 400), (300, 460), (230,) * 3, 5)
    cv2.line(probability, (500, 400), (300, 460), 0.9, 5)

    mask, _ = detector._candidate_mask(
        frame, probability, [BBox(x1=280, y1=380, x2=540, y2=405)],
        np.full_like(probability, 0.8),
    )

    assert mask[409, 470] == 0
    assert detector.occlusion_mask is not None
    assert detector.occlusion_mask[409, 470] == 255
    assert mask[448, 340] == 255


def test_margin_rescue_preserves_full_shape_vehicle_contamination_context() -> None:
    cv2 = pytest.importorskip("cv2")
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (690, 480), (710, 620), (230,) * 3, -1)
    cv2.fillPoly(frame, [np.asarray([(640, 490), (700, 420), (760, 490)])], (230,) * 3)
    expanded = np.zeros((720, 1280), np.uint8)
    expanded[410:630, 630:770] = 255
    baseline = _detector()
    baseline.occlusion_mask = expanded.copy()
    expected = baseline._paint_context(frame)
    detector = _detector()
    detector._marking_occlusion_mask = expanded
    effective = np.zeros_like(expanded)
    detector.occlusion_mask = effective

    actual = detector._paint_context(frame)

    for old, new in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(old, new)
    assert detector.occlusion_mask is effective
    assert detector.occlusion_mask[550, 700] == 0


def test_full_arrow_context_rejects_a_narrow_model_shaft_but_keeps_stripe() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (400, 410), (412, 670), (230, 230, 230), -1)
    cv2.rectangle(frame, (690, 480), (710, 620), (230, 230, 230), -1)
    cv2.fillPoly(frame, [np.asarray([(640, 490), (700, 420), (760, 490)])], (230,)*3)
    paint, bad = detector._paint_context(frame)
    road = np.full((720, 1280), 0.8, np.float32)
    shaft = _curve(700, 700, top=480, bottom=610)
    stripe = _curve(406, 406, top=420, bottom=660)
    assert detector._context_rejection(shaft, paint, bad, road) == "arrow_paint_context"
    assert detector._context_rejection(stripe, paint, bad, road) is None


def test_road_edge_is_rejected_while_painted_line_inside_road_is_kept() -> None:
    detector = _detector()
    paint = np.ones((720, 1280), np.uint8)
    bad = np.zeros_like(paint)
    road = np.full(paint.shape, 0.8, np.float32)
    road[:, 900:] = 0.3
    assert detector._context_rejection(
        _curve(900, 900), paint, bad, road
    ) == "road_edge_context"
    assert detector._context_rejection(_curve(700, 700), paint, bad, road) is None


def test_short_fragments_reach_merger_before_output_length_gate() -> None:
    detector = _detector()
    mask = np.zeros((720, 1280), np.uint8)
    mask[420:443, 500:507] = 255
    mask[457:480, 500:507] = 255
    probability = np.full(mask.shape, 0.8, np.float32)
    candidates, _, _ = detector._component_curves(mask, probability, probability)
    assert len(candidates) == 2
    merged = detector._merge_fragments(candidates, 1280, 720)
    assert len(merged) == 1
    assert merged[0].component_height_ratio > detector._minimum_fragment_span


def test_smoothing_does_not_bend_new_endpoints_toward_old_range() -> None:
    detector = _detector()
    previous = _curve(500, 600, top=400, bottom=500)
    current = _curve(490, 690, top=390, bottom=590)
    smoothed = detector._smooth(previous, current, "lane-1", 2, 1280)
    assert smoothed.points[0] == current.points[0]
    assert smoothed.points[-1] == current.points[-1]


def test_detect_confirms_real_stripe_without_confirming_arrow_shaft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (400, 410), (412, 670), (230,)*3, -1)
    cv2.rectangle(frame, (690, 480), (710, 620), (230,)*3, -1)
    cv2.fillPoly(frame, [np.asarray([(640, 490), (700, 420), (760, 490)])], (230,)*3)
    probability = np.zeros((720, 1280), np.float32)
    probability[410:671, 400:413] = 0.8
    probability[480:621, 695:706] = 0.8
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    assert detector.detect(frame).status is LaneLineStatus.UNKNOWN
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert all(abs(p.x - 406) < 1 for p in result.curves[0].points)
    assert result.diagnostics.rejection_reasons["arrow_paint_context"] == 1


def test_single_paint_pixel_does_not_attempt_shape_fit() -> None:
    detector = _detector()
    frame = np.full((100, 200, 3), 60, np.uint8)
    frame[60, 100] = 230
    _, rejected = detector._paint_context(frame)
    assert not rejected.any()


@pytest.mark.parametrize("mirror", [False, True])
def test_distant_small_arrow_head_is_rejected_without_rejecting_a_lane(mirror: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.fillPoly(frame, [np.asarray([(700, 430), (691, 446), (709, 446)])], (230,) * 3)
    cv2.line(frame, (400, 410), (410, 610), (230,) * 3, 7)
    arrow = _curve(700, 700, top=435, bottom=443)
    stripe = _curve(401, 409, top=430, bottom=590)
    if mirror:
        frame = frame[:, ::-1].copy()
        arrow = _curve(579, 579, top=435, bottom=443)
        stripe = _curve(878, 870, top=430, bottom=590)
    paint, bad = detector._paint_context(frame)
    road = np.full(paint.shape, 0.8, np.float32)
    assert detector._context_rejection(arrow, paint, bad, road) == "arrow_paint_context"
    assert detector._context_rejection(stripe, paint, bad, road, frame) is None


def test_painted_boundary_survives_one_sided_drivable_prediction() -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[:, 695:706] = 230
    paint = np.ones((720, 1280), np.uint8)
    rejected = np.zeros_like(paint)
    road = np.full(paint.shape, 0.8, np.float32)
    road[:, 700:] = 0.3
    curve = _curve(700, 700)
    assert detector._context_rejection(curve, paint, rejected, road, frame) is None
    # A brightness step at a sidewalk is not a stripe with dark pavement on both sides.
    frame[:, 695:] = 230
    assert detector._context_rejection(curve, paint, rejected, road, frame) == "road_edge_context"


@pytest.mark.parametrize("mirror", [False, True])
def test_transverse_line_rejected_without_removing_outer_longitudinal_lane(mirror: bool) -> None:
    detector = _detector()
    paint = np.ones((720, 1280), np.uint8)
    bad = np.zeros_like(paint)
    road = np.full(paint.shape, 0.8, np.float32)
    # Same shallow slope magnitude, different convergence into the scene.
    stop = _curve(350, 950, top=450, bottom=520)
    lane = _curve(300, 100, top=420, bottom=460)
    if mirror:
        stop = stop.model_copy(update={"points": tuple(
            Point2D(x=1280-p.x, y=p.y) for p in stop.points
        )})
        lane = lane.model_copy(update={"points": tuple(
            Point2D(x=1280-p.x, y=p.y) for p in lane.points
        )})
    assert detector._context_rejection(stop, paint, bad, road) == "transverse_marking"
    assert detector._context_rejection(lane, paint, bad, road) is None
    assert detector._context_rejection(_curve(700, 700), paint, bad, road) is None


def test_weak_bright_curb_does_not_override_road_boundary() -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), 70, np.uint8)
    frame[:, 695:706] = 98
    paint = np.ones((720, 1280), np.uint8)
    road = np.full(paint.shape, 0.8, np.float32)
    road[:, 700:] = 0.3
    assert detector._context_rejection(
        _curve(700, 700), paint, np.zeros_like(paint), road, frame
    ) == "road_edge_context"
    frame[:, 695:706] = 180
    assert detector._context_rejection(
        _curve(700, 700), paint, np.zeros_like(paint), road, frame
    ) is None


def test_box_clipping_does_not_turn_stripe_into_arrow_context() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (400, 410), (412, 670), (230,)*3, -1)
    detector.occlusion_mask = np.full((720, 1280), 255, np.uint8)
    detector.occlusion_mask[500:530, 395:420] = 0
    paint, bad = detector._paint_context(frame)
    assert paint[515, 406] > 0
    assert bad[515, 406] == 0


def test_partly_masked_arrow_head_still_rejects_visible_shaft() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (690, 480), (710, 620), (230,)*3, -1)
    cv2.fillPoly(frame, [np.asarray([(640, 490), (700, 420), (760, 490)])], (230,)*3)
    detector.occlusion_mask = np.zeros((720, 1280), np.uint8)
    detector.occlusion_mask[:455] = 255
    paint, bad = detector._paint_context(frame)
    assert detector._context_rejection(
        _curve(700, 700, top=500, bottom=610), paint, bad,
        np.full(paint.shape, 0.8, np.float32),
    ) == "arrow_paint_context"


def test_short_visible_stripe_confirms_but_tiny_chip_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[450:545, 697:704] = 230
    frame[500:517, 495:506] = 230
    probability = np.zeros((720, 1280), np.float32)
    probability[450:545, 697:704] = 0.8
    probability[500:517, 495:506] = 0.8
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    boxes = [BBox(x1=690, y1=450, x2=710, y2=495)]
    detector.detect(frame, boxes)
    result = detector.detect(frame, boxes)
    assert len(result.curves) == 1
    assert result.curves[0].points[0].x == pytest.approx(700)
    assert result.diagnostics.rejection_reasons["unmerged_short_fragment"] == 1


@pytest.mark.parametrize("span", [36, 44])
def test_candidate_span_does_not_replace_short_stripe_evidence(
    monkeypatch: pytest.MonkeyPatch, span: int,
) -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[450:451 + span, 686:715] = 230
    frame[450:545, 897:904] = 230
    probability = np.zeros((720, 1280), np.float32)
    probability[450:451 + span, 686:715] = 0.8
    probability[450:545, 897:904] = 0.8
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    candidates = [
        _curve(700, 700, top=450, bottom=450 + span),
        _curve(900, 900, top=450, bottom=544),
    ]
    monkeypatch.setattr(detector, "_component_curves", lambda *a, **k: (candidates, Counter(), 2))
    monkeypatch.setattr(detector, "_context_rejection", lambda *a, **k: None)
    monkeypatch.setattr(detector, "_refine_paint_geometry", lambda curve, _: curve)
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert result.curves[0].points[0].x == pytest.approx(900)
    assert result.diagnostics.rejection_reasons["unmerged_short_fragment"] == 1


@pytest.mark.parametrize("color", [(30, 130, 180), (50, 65, 170)])
def test_colored_curb_rejects_bright_lip_but_not_interior_lane(color: tuple[int, ...]) -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[:, 695:706] = 200
    frame[:, 680:693] = color
    paint = np.ones((720, 1280), np.uint8)
    road = np.full(paint.shape, 0.8, np.float32)
    bad = np.zeros_like(paint)
    assert detector._context_rejection(_curve(700, 700), paint, bad, road, frame) is None
    road[:, :700] = 0.3
    assert detector._context_rejection(
        _curve(700, 700), paint, bad, road, frame
    ) == "colored_curb_context"


def test_arrow_shoulder_is_rejected_but_smooth_perspective_taper_is_kept() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    # A thick shaft makes the overall width percentile ratio a weak cue.
    cv2.rectangle(frame, (690, 510), (710, 650), (230,)*3, -1)
    cv2.fillPoly(frame, [np.asarray([(675, 515), (700, 430), (725, 515)])], (230,)*3)
    cv2.fillPoly(frame, [np.asarray([(400, 430), (410, 430), (418, 650), (392, 650)])],
                 (230,)*3)
    paint, bad = detector._paint_context(frame)
    road = np.full(paint.shape, 0.8, np.float32)
    assert detector._context_rejection(
        _curve(700, 700, top=520, bottom=640), paint, bad, road
    ) == "arrow_paint_context"
    assert detector._context_rejection(
        _curve(405, 405, top=440, bottom=640), paint, bad, road
    ) is None


def test_isolated_short_stroke_is_not_rescued_without_occlusion() -> None:
    detector = _detector()
    detector.occlusion_mask = np.zeros((720, 1280), np.uint8)
    short = _curve(830, 880, top=440, bottom=470)
    assert not detector._occluded_fragment(short, 1280)
    detector.occlusion_mask[425:440, 825:835] = 255
    assert detector._occluded_fragment(short, 1280)
    assert not detector._occluded_fragment(_curve(830, 835, top=440, bottom=450), 1280)


def test_car_mask_proximity_requires_a_visible_narrow_stripe() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    detector.occlusion_mask = np.zeros((720, 1280), np.uint8)
    detector.occlusion_mask[425:440, 825:835] = 255
    short = _curve(830, 880, top=440, bottom=470)
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (790, 420), (925, 490), (230,) * 3, -1)
    assert not detector._occluded_fragment(short, 1280, frame)
    frame[:] = 60
    cv2.line(frame, (830, 440), (880, 470), (230,) * 3, 5)
    assert detector._occluded_fragment(short, 1280, frame)


def test_contrast_filter_preserves_the_original_painted_component_shape() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[400:600, 300:500] = 180
    frame[450:550, 375:385] = 255
    probability = np.zeros((720, 1280), np.float32)
    probability[400:600, 300:500] = 0.9
    mask, _ = detector._candidate_mask(frame, probability, [])
    count, _, _, _ = cv2.connectedComponentsWithStats(detector._parent_lane_mask, 8)
    # Splitting the bright component leaves detached, apparently narrow pieces
    # and loses the original shape evidence used to reject non-lane markings.
    assert count - 1 == 1
    assert detector._parent_lane_mask[500, 350] > 0
    assert mask[500, 350] == 0


def test_visible_paint_gap_can_ignore_a_short_fragment_tangent_kink() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    upper = _curve(700, 700, top=440, bottom=520)
    lower = _curve(700, 700, top=590, bottom=616)
    t = np.linspace(0, 1, 24)
    lower = lower.model_copy(update={"points": tuple(
        Point2D(x=float(700 + 32 * v * (1 - v)), y=float(590 + 26 * v)) for v in t
    )})
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(frame, (700, 440), (700, 616), (230,) * 3, 9)
    assert detector._try_merge_fragments(upper, lower, 1280, 720) is None
    assert detector._try_merge_fragments(upper, lower, 1280, 720, frame) is not None
    frame[521:590] = 60
    assert detector._try_merge_fragments(upper, lower, 1280, 720, frame) is None
    cv2.line(frame, (700, 440), (700, 616), (230,) * 3, 9)
    detector.occlusion_mask = np.zeros((720, 1280), np.uint8)
    detector.occlusion_mask[540:560, 690:710] = 255
    assert detector._try_merge_fragments(upper, lower, 1280, 720, frame) is None
    detector.occlusion_mask[:] = 0
    frame[:] = 60
    cv2.line(frame, (712, 440), (712, 616), (230,) * 3, 9)
    assert detector._try_merge_fragments(upper, lower, 1280, 720, frame) is None


def test_detached_arrow_triangle_is_not_a_short_lane() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.fillPoly(frame, [np.asarray([(700, 430), (680, 600), (720, 600)])], (230,)*3)
    paint, bad = detector._paint_context(frame)
    assert detector._context_rejection(
        _curve(700, 700, top=480, bottom=590), paint, bad,
        np.full(paint.shape, 0.8, np.float32),
    ) == "arrow_paint_context"


def test_parallel_but_offset_fragments_do_not_merge() -> None:
    detector = _detector()
    upper = _curve(611, 678, top=404, bottom=421).model_copy(update={
        "points": (Point2D(x=611, y=404), Point2D(x=652, y=412.5), Point2D(x=678, y=421)),
    })
    lower = _curve(858, 1042, top=457, bottom=545).model_copy(update={
        "points": (Point2D(x=858, y=457), Point2D(x=972, y=501), Point2D(x=1042, y=545)),
    })
    assert detector._try_merge_fragments(upper, lower, 1280, 720) is None


def test_crossing_curves_cannot_inherit_confirmation() -> None:
    detector = _detector()
    original = _curve(800, 900, top=400, bottom=500)
    detector._stabilize([original], 1280)
    before = detector._stabilize([original], 1280).curves[0]
    crossing = _curve(900, 800, top=400, bottom=500)
    result = detector._associate([crossing], 1280)[0]
    assert result.boundary_id != before.boundary_id
    assert result.confirmed_frames == 1


def test_normal_extension_and_occlusion_keep_identity() -> None:
    detector = _detector()
    first = detector._associate([_curve(700, 800, top=400, bottom=500)], 1280)[0]
    extended = detector._associate([_curve(702, 1002, top=400, bottom=700)], 1280)[0]
    shortened = detector._associate([_curve(852, 972, top=550, bottom=670)], 1280)[0]
    assert first.boundary_id == extended.boundary_id == shortened.boundary_id


def test_small_local_smoothing_steps_do_not_reset_a_real_stripe() -> None:
    import json
    from pathlib import Path

    fixture = json.loads((
        Path(__file__).resolve().parents[1] / "fixtures/lane_association_jitter.json"
    ).read_text())
    detector = _detector()
    previous = detector._associate([LaneCurve(**fixture["previous"])], fixture["width"])[0]

    current = detector._associate([LaneCurve(**fixture["current"])], fixture["width"])[0]

    assert current.boundary_id == previous.boundary_id
    assert current.confirmed_frames == 2


def test_smoothing_does_not_leave_current_thin_stripe() -> None:
    detector = _detector()
    previous = _curve(500, 600)
    current = _curve(540, 640)
    smoothed = detector._smooth(previous, current, "lane-1", 3, 1280)
    assert max(abs(p.x-q.x) for p, q in zip(smoothed.points, current.points, strict=True)) <= 8


def test_short_shallow_glare_is_rejected_but_paint_is_kept() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 160, np.uint8)
    curve = _curve(880, 1000, top=450, bottom=510)
    road = np.full((720, 1280), 0.9, np.float32)
    paint = np.ones((720, 1280), np.uint8)
    rejected = np.zeros_like(paint)
    assert detector._context_rejection(curve, paint, rejected, road, frame) is not None
    cv2.line(frame, (880, 450), (1000, 510), (255, 255, 255), 6)
    assert detector._context_rejection(curve, paint, rejected, road, frame) is None


def test_visible_painted_tail_survives_low_road_candidate_prefix() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 100, np.uint8)
    probability = np.zeros((720, 1280), np.float32)
    cv2.line(probability, (438, 483), (333, 551), 0.9, 8)
    cv2.line(frame, (377, 521), (333, 551), (240, 240, 240), 8)
    road = np.full_like(probability, 0.3)
    detector._segmentation_probabilities = lambda _: (probability, road)  # type: ignore[method-assign]
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert result.curves[0].points[0].y >= 510
    assert result.curves[0].points[-1].y >= 550


def test_dim_shallow_stripe_needs_symbol_evidence_before_rejection() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 100, np.uint8)
    curve = _curve(355, 147, top=394, bottom=449)
    # A visible dash supports only part of the fitted longitudinal fragment.
    cv2.line(frame, (355, 394), (250, 422), (230,)*3, 6)
    paint = np.ones((720, 1280), np.uint8)
    symbols = np.zeros_like(paint)
    road = np.full(paint.shape, 0.9, np.float32)
    assert detector._context_rejection(curve, paint, symbols, road, frame) is None
    # The same weak fragment beside an arrow mask is not reliable lane evidence.
    cv2.line(symbols, (355, 380), (147, 435), 1, 3)
    assert detector._context_rejection(
        curve, paint, symbols, road, frame
    ) == "weak_transverse_paint"


@pytest.mark.parametrize("mirror", [False, True])
def test_crosswalk_group_rejects_thin_model_edges_but_keeps_nearby_lane(
    monkeypatch: pytest.MonkeyPatch, mirror: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    probability = np.zeros((720, 1280), np.float32)
    for x in (300, 460, 620):
        cv2.rectangle(frame, (x, 450), (x + 30, 650), (230,) * 3, -1)
        probability[450:631, x + 5:x + 11] = 0.8
    cv2.line(frame, (900, 400), (950, 650), (230,) * 3, 9)
    cv2.line(probability, (900, 400), (950, 650), 0.8, 9)
    if mirror:
        frame, probability = frame[:, ::-1].copy(), probability[:, ::-1].copy()
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert result.diagnostics.rejection_reasons["crosswalk_context"] == 3


def test_parking_corner_rejects_disconnected_grid_side_without_removing_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.polylines(frame, [np.array([(650, 440), (650, 620), (820, 620)])],
                  False, (230,) * 3, 9)
    cv2.line(frame, (820, 470), (820, 603), (230,) * 3, 9)
    cv2.line(frame, (450, 420), (450, 650), (230,) * 3, 9)
    probability = np.zeros(frame.shape[:2], np.float32)
    probability[470:600, 817:824] = 0.8
    probability[420:651, 447:454] = 0.8
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert result.curves[0].points[0].x == pytest.approx(450)
    assert result.diagnostics.rejection_reasons["parking_marking_context"] == 1


def test_recognised_marking_cannot_survive_as_a_carried_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    frame[450:651, 695:706] = 230
    probability = np.zeros(frame.shape[:2], np.float32)
    probability[450:651, 695:706] = 0.8
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    assert len(detector.detect(frame).curves) == 1
    for x in (300, 500, 690):
        cv2.rectangle(frame, (x, 450), (x + 45, 650), (230,) * 3, -1)
    probability[:] = 0
    assert not detector.detect(frame).curves


@pytest.mark.parametrize("mirror", [False, True])
def test_shallow_painted_lane_survives_missing_semantic_road(
    monkeypatch: pytest.MonkeyPatch, mirror: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    probability = np.zeros(frame.shape[:2], np.float32)
    cv2.line(frame, (200, 450), (420, 400), (200,) * 3, 9)
    cv2.line(probability, (200, 450), (420, 400), 0.8, 9)
    if mirror:
        frame, probability = frame[:, ::-1].copy(), probability[:, ::-1].copy()
    road = np.full(probability.shape, 0.2, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1


@pytest.mark.parametrize("mirror", [False, True])
def test_dark_curve_beside_narrow_yellow_curb_keeps_color_rejection(mirror: bool) -> None:
    detector = _detector()
    frame = np.full((720, 1280, 3), (75, 60, 50), np.uint8)
    frame[:, 704:711] = (30, 130, 180)
    curve = _curve(700, 700, top=430, bottom=640)
    if mirror:
        frame = frame[:, ::-1].copy()
        curve = _curve(579, 579, top=430, bottom=640)
    paint = np.ones(frame.shape[:2], np.uint8)
    road = np.full(paint.shape, 0.2, np.float32)
    assert detector._context_rejection(
        curve, paint, np.zeros_like(paint), road, frame,
    ) == "colored_curb_context"


def test_bent_model_halo_is_aligned_to_straight_visible_paint() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(frame, (600, 410), (600, 650), (220,) * 3, 9)
    ys = np.linspace(410, 650, 24)
    candidate = _curve(614, 614, top=410, bottom=650).model_copy(update={
        "points": tuple(Point2D(x=600 + 14 * ((y - 530) / 120) ** 2, y=y) for y in ys),
    })

    refined = detector._refine_paint_geometry(candidate, frame)

    assert all(abs(point.x - 600) <= 2 for point in refined.points)
    assert refined.points[-1].y - refined.points[0].y >= 200


def test_geometry_refinement_preserves_genuinely_curved_paint() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    ys = np.linspace(410, 650, 24)
    points = tuple(Point2D(x=600 + 65 * ((y - 530) / 120) ** 2, y=y) for y in ys)
    cv2.polylines(frame, [np.rint([(p.x, p.y) for p in points]).astype(np.int32)],
                  False, (220,) * 3, 9)
    candidate = _curve(665, 665, top=410, bottom=650).model_copy(update={"points": points})

    assert detector._refine_paint_geometry(candidate, frame) is candidate


def test_geometry_refinement_does_not_create_paint_on_uniform_pavement() -> None:
    detector = _detector()
    candidate = _curve(600, 620).model_copy(update={
        "points": tuple(Point2D(x=600 + index, y=410 + index * 10) for index in range(24)),
    })

    assert detector._refine_paint_geometry(
        candidate, np.full((720, 1280, 3), 100, np.uint8),
    ) is candidate


def test_geometry_refinement_preserves_widening_nearby_paint() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.fillPoly(frame, [np.asarray([(795, 430), (805, 430), (821, 710), (779, 710)])],
                 (160,) * 3)
    candidate = _curve(800, 800, top=430, bottom=710).model_copy(update={
        "points": tuple(Point2D(x=800, y=y) for y in np.linspace(430, 710, 24)),
    })

    refined = detector._refine_paint_geometry(candidate, frame)

    assert refined.points[-1].y >= 700
    assert max(abs(point.x - 800) for point in refined.points) <= 2


def test_painted_candidate_wins_over_more_confident_asphalt_duplicate() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(frame, (400, 410), (400, 650), (220,) * 3, 9)
    painted = _curve(400, 400, confidence=0.6, top=410, bottom=650)
    halo = _curve(414, 414, confidence=0.9, top=410, bottom=650)

    assert detector._deduplicate([halo, painted], 1280, frame) == [painted]


def test_bright_uniform_pavement_cannot_connect_a_model_halo_to_paint() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 110, np.uint8)
    cv2.line(frame, (600, 410), (600, 650), (230,) * 3, 9)
    probability = np.zeros(frame.shape[:2], np.float32)
    probability[410:651, 580:641] = 0.8

    mask, _ = detector._candidate_mask(frame, probability, [], probability)
    assert mask[500, 600] > 0
    assert mask[500, 620] == 0
    detector._segmentation_probabilities = lambda _: (
        probability, np.full(probability.shape, 0.8, np.float32),
    )
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    assert all(abs(p.x - 600) <= 3 for p in result.curves[0].points)
    assert all(410 <= p.y <= 650 for p in result.curves[0].points)
    detector.reset()
    frame[:] = 110
    detector.detect(frame)
    assert not detector.detect(frame).curves


def test_strong_shallow_stripe_survives_one_sided_road_context() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(frame, (360, 440), (480, 406), (220,) * 3, 7)
    paint = np.zeros((720, 1280), np.uint8)
    cv2.line(paint, (360, 440), (480, 406), 1, 7)
    road = np.full(paint.shape, 0.2, np.float32)
    road[445:] = 0.8
    stripe = _curve(480, 360, top=406, bottom=440)

    assert detector._context_rejection(stripe, paint, np.zeros_like(paint), road, frame) is None


def test_complex_parking_corner_is_not_limited_to_twelve_vertices() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(frame, (700, 430), (810, 650), (230,) * 3, 5)
    cv2.line(frame, (810, 650), (1040, 650), (230,) * 3, 5)
    for y in range(470, 630, 30):
        x = 700 + (y - 430) // 2
        cv2.line(frame, (x, y), (x + 95, y + 8), (230,) * 3, 5)

    detector._paint_context(frame)

    assert detector._marking_masks["parking_marking_context"][550, 760] > 0


def test_long_slanted_crosswalk_bars_keep_the_existing_group_rule() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    bar = np.asarray([(400, 450), (570, 485), (568, 498), (398, 463)])
    for offset in (0, 220, 440):
        cv2.fillPoly(frame, [bar + (offset, 0)], (230,) * 3)

    detector._paint_context(frame)

    assert detector._marking_masks["crosswalk_context"][477, 700] > 0


@pytest.mark.parametrize("mirror", [False, True])
def test_distant_connected_parallel_grid_is_distinct_from_one_lane(mirror: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    grid = np.zeros((720, 1280), np.uint8)
    for shift in (0, 25, 50):
        cv2.line(grid, (800 + shift, 380 - shift // 5),
                 (865 + shift, 410 - shift // 5), 1, 5)
    cv2.line(grid, (865, 410), (915, 400), 1, 5)
    lane = np.zeros_like(grid)
    cv2.line(lane, (800, 380), (865, 410), 1, 5)
    if mirror:
        grid, lane = grid[:, ::-1].copy(), lane[:, ::-1].copy()

    assert np.count_nonzero(detector._parallel_markings(grid)) > 0
    assert np.count_nonzero(detector._parallel_markings(lane)) == 0


def test_single_broad_stripe_and_parallel_thin_lanes_are_not_a_crosswalk() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 60, np.uint8)
    for x in (300, 460):
        cv2.rectangle(frame, (x, 450), (x + 9, 650), (230,) * 3, -1)
    cv2.rectangle(frame, (620, 450), (665, 650), (230,) * 3, -1)
    detector._paint_context(frame)
    assert not detector._marking_masks["crosswalk_context"].any()


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("lightness", [65, 230])
def test_labelled_bay_excludes_long_side_but_preserves_adjacent_lane(
    monkeypatch: pytest.MonkeyPatch, mirror: bool, lightness: int,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    cv2.line(frame, (450, 420), (450, 650), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 500), (x + 40, 530), (lightness,) * 3, 5)
    probability = np.zeros(frame.shape[:2], np.float32)
    probability[400:681, 716:725] = 0.8
    probability[420:651, 446:455] = 0.8
    if mirror:
        frame = frame[:, ::-1].copy()
        probability = probability[:, ::-1].copy()
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    expected_x = 1279 - 450 if mirror else 450
    assert result.curves[0].points[0].x == pytest.approx(expected_x, abs=2)
    assert result.diagnostics.rejection_reasons["labelled_bay_context"] == 1


@pytest.mark.parametrize("kind", ["single_symbol", "crosswalk", "vehicle", "off_road"])
def test_unreliable_paint_cannot_supply_labelled_bay_evidence(kind: str) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    count = 1 if kind == "single_symbol" else 3
    for x in (790, 845, 900)[:count]:
        if kind == "crosswalk":
            cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, -1)
        else:
            cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, 5)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    if kind == "vehicle":
        detector._marking_occlusion_mask = np.zeros(road.shape, np.uint8)
        detector._marking_occlusion_mask[490:540, 780:950] = 1
    if kind == "off_road":
        road[490:540, 780:950] = 0.1
    assert not np.any(detector._labelled_bay_markings(frame, road))


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("grid_color", [(230, 230, 230), (130, 150, 155)])
def test_sparse_junction_grid_cannot_remove_an_adjacent_true_lane(
    monkeypatch: pytest.MonkeyPatch, mirror: bool, grid_color: tuple[int, int, int],
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    for x in (790, 860, 930):
        cv2.line(frame, (x, 470), (x, 590), grid_color, 5)
    for y in (470, 510, 550, 590):
        cv2.line(frame, (790, y), (930, y), grid_color, 5)
    probability = np.zeros(frame.shape[:2], np.float32)
    probability[400:681, 716:725] = 0.8
    if mirror:
        frame = frame[:, ::-1].copy()
        probability = probability[:, ::-1].copy()
    road = np.full(probability.shape, 0.8, np.float32)
    monkeypatch.setattr(detector, "_segmentation_probabilities", lambda _: (probability, road))
    detector.detect(frame)
    result = detector.detect(frame)
    assert len(result.curves) == 1
    expected_x = 1279 - 720 if mirror else 720
    assert result.curves[0].points[0].x == pytest.approx(expected_x, abs=2)
    assert not result.diagnostics.rejection_reasons.get("labelled_bay_context")


def test_bay_mask_follows_verified_road_motion_and_rejects_scene_cut() -> None:
    cv2 = pytest.importorskip("cv2")
    random = np.random.default_rng(12)
    gray = random.integers(25, 75, (360, 640), dtype=np.uint8)
    features = np.ones(gray.shape, np.uint8)
    mask = np.zeros_like(gray)
    mask[200:300, 400:408] = 1
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 4], [0, 1, 3]]), (640, 360))
    result = YoloPLaneLineDetector._transport_bay_mask(gray, moved, features, features, mask)
    assert result[253, 407] == 1
    assert result[250, 400] == 0
    unrelated = random.integers(25, 75, gray.shape, dtype=np.uint8)
    assert not np.any(YoloPLaneLineDetector._transport_bay_mask(
        gray, unrelated, features, features, mask,
    ))


def test_labelled_bay_history_requires_current_visible_paint_and_expires() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    random = np.random.default_rng(8)
    base = random.integers(20, 35, (720, 1280), dtype=np.uint8)
    frame = np.repeat(base[:, :, None], 3, axis=2)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, 5)
    road = np.full(base.shape, 0.8, np.float32)
    assert detector._labelled_bay_markings(frame, road)[600, 720] > 0
    without_label = frame.copy()
    without_label[490:540, 780:950] = np.repeat(base[490:540, 780:950, None], 3, axis=2)
    moved = cv2.warpAffine(without_label, np.float32([[1, 0, 4], [0, 1, 3]]), (1280, 720))
    assert detector._labelled_bay_markings(moved, road)[603, 724] > 0
    assert detector._bay_history is not None
    gray, features, mask, _ = detector._bay_history
    detector._bay_history = (gray, features, mask, 45)
    assert not np.any(detector._labelled_bay_markings(moved, road))
    detector._labelled_bay_markings(frame, road)
    detector._marking_occlusion_mask = np.ones(base.shape, np.uint8)
    assert not np.any(detector._labelled_bay_markings(without_label, road))
    assert detector._bay_history is None
    detector._marking_occlusion_mask = None
    detector._labelled_bay_markings(frame, road)
    detector.reset()
    assert detector._bay_history is None


def test_skewed_label_outline_is_not_its_own_bay_border() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    paint = np.zeros((720, 1280), np.uint8)
    for x in (600, 660, 720):
        cv2.rectangle(paint, (x, 400), (x + 45, 440), 1, 6)
    transform = np.asarray([[1, 0, 170], [0.45, 1, -180]], np.float32)
    paint = cv2.warpAffine(paint, transform, (1280, 720))
    frame = np.full((720, 1280, 3), 25, np.uint8)
    frame[paint > 0] = 230
    road = np.full(paint.shape, 0.8, np.float32)
    assert not np.any(detector._labelled_bay_markings(frame, road))


@pytest.mark.parametrize("mode", ["nearby", "jump", "scene_cut", "no_paint"])
def test_observed_bay_stripe_following_preserves_parallel_lane_and_stops(mode: str) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    base = np.tile(np.linspace(35, 145, 720, dtype=np.uint8)[:, None], (1, 1280))
    frame = np.repeat(base[:, :, None], 3, axis=2)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    cv2.line(frame, (680, 400), (680, 680), (230,) * 3, 9)
    previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_mask = np.zeros_like(base)
    old_mask[400:681, 714:727] = 1
    shift = 50 if mode == "jump" else 4
    current = cv2.warpAffine(frame, np.asarray([[1, 0, shift], [0, 1, 3]], np.float32),
                            (1280, 720))
    if mode == "scene_cut":
        current = np.repeat(base[::-1, :, None], 3, axis=2)
        cv2.line(current, (724, 403), (724, 683), (230,) * 3, 9)
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    paint = (gray > 200).astype(np.uint8)
    if mode == "no_paint":
        paint[:] = 0
    result = detector._follow_bay_paint(previous_gray, gray, old_mask, paint, current)
    if mode == "nearby":
        assert result[600, 724] > 0
        assert result[600, 684] == 0
    else:
        assert not np.any(result)


@pytest.mark.parametrize("mirror", [False, True])
def test_distant_label_cannot_reject_an_unrelated_lane_end(mirror: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    paint = np.zeros((720, 1280), np.uint8)
    cv2.line(paint, (720, 330), (720, 380), 1, 7)
    for x in (790, 845, 900):
        cv2.rectangle(paint, (x, 500), (x + 40, 530), 1, 5)
    if mirror:
        paint = paint[:, ::-1].copy()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    frame[paint > 0] = 230
    result = detector._labelled_bay_sides(paint, frame)
    assert not np.any(result)


@pytest.mark.parametrize("mirror", [False, True])
def test_label_ahead_of_a_long_lane_dash_cannot_supply_bay_context(mirror: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 580), (720, 710), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 480), (x + 40, 510), (230,) * 3, 5)
    if mirror:
        frame = frame[:, ::-1].copy()
    road = np.full(frame.shape[:2], 0.8, np.float32)
    assert not np.any(detector._labelled_bay_markings(frame, road))


def test_connected_manhole_does_not_turn_a_bay_border_into_a_label() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    paint = np.zeros((720, 1280), np.uint8)
    cv2.line(paint, (720, 400), (720, 680), 1, 7)
    cv2.circle(paint, (720, 600), 30, 1, 6)
    for x in (790, 845, 900):
        cv2.rectangle(paint, (x, 500), (x + 40, 530), 1, 5)
    frame = np.full((720, 1280, 3), 25, np.uint8)
    frame[paint > 0] = 230
    result = detector._labelled_bay_sides(paint, frame)
    assert result[450, 720] > 0


@pytest.mark.parametrize("mirror", [False, True])
def test_nearby_wide_bay_paint_preserves_the_adjacent_lane(mirror: bool) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 25)
    cv2.line(frame, (450, 420), (450, 650), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, 5)
    if mirror:
        frame = frame[:, ::-1].copy()
    road = np.full(frame.shape[:2], 0.8, np.float32)
    result = detector._labelled_bay_markings(frame, road)
    bay_x, lane_x = (559, 829) if mirror else (720, 450)
    assert result[600, bay_x] > 0
    assert result[600, lane_x] == 0


def test_unrelated_fresh_label_cannot_replace_or_renew_an_observed_bay() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    random = np.random.default_rng(8)
    base = random.integers(20, 35, (720, 1280), dtype=np.uint8)
    frame = np.repeat(base[:, :, None], 3, axis=2)
    cv2.line(frame, (720, 400), (720, 680), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, 5)
    road = np.full(base.shape, 0.8, np.float32)
    detector._labelled_bay_markings(frame, road)
    assert detector._bay_history is not None
    gray, features, mask, _ = detector._bay_history
    detector._bay_history = (gray, features, mask, 10)
    current = frame.copy()
    current[490:540, 780:950] = np.repeat(base[490:540, 780:950, None], 3, axis=2)
    cv2.line(current, (300, 400), (300, 680), (230,) * 3, 9)
    for x in (370, 425, 480):
        cv2.rectangle(current, (x, 500), (x + 40, 530), (230,) * 3, 5)
    result = detector._labelled_bay_markings(current, road)
    assert result[600, 720] > 0
    assert result[600, 300] > 0
    assert detector._bay_history is not None
    assert detector._bay_history[3] == 11
    assert detector._bay_history[2][600, 300] == 0


def test_camera_footer_text_cannot_supply_a_road_label() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 450), (720, 710), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 688), (x + 40, 711), (230,) * 3, 5)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    assert not np.any(detector._labelled_bay_markings(frame, road))


@pytest.mark.parametrize("lightness", [65, 230])
def test_unlabelled_bay_elbow_excludes_side_and_keeps_parallel_lane(lightness: int) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (820, 620), (lightness,) * 3, 7)
    cv2.line(frame, (820, 620), (1080, 620), (lightness,) * 3, 7)
    cv2.line(frame, (450, 420), (450, 650), (230,) * 3, 9)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    result = detector._labelled_bay_markings(frame, road)
    assert result[550, 788] > 0
    assert result[600, 450] == 0


def test_stop_line_crossbar_does_not_turn_its_lane_into_a_bay() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 400), (720, 620), (230,) * 3, 7)
    cv2.line(frame, (400, 620), (1080, 620), (230,) * 3, 7)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    assert not np.any(detector._labelled_bay_markings(frame, road))


def test_label_near_a_short_dash_cannot_seed_a_parking_border() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (720, 450), (720, 540), (230,) * 3, 9)
    for x in (790, 845, 900):
        cv2.rectangle(frame, (x, 500), (x + 40, 530), (230,) * 3, 5)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    assert not np.any(detector._labelled_bay_markings(frame, road))


def test_bay_elbow_recovers_only_collinear_observed_paint_across_a_worn_gap() -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    frame = np.full((720, 1280, 3), 25, np.uint8)
    cv2.line(frame, (630, 297), (650, 331), (230,) * 3, 7)
    cv2.line(frame, (720, 450), (820, 620), (230,) * 3, 7)
    cv2.line(frame, (820, 620), (1080, 620), (230,) * 3, 7)
    cv2.line(frame, (450, 420), (450, 650), (230,) * 3, 9)
    road = np.full(frame.shape[:2], 0.8, np.float32)
    result = detector._labelled_bay_markings(frame, road)
    assert result[320, 644] > 0
    assert result[600, 450] == 0
    assert result[400, 691] == 0  # Missing paint is not reconstructed.


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((662, 376), (687, 398), True),
        ((625, 408), (630, 440), False),
        ((630, 405), (655, 427), False),
        ((510, 243), (535, 265), False),
    ],
)
def test_verified_bay_extension_preserves_other_stripes_and_limits_the_gap(
    start: tuple[int, int], end: tuple[int, int], expected: bool,
) -> None:
    cv2 = pytest.importorskip("cv2")
    detector = _detector()
    marking = np.zeros((720, 1280), np.uint8)
    cv2.line(marking, (740, 447), (1010, 682), 1, 15)
    curve = _curve(start[0], end[0], top=start[1], bottom=end[1])
    assert detector._near_bay_extension(curve, marking) is expected

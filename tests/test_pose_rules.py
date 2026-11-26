from __future__ import annotations

import pytest

from body.anthropometry import (
    AnthropometricExtractor,
    ScaleCalibrator,
    TapeValidationSample,
    compute_cm_per_pixel,
    compute_cm_per_pixel_from_landmarks,
    estimate_body_height_details,
    estimate_full_body_height_px,
    evaluate_against_tape_measurements,
    validate_user_anthropometrics,
)
from body.landmarks import LandmarkPoint, PoseLandmarkIndex
from body.pipeline import BodyAnalysisPipeline
from body.pose_rules import PoseRuleEngine, RuleConfig, detect_fist_or_hand_trigger
from schemas.body import Body


class _DummyMask:
    def __init__(self, height: int, width: int, y1: int, y2: int, x1: int, x2: int) -> None:
        self.shape = (height, width)
        self._y1 = y1
        self._y2 = y2
        self._x1 = x1
        self._x2 = x2

    def __getitem__(self, item: tuple[int, slice]) -> list[int]:
        row_y, col_slice = item
        start = col_slice.start or 0
        stop = col_slice.stop or self.shape[1]
        if not (self._y1 <= row_y < self._y2):
            return [0] * (stop - start)
        return [
            255 if (self._x1 <= x < self._x2) else 0
            for x in range(start, stop)
        ]


class _SplitLegMask:
    """Simulates a silhouette with an empty gap between the legs along the body midline."""

    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width)

    def __getitem__(self, item: tuple[int, slice]) -> list[int]:
        row_y, col_slice = item
        start = col_slice.start or 0
        stop = col_slice.stop or self.shape[1]
        # Head + torso solid from y=42..390 (x=540..740)
        if 42 <= row_y < 390:
            return [255 if (540 <= x < 740) else 0 for x in range(start, stop)]
        # Two separated legs from y=390..655 (left x=555..595, right x=685..725; gap at x=640!)
        if 390 <= row_y <= 655:
            return [
                255 if ((555 <= x < 595) or (685 <= x < 725)) else 0
                for x in range(start, stop)
            ]
        return [0] * (stop - start)


def _sync_pixels(
    landmarks: dict[int, LandmarkPoint],
    width: int = 1280,
    height: int = 720,
) -> dict[int, LandmarkPoint]:
    for lm in landmarks.values():
        lm.px = int(round(lm.x * width))
        lm.py = int(round(lm.y * height))
    return landmarks


def _make_valid_pose_landmarks(
    width: int = 1280,
    height: int = 720,
) -> dict[int, LandmarkPoint]:
    # Realistic human A-pose proportions in pixel space on 1280x720:
    # Shoulder width: 0.12 * 1280 = 154 px; Torso length: 0.28 * 720 = 202 px (ratio ~0.76)
    coords = {
        PoseLandmarkIndex.NOSE: (0.50, 0.14),
        PoseLandmarkIndex.LEFT_EYE: (0.485, 0.12),
        PoseLandmarkIndex.RIGHT_EYE: (0.515, 0.12),
        PoseLandmarkIndex.LEFT_SHOULDER: (0.44, 0.26),
        PoseLandmarkIndex.RIGHT_SHOULDER: (0.56, 0.26),
        PoseLandmarkIndex.LEFT_ELBOW: (0.39, 0.40),
        PoseLandmarkIndex.RIGHT_ELBOW: (0.61, 0.40),
        PoseLandmarkIndex.LEFT_WRIST: (0.35, 0.52),
        PoseLandmarkIndex.RIGHT_WRIST: (0.65, 0.52),
        PoseLandmarkIndex.LEFT_HIP: (0.46, 0.54),
        PoseLandmarkIndex.RIGHT_HIP: (0.54, 0.54),
        PoseLandmarkIndex.LEFT_KNEE: (0.45, 0.70),
        PoseLandmarkIndex.RIGHT_KNEE: (0.55, 0.70),
        PoseLandmarkIndex.LEFT_ANKLE: (0.45, 0.86),
        PoseLandmarkIndex.RIGHT_ANKLE: (0.55, 0.86),
        PoseLandmarkIndex.LEFT_HEEL: (0.45, 0.89),
        PoseLandmarkIndex.RIGHT_HEEL: (0.55, 0.89),
    }

    landmarks: dict[int, LandmarkPoint] = {}
    for idx, (x, y) in coords.items():
        landmarks[int(idx)] = LandmarkPoint(
            index=int(idx),
            name=idx.name,
            x=x,
            y=y,
            z=0.0,
            visibility=0.95,
            px=int(round(x * width)),
            py=int(round(y * height)),
        )
    return landmarks


def test_valid_pose_passes_all_five_rules() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    result = engine.evaluate(landmarks)

    assert result.person_detected is True
    assert result.head_visible is True
    assert result.ankles_visible is True
    assert result.heels_visible is True
    assert result.full_body_visible is True
    assert result.front_facing is True
    assert len(result.rules) == 5
    assert [r.key for r in result.rules] == [
        "visibility",
        "body_fill",
        "front_facing",
        "arm_spread",
        "shoulders",
    ]
    assert result.all_rules_passed is True
    assert "Perfect pose" in result.primary_message


def test_angles_are_aspect_ratio_invariant_between_landscape_and_portrait() -> None:
    """Identical pixel geometry on 1280x720 vs 720x1280 must yield identical angles."""
    engine_landscape = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    engine_portrait = PoseRuleEngine(RuleConfig(message_smoothing_window=1))

    lm_land = _make_valid_pose_landmarks(width=1280, height=720)
    # Build identical physical pixel offsets inside a 720x1280 portrait canvas
    lm_port: dict[int, LandmarkPoint] = {}
    for idx, pt in lm_land.items():
        px_p = pt.px - (1280 - 720) // 2
        py_p = pt.py + 180
        lm_port[idx] = LandmarkPoint(
            index=idx,
            name=pt.name,
            x=px_p / 720.0,
            y=py_p / 1280.0,
            z=pt.z,
            visibility=pt.visibility,
            px=px_p,
            py=py_p,
        )

    res_land = engine_landscape.evaluate(lm_land)
    res_port = engine_portrait.evaluate(lm_port)

    assert res_land.shoulder_tilt_deg == pytest.approx(res_port.shoulder_tilt_deg, abs=1e-4)
    assert res_land.left_arm_angle_deg == pytest.approx(res_port.left_arm_angle_deg, abs=1e-4)
    assert res_land.right_arm_angle_deg == pytest.approx(res_port.right_arm_angle_deg, abs=1e-4)
    assert res_land.front_facing is True and res_port.front_facing is True


def test_missing_heels_triggers_step_back() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_HEEL].y = 1.03
    landmarks[PoseLandmarkIndex.RIGHT_HEEL].y = 1.03
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.heels_visible is False
    assert result.full_body_visible is False
    assert result.all_rules_passed is False
    assert "step back" in result.primary_message.lower()


def test_clipped_crown_at_top_edge_fails_head_visibility() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    # Eyes at y=0.025 -> crown extends above y=0 (cut off at top edge of camera)
    landmarks[PoseLandmarkIndex.NOSE].y = 0.04
    landmarks[PoseLandmarkIndex.LEFT_EYE].y = 0.025
    landmarks[PoseLandmarkIndex.RIGHT_EYE].y = 0.025
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.head_visible is False
    assert result.all_rules_passed is False
    assert "step back" in result.primary_message.lower()


def test_sideways_posture_fails_front_facing_rule() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_SHOULDER].x = 0.48
    landmarks[PoseLandmarkIndex.RIGHT_SHOULDER].x = 0.52
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.front_facing is False
    assert result.all_rules_passed is False
    assert result.primary_message == "Face the camera directly"


def test_deep_shoulder_rotation_fails_front_facing_rule() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_SHOULDER].z = -0.25
    landmarks[PoseLandmarkIndex.RIGHT_SHOULDER].z = 0.25

    result = engine.evaluate(landmarks)
    assert result.front_facing is False
    assert result.all_rules_passed is False
    assert result.primary_message == "Face the camera directly"


def test_arms_too_close_triggers_spread_arms_message() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_WRIST].x = 0.44
    landmarks[PoseLandmarkIndex.LEFT_WRIST].y = 0.54
    landmarks[PoseLandmarkIndex.RIGHT_WRIST].x = 0.56
    landmarks[PoseLandmarkIndex.RIGHT_WRIST].y = 0.54
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.primary_message == "Spread your arms a little"


def test_tilted_shoulders_triggers_shoulder_message() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.RIGHT_SHOULDER].y = 0.30
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.shoulder_tilt_deg > 5.0
    assert "shoulders" in result.primary_message.lower()


def test_user_too_far_triggers_step_closer() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.NOSE].y = 0.32
    landmarks[PoseLandmarkIndex.LEFT_EYE].y = 0.30
    landmarks[PoseLandmarkIndex.RIGHT_EYE].y = 0.30
    landmarks[PoseLandmarkIndex.LEFT_ANKLE].y = 0.65
    landmarks[PoseLandmarkIndex.RIGHT_ANKLE].y = 0.65
    landmarks[PoseLandmarkIndex.LEFT_HEEL].y = 0.67
    landmarks[PoseLandmarkIndex.RIGHT_HEEL].y = 0.67
    _sync_pixels(landmarks)

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.primary_message == "Take a step closer"


def test_pose_countdown_locks_ready_after_hold_frames() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1, ready_hold_frames=3))
    landmarks = _make_valid_pose_landmarks()

    r1 = engine.evaluate(landmarks)
    assert not r1.is_locked_ready
    r2 = engine.evaluate(landmarks)
    assert not r2.is_locked_ready
    r3 = engine.evaluate(landmarks)
    assert r3.is_locked_ready is True
    assert r3.hold_progress == pytest.approx(1.0)
    assert "READY FOR CAPTURE" in r3.primary_message


def test_fist_gesture_trigger_detection() -> None:
    landmarks = _make_valid_pose_landmarks()
    assert detect_fist_or_hand_trigger(landmarks) is False

    r_elb = landmarks[PoseLandmarkIndex.RIGHT_ELBOW]
    landmarks[PoseLandmarkIndex.RIGHT_WRIST] = LandmarkPoint(
        index=int(PoseLandmarkIndex.RIGHT_WRIST),
        name="RIGHT_WRIST",
        x=r_elb.x + 0.04,
        y=r_elb.y - 0.10,
        z=0.0,
        visibility=0.95,
        px=int(round((r_elb.x + 0.04) * 1280)),
        py=int(round((r_elb.y - 0.10) * 720)),
    )
    rw = landmarks[PoseLandmarkIndex.RIGHT_WRIST]
    for finger_idx in (
        PoseLandmarkIndex.RIGHT_INDEX,
        PoseLandmarkIndex.RIGHT_PINKY,
        PoseLandmarkIndex.RIGHT_THUMB,
    ):
        landmarks[int(finger_idx)] = LandmarkPoint(
            index=int(finger_idx),
            name=finger_idx.name,
            x=rw.x + 0.01,
            y=rw.y - 0.01,
            z=0.0,
            visibility=0.90,
            px=int(round((rw.x + 0.01) * 1280)),
            py=int(round((rw.y - 0.01) * 720)),
        )

    assert detect_fist_or_hand_trigger(landmarks) is True


def test_anthropometric_extraction_and_silhouette() -> None:
    extractor = AnthropometricExtractor(reference_height_cm=175.0, reference_weight_kg=72.0)
    landmarks = _make_valid_pose_landmarks()
    mask = _DummyMask(720, 1280, 50, 645, 520, 760)

    m = extractor.extract(landmarks, frame_height=720, segmentation_mask=mask)
    assert m is not None
    assert m.full_body_px > 400
    assert m.cm_per_pixel is not None and m.cm_per_pixel > 0
    assert m.scale_method == "segmentation_mask_refined"
    assert m.shoulder_width_cm is not None and m.shoulder_width_cm > 30
    assert m.chest_silhouette_px is not None and m.chest_silhouette_px > 200
    assert m.body_width_cm is not None and m.body_width_cm > 20
    assert m.estimated_upper_size in {"XS", "S", "M", "L", "XL", "XXL"}
    assert m.estimated_lower_size in {"XS", "S", "M", "L", "XL", "XXL"}


def test_cm_per_pixel_dual_zone_mask_refinement_with_split_legs() -> None:
    scale = compute_cm_per_pixel(height_cm=175.0, full_body_px=500.0)
    assert scale == pytest.approx(0.35)

    landmarks = _make_valid_pose_landmarks()
    details_no_mask = estimate_body_height_details(landmarks, frame_height=720)
    assert details_no_mask.method == "anthropometric_proportional"

    full_px = estimate_full_body_height_px(landmarks, frame_height=720)
    scale_from_lm = compute_cm_per_pixel_from_landmarks(
        landmarks, height_cm=175.0, frame_height=720
    )
    assert scale_from_lm == pytest.approx(175.0 / full_px)

    # Split-leg silhouette has empty background gap along nose.px=640 below y=390;
    # dual-zone sole scanner must still detect shoes at y=655 across left/right heels!
    split_mask = _SplitLegMask(720, 1280)
    details_split = estimate_body_height_details(
        landmarks, frame_height=720, segmentation_mask=split_mask
    )
    assert details_split.method == "segmentation_mask_refined"
    assert details_split.crown_y_px == pytest.approx(42.0)
    assert details_split.lowest_foot_y_px == pytest.approx(655.0)
    assert details_split.full_body_px == pytest.approx(613.0)


def test_scale_calibrator_smooths_jitter_and_dampens_spikes() -> None:
    calibrator = ScaleCalibrator(window_size=5, max_relative_jump=0.10)
    for val in (0.300, 0.302, 0.298, 0.301):
        calibrator.update(val)

    stabilized = calibrator.update(0.450)
    assert stabilized == pytest.approx(0.301, abs=0.005)
    assert calibrator.sample_count == 5


def test_weight_and_height_override_recalculates_body_schema() -> None:
    landmarks = _make_valid_pose_landmarks()
    extractor = AnthropometricExtractor(reference_height_cm=170.0, reference_weight_kg=55.0)
    m = extractor.extract(landmarks, frame_height=720)
    assert m is not None

    schema_base = m.to_body_schema(landmarks)
    schema_override = m.to_body_schema(landmarks, weight_kg=105.0, height_cm=190.0)

    assert schema_override.height_cm == 190.0
    assert schema_override.weight_kg == 105.0
    assert schema_override.cm_per_pixel > schema_base.cm_per_pixel
    assert schema_override.shoulder_width_cm > schema_base.shoulder_width_cm
    assert schema_override.body_width_cm > schema_base.body_width_cm


def test_cm_per_pixel_and_fr1_validation_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="height_cm"):
        compute_cm_per_pixel(height_cm=135.0, full_body_px=500.0)

    with pytest.raises(ValueError, match="height_cm"):
        compute_cm_per_pixel(height_cm=215.0, full_body_px=500.0)

    with pytest.raises(ValueError, match="full_body_px"):
        compute_cm_per_pixel(height_cm=175.0, full_body_px=0.0)

    with pytest.raises(ValueError, match="weight_kg"):
        validate_user_anthropometrics(height_cm=175.0, weight_kg=35.0)

    with pytest.raises(ValueError, match="weight_kg"):
        validate_user_anthropometrics(height_cm=175.0, weight_kg=155.0)


def test_to_body_schema_and_pipeline_produce_valid_pydantic_message() -> None:
    pipeline = BodyAnalysisPipeline(height_cm=178.0, weight_kg=74.0, stabilize_scale=True)
    landmarks = _make_valid_pose_landmarks()
    analysis = pipeline.analyze_landmarks(landmarks, frame_height=720, segmentation_mask=None)

    assert analysis.evaluation.all_rules_passed is True
    assert analysis.measurements is not None
    body_msg = analysis.body_schema
    assert isinstance(body_msg, Body)
    assert body_msg.schema_version == "1.0"
    assert body_msg.height_cm == 178.0
    assert body_msg.weight_kg == 74.0
    assert body_msg.cm_per_pixel > 0
    assert body_msg.shoulder_width_cm > 0
    assert body_msg.torso_length_cm > 0
    assert body_msg.body_width_cm > 0
    assert len(body_msg.pose_keypoints) == len(landmarks)
    assert Body.model_validate_json(body_msg.model_dump_json()) == body_msg


def test_tape_measurement_validation_nfr3() -> None:
    samples = [
        TapeValidationSample("P1", 44.0, 42.5, 61.0, 62.8),
        TapeValidationSample("P2", 41.5, 43.0, 58.0, 56.2),
        TapeValidationSample("P3", 46.0, 44.2, 64.0, 61.9),
    ]
    report = evaluate_against_tape_measurements(samples)
    assert report.participant_count == 3
    assert report.shoulder_mae_cm < 3.0
    assert report.torso_mae_cm < 3.0
    assert report.passes_nfr3_limit is True
    assert report.reaches_nfr3_target is True

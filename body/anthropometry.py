from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from body.landmarks import LandmarkPoint, PoseLandmarkIndex
from body.scale import (
    DEFAULT_FRAME_CROWN_RATIO,
    MAX_HEIGHT_CM,
    MAX_WEIGHT_KG,
    MIN_HEIGHT_CM,
    MIN_WEIGHT_KG,
    BodyHeightEstimate,
    ScaleCalibrator,
    compute_cm_per_pixel,
    compute_cm_per_pixel_from_landmarks,
    estimate_body_height_details,
    estimate_full_body_height_px,
    validate_user_anthropometrics,
)
from schemas.body import Body, PoseKeypoint

HEAD_CROWN_OFFSET_RATIO: float = DEFAULT_FRAME_CROWN_RATIO
REFERENCE_BMI: float = 22.0
NFR3_MAX_MAE_CM: float = 4.0
NFR3_TARGET_MAE_CM: float = 3.0


def _weight_width_adjustment_factor(
    height_cm: float | None,
    weight_kg: float | None,
) -> float:
    if not height_cm or height_cm <= 0 or not weight_kg or weight_kg <= 0:
        return 1.0
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)
    return max(0.75, min(1.35, math.sqrt(bmi / REFERENCE_BMI)))


@dataclass
class AnthropometricMeasurements:
    full_body_px: float
    shoulder_width_px: float
    hip_width_px: float
    torso_length_px: float
    left_arm_length_px: float
    right_arm_length_px: float
    inseam_leg_length_px: float
    shoulder_to_hip_ratio: float
    cm_per_pixel: float | None = None
    chest_silhouette_px: float | None = None
    waist_silhouette_px: float | None = None
    hip_silhouette_px: float | None = None
    reference_height_cm: float | None = None
    reference_weight_kg: float | None = None
    shoulder_width_cm: float | None = None
    hip_width_cm: float | None = None
    torso_length_cm: float | None = None
    arm_length_cm: float | None = None
    leg_length_cm: float | None = None
    chest_width_cm: float | None = None
    waist_width_cm: float | None = None
    hip_silhouette_cm: float | None = None
    body_width_cm: float | None = None
    estimated_upper_size: str | None = None
    estimated_lower_size: str | None = None
    scale_method: str = "anthropometric_proportional"
    scale_confidence: float = 1.0
    chest_span_pts: tuple[int, int, int] | None = None
    waist_span_pts: tuple[int, int, int] | None = None
    hip_span_pts: tuple[int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {}
        for k, v in asdict(self).items():
            if k in {"chest_span_pts", "waist_span_pts", "hip_span_pts"}:
                continue
            data[k] = round(v, 4) if isinstance(v, float) else v
        return data

    def to_body_schema(
        self,
        landmarks: dict[int, LandmarkPoint],
        weight_kg: float | None = None,
        height_cm: float | None = None,
    ) -> Body:
        """Build and validate a `schemas.body.Body` message from extracted measurements."""
        resolved_height = (
            height_cm
            if height_cm is not None
            else (self.reference_height_cm if self.reference_height_cm is not None else 175.0)
        )
        resolved_weight = (
            weight_kg
            if weight_kg is not None
            else (self.reference_weight_kg if self.reference_weight_kg is not None else 70.0)
        )
        resolved_height, resolved_weight = validate_user_anthropometrics(
            resolved_height, resolved_weight
        )

        same_ref_height = (
            self.reference_height_cm is not None
            and abs(resolved_height - self.reference_height_cm) < 1e-6
        )
        if same_ref_height and self.cm_per_pixel is not None and self.cm_per_pixel > 0:
            cm_per_px = self.cm_per_pixel
        else:
            cm_per_px = compute_cm_per_pixel(resolved_height, self.full_body_px)

        shoulder_cm = self.shoulder_width_px * cm_per_px
        torso_cm = self.torso_length_px * cm_per_px

        weight_factor = _weight_width_adjustment_factor(resolved_height, resolved_weight)
        base_silhouette_px = (
            self.waist_silhouette_px
            or self.chest_silhouette_px
            or self.hip_silhouette_px
        )
        if base_silhouette_px is not None:
            resolved_body_width_cm = (
                base_silhouette_px * cm_per_px * (0.7 + 0.3 * weight_factor)
            )
        else:
            resolved_body_width_cm = shoulder_cm * 0.82 * weight_factor

        if not landmarks:
            raise ValueError("At least one pose keypoint is required to build Body schema.")

        keypoints = [
            PoseKeypoint(
                name=lm.name.lower(),
                x_px=round(float(max(0, lm.px)), 2),
                y_px=round(float(max(0, lm.py)), 2),
                confidence=round(min(1.0, max(0.0, float(lm.visibility))), 4),
            )
            for _, lm in sorted(landmarks.items())
        ]

        return Body(
            schema_version="1.0",
            height_cm=round(float(resolved_height), 2),
            weight_kg=round(float(resolved_weight), 2),
            cm_per_pixel=round(float(cm_per_px), 5),
            shoulder_width_cm=round(float(shoulder_cm), 2),
            torso_length_cm=round(float(torso_cm), 2),
            body_width_cm=round(float(resolved_body_width_cm), 2),
            pose_keypoints=keypoints,
        )


@dataclass(frozen=True)
class TapeValidationSample:
    """Comparison entry between system-predicted measurements and physical tape measurements."""

    participant_id: str
    tape_shoulder_width_cm: float
    pred_shoulder_width_cm: float
    tape_torso_length_cm: float
    pred_torso_length_cm: float

    @property
    def shoulder_abs_error_cm(self) -> float:
        return abs(self.pred_shoulder_width_cm - self.tape_shoulder_width_cm)

    @property
    def torso_abs_error_cm(self) -> float:
        return abs(self.pred_torso_length_cm - self.tape_torso_length_cm)


@dataclass(frozen=True)
class TapeValidationReport:
    """Summary metrics verifying NFR-3 (MAE <= 4.0 cm, target <= 3.0 cm on >= 3 participants)."""

    participant_count: int
    shoulder_mae_cm: float
    torso_mae_cm: float
    shoulder_bias_cm: float
    torso_bias_cm: float
    shoulder_max_error_cm: float
    torso_max_error_cm: float
    passes_nfr3_limit: bool
    reaches_nfr3_target: bool


def evaluate_against_tape_measurements(
    samples: list[TapeValidationSample],
) -> TapeValidationReport:
    """Compute MAE, bias, and NFR-3 compliance against physical tape measurements."""
    if not samples:
        raise ValueError("At least one TapeValidationSample is required.")

    n = len(samples)
    shoulder_abs = [s.shoulder_abs_error_cm for s in samples]
    torso_abs = [s.torso_abs_error_cm for s in samples]
    shoulder_signed = [s.pred_shoulder_width_cm - s.tape_shoulder_width_cm for s in samples]
    torso_signed = [s.pred_torso_length_cm - s.tape_torso_length_cm for s in samples]

    shoulder_mae = sum(shoulder_abs) / n
    torso_mae = sum(torso_abs) / n
    shoulder_bias = sum(shoulder_signed) / n
    torso_bias = sum(torso_signed) / n

    passes_limit = shoulder_mae <= NFR3_MAX_MAE_CM and torso_mae <= NFR3_MAX_MAE_CM
    reaches_target = shoulder_mae <= NFR3_TARGET_MAE_CM and torso_mae <= NFR3_TARGET_MAE_CM

    return TapeValidationReport(
        participant_count=n,
        shoulder_mae_cm=round(shoulder_mae, 4),
        torso_mae_cm=round(torso_mae, 4),
        shoulder_bias_cm=round(shoulder_bias, 4),
        torso_bias_cm=round(torso_bias, 4),
        shoulder_max_error_cm=round(max(shoulder_abs), 4),
        torso_max_error_cm=round(max(torso_abs), 4),
        passes_nfr3_limit=passes_limit,
        reaches_nfr3_target=reaches_target,
    )


def _dist_px(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return math.hypot(float(a.px - b.px), float(a.py - b.py))


def _measure_mask_row_span(
    mask: Any,
    row_y: int,
    center_x: int,
    max_half_width: int,
) -> tuple[float, int, int, int] | None:
    h, w = mask.shape[:2]
    if not (0 <= row_y < h):
        return None
    x_min = max(0, center_x - max_half_width)
    x_max = min(w - 1, center_x + max_half_width)
    if x_max <= x_min:
        return None

    row_slice = mask[row_y, x_min : x_max + 1]
    if hasattr(row_slice, "ndim") and hasattr(row_slice, "nonzero"):
        fg_arr = (row_slice > 127).nonzero()[0]
        if len(fg_arr) < 4:
            return None
        left_x = int(x_min + int(fg_arr[0]))
        right_x = int(x_min + int(fg_arr[-1]))
    else:
        fg_indices = [idx for idx, val in enumerate(row_slice) if int(val) > 127]
        if len(fg_indices) < 4:
            return None
        left_x = int(x_min + fg_indices[0])
        right_x = int(x_min + fg_indices[-1])

    span = float(right_x - left_x)
    if span < 10.0:
        return None
    return span, left_x, right_x, row_y


def _estimate_upper_size(
    shoulder_cm: float | None, chest_cm: float | None
) -> str | None:
    ref = chest_cm if (chest_cm is not None and 24.0 <= chest_cm <= 70.0) else shoulder_cm
    if ref is None:
        return None
    if ref < 37.0:
        return "XS"
    if ref < 41.0:
        return "S"
    if ref < 45.0:
        return "M"
    if ref < 49.0:
        return "L"
    if ref < 53.0:
        return "XL"
    return "XXL"


def _estimate_lower_size(
    hip_joint_cm: float | None,
    hip_silhouette_cm: float | None,
    waist_cm: float | None,
) -> str | None:
    if hip_silhouette_cm is not None and 24.0 <= hip_silhouette_cm <= 65.0:
        ref = hip_silhouette_cm
    elif waist_cm is not None and 22.0 <= waist_cm <= 60.0:
        ref = waist_cm * 1.12
    elif hip_joint_cm is not None:
        ref = hip_joint_cm * 1.45
    else:
        return None

    if ref < 33.0:
        return "XS"
    if ref < 36.5:
        return "S"
    if ref < 40.0:
        return "M"
    if ref < 43.5:
        return "L"
    if ref < 47.0:
        return "XL"
    return "XXL"


class AnthropometricExtractor:
    def __init__(
        self,
        reference_height_cm: float | None = 175.0,
        reference_weight_kg: float | None = 70.0,
        stabilize_scale: bool = False,
        scale_window_size: int = 15,
    ) -> None:
        self.reference_height_cm = reference_height_cm
        self.reference_weight_kg = reference_weight_kg
        self._calibrator: ScaleCalibrator | None = (
            ScaleCalibrator(window_size=scale_window_size) if stabilize_scale else None
        )

    def reset_calibration(self) -> None:
        if self._calibrator is not None:
            self._calibrator.reset()

    def extract(
        self,
        landmarks: dict[int, LandmarkPoint] | None,
        frame_height: int,
        segmentation_mask: Any = None,
    ) -> AnthropometricMeasurements | None:
        if not landmarks:
            return None

        nose = landmarks.get(PoseLandmarkIndex.NOSE)
        l_shoulder = landmarks.get(PoseLandmarkIndex.LEFT_SHOULDER)
        r_shoulder = landmarks.get(PoseLandmarkIndex.RIGHT_SHOULDER)
        l_elbow = landmarks.get(PoseLandmarkIndex.LEFT_ELBOW)
        r_elbow = landmarks.get(PoseLandmarkIndex.RIGHT_ELBOW)
        l_wrist = landmarks.get(PoseLandmarkIndex.LEFT_WRIST)
        r_wrist = landmarks.get(PoseLandmarkIndex.RIGHT_WRIST)
        l_hip = landmarks.get(PoseLandmarkIndex.LEFT_HIP)
        r_hip = landmarks.get(PoseLandmarkIndex.RIGHT_HIP)
        l_knee = landmarks.get(PoseLandmarkIndex.LEFT_KNEE)
        r_knee = landmarks.get(PoseLandmarkIndex.RIGHT_KNEE)
        l_ankle = landmarks.get(PoseLandmarkIndex.LEFT_ANKLE)
        r_ankle = landmarks.get(PoseLandmarkIndex.RIGHT_ANKLE)
        l_heel = landmarks.get(PoseLandmarkIndex.LEFT_HEEL)
        r_heel = landmarks.get(PoseLandmarkIndex.RIGHT_HEEL)

        required = (
            nose,
            l_shoulder,
            r_shoulder,
            l_elbow,
            r_elbow,
            l_wrist,
            r_wrist,
            l_hip,
            r_hip,
            l_ankle,
            r_ankle,
            l_heel,
            r_heel,
        )
        if any(pt is None for pt in required):
            return None

        assert nose and l_shoulder and r_shoulder
        assert l_elbow and r_elbow and l_wrist and r_wrist
        assert l_hip and r_hip and l_ankle and r_ankle and l_heel and r_heel

        height_details: BodyHeightEstimate = estimate_body_height_details(
            landmarks,
            frame_height=frame_height,
            segmentation_mask=segmentation_mask,
        )
        full_body_px = height_details.full_body_px

        shoulder_width_px = _dist_px(l_shoulder, r_shoulder)
        hip_width_px = _dist_px(l_hip, r_hip)

        mid_shoulder_x = 0.5 * (l_shoulder.px + r_shoulder.px)
        mid_shoulder_y = 0.5 * (l_shoulder.py + r_shoulder.py)
        mid_hip_x = 0.5 * (l_hip.px + r_hip.px)
        mid_hip_y = 0.5 * (l_hip.py + r_hip.py)
        torso_length_px = math.hypot(
            mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y
        )

        left_arm_length_px = _dist_px(l_shoulder, l_elbow) + _dist_px(l_elbow, l_wrist)
        right_arm_length_px = _dist_px(r_shoulder, r_elbow) + _dist_px(
            r_elbow, r_wrist
        )

        if l_knee and r_knee:
            left_leg_px = _dist_px(l_hip, l_knee) + _dist_px(l_knee, l_ankle)
            right_leg_px = _dist_px(r_hip, r_knee) + _dist_px(r_knee, r_ankle)
            inseam_leg_length_px = 0.5 * (left_leg_px + right_leg_px)
        else:
            inseam_leg_length_px = 0.5 * (
                _dist_px(l_hip, l_ankle) + _dist_px(r_hip, r_ankle)
            )

        shoulder_to_hip_ratio = shoulder_width_px / max(1.0, hip_width_px)

        chest_silhouette_px: float | None = None
        waist_silhouette_px: float | None = None
        hip_silhouette_px: float | None = None
        chest_span_pts: tuple[int, int, int] | None = None
        waist_span_pts: tuple[int, int, int] | None = None
        hip_span_pts: tuple[int, int, int] | None = None

        if segmentation_mask is not None and shoulder_width_px > 10.0:
            center_x = int(round(0.5 * (mid_shoulder_x + mid_hip_x)))
            max_half_w = int(round(shoulder_width_px * 0.75))
            chest_y = int(round(mid_shoulder_y + 0.28 * (mid_hip_y - mid_shoulder_y)))
            waist_y = int(round(mid_shoulder_y + 0.62 * (mid_hip_y - mid_shoulder_y)))
            hip_y = int(round(mid_hip_y))

            chest_res = _measure_mask_row_span(
                segmentation_mask, chest_y, center_x, max_half_w
            )
            if chest_res is not None:
                chest_silhouette_px = chest_res[0]
                chest_span_pts = (chest_res[1], chest_res[2], chest_res[3])

            waist_res = _measure_mask_row_span(
                segmentation_mask, waist_y, center_x, max_half_w
            )
            if waist_res is not None:
                waist_silhouette_px = waist_res[0]
                waist_span_pts = (waist_res[1], waist_res[2], waist_res[3])

            hip_res = _measure_mask_row_span(
                segmentation_mask, hip_y, center_x, max_half_w
            )
            if hip_res is not None:
                hip_silhouette_px = hip_res[0]
                hip_span_pts = (hip_res[1], hip_res[2], hip_res[3])

        raw_cm_per_px = (
            compute_cm_per_pixel(
                self.reference_height_cm,
                full_body_px,
                validate_height_range=False,
            )
            if self.reference_height_cm and self.reference_height_cm > 0
            else None
        )
        if raw_cm_per_px is not None and self._calibrator is not None:
            cm_per_px: float | None = self._calibrator.update(raw_cm_per_px)
        else:
            cm_per_px = raw_cm_per_px

        shoulder_width_cm = (
            shoulder_width_px * cm_per_px if cm_per_px is not None else None
        )
        hip_width_cm = hip_width_px * cm_per_px if cm_per_px is not None else None
        torso_length_cm = (
            torso_length_px * cm_per_px if cm_per_px is not None else None
        )
        arm_length_cm = (
            0.5 * (left_arm_length_px + right_arm_length_px) * cm_per_px
            if cm_per_px is not None
            else None
        )
        leg_length_cm = (
            inseam_leg_length_px * cm_per_px if cm_per_px is not None else None
        )
        chest_width_cm = (
            chest_silhouette_px * cm_per_px
            if (cm_per_px is not None and chest_silhouette_px is not None)
            else None
        )
        waist_width_cm = (
            waist_silhouette_px * cm_per_px
            if (cm_per_px is not None and waist_silhouette_px is not None)
            else None
        )
        hip_silhouette_cm = (
            hip_silhouette_px * cm_per_px
            if (cm_per_px is not None and hip_silhouette_px is not None)
            else None
        )

        weight_factor = _weight_width_adjustment_factor(
            self.reference_height_cm, self.reference_weight_kg
        )
        base_silhouette_cm = waist_width_cm or chest_width_cm or hip_silhouette_cm
        if base_silhouette_cm is not None:
            body_width_cm: float | None = base_silhouette_cm * (0.7 + 0.3 * weight_factor)
        elif shoulder_width_cm is not None:
            body_width_cm = shoulder_width_cm * 0.82 * weight_factor
        else:
            body_width_cm = None

        estimated_upper_size = _estimate_upper_size(
            shoulder_width_cm, chest_width_cm
        )
        estimated_lower_size = _estimate_lower_size(
            hip_width_cm, hip_silhouette_cm, waist_width_cm
        )

        return AnthropometricMeasurements(
            full_body_px=full_body_px,
            shoulder_width_px=shoulder_width_px,
            hip_width_px=hip_width_px,
            torso_length_px=torso_length_px,
            left_arm_length_px=left_arm_length_px,
            right_arm_length_px=right_arm_length_px,
            inseam_leg_length_px=inseam_leg_length_px,
            shoulder_to_hip_ratio=shoulder_to_hip_ratio,
            cm_per_pixel=cm_per_px,
            chest_silhouette_px=chest_silhouette_px,
            waist_silhouette_px=waist_silhouette_px,
            hip_silhouette_px=hip_silhouette_px,
            reference_height_cm=self.reference_height_cm,
            reference_weight_kg=self.reference_weight_kg,
            shoulder_width_cm=shoulder_width_cm,
            hip_width_cm=hip_width_cm,
            torso_length_cm=torso_length_cm,
            arm_length_cm=arm_length_cm,
            leg_length_cm=leg_length_cm,
            chest_width_cm=chest_width_cm,
            waist_width_cm=waist_width_cm,
            hip_silhouette_cm=hip_silhouette_cm,
            body_width_cm=body_width_cm,
            estimated_upper_size=estimated_upper_size,
            estimated_lower_size=estimated_lower_size,
            scale_method=height_details.method,
            scale_confidence=height_details.confidence,
            chest_span_pts=chest_span_pts,
            waist_span_pts=waist_span_pts,
            hip_span_pts=hip_span_pts,
        )


__all__ = [
    "AnthropometricExtractor",
    "AnthropometricMeasurements",
    "BodyHeightEstimate",
    "HEAD_CROWN_OFFSET_RATIO",
    "MAX_HEIGHT_CM",
    "MAX_WEIGHT_KG",
    "MIN_HEIGHT_CM",
    "MIN_WEIGHT_KG",
    "NFR3_MAX_MAE_CM",
    "NFR3_TARGET_MAE_CM",
    "ScaleCalibrator",
    "TapeValidationReport",
    "TapeValidationSample",
    "compute_cm_per_pixel",
    "compute_cm_per_pixel_from_landmarks",
    "estimate_body_height_details",
    "estimate_full_body_height_px",
    "evaluate_against_tape_measurements",
    "validate_user_anthropometrics",
]

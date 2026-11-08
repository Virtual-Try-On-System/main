from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from body.landmarks import LandmarkPoint, PoseLandmarkIndex


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
    shoulder_width_cm: float | None = None
    hip_width_cm: float | None = None
    torso_length_cm: float | None = None
    arm_length_cm: float | None = None
    leg_length_cm: float | None = None
    chest_width_cm: float | None = None
    waist_width_cm: float | None = None
    hip_silhouette_cm: float | None = None
    estimated_upper_size: str | None = None
    estimated_lower_size: str | None = None
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
    def __init__(self, reference_height_cm: float | None = 175.0) -> None:
        self.reference_height_cm = reference_height_cm

    def extract(
        self,
        landmarks: dict[int, LandmarkPoint] | None,
        frame_height: int,
        segmentation_mask: Any = None,
    ) -> AnthropometricMeasurements | None:
        if not landmarks:
            return None

        nose = landmarks.get(PoseLandmarkIndex.NOSE)
        l_eye = landmarks.get(PoseLandmarkIndex.LEFT_EYE)
        r_eye = landmarks.get(PoseLandmarkIndex.RIGHT_EYE)
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

        eye_py = min(
            l_eye.py if l_eye else nose.py,
            r_eye.py if r_eye else nose.py,
        )
        head_top_py = max(0.0, float(eye_py) - 0.04 * frame_height)
        lowest_heel_py = float(max(l_heel.py, r_heel.py, l_ankle.py, r_ankle.py))
        full_body_px = max(1.0, lowest_heel_py - head_top_py)

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

        cm_per_px = (
            (self.reference_height_cm / full_body_px)
            if self.reference_height_cm and self.reference_height_cm > 0
            else None
        )

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
            shoulder_width_cm=shoulder_width_cm,
            hip_width_cm=hip_width_cm,
            torso_length_cm=torso_length_cm,
            arm_length_cm=arm_length_cm,
            leg_length_cm=leg_length_cm,
            chest_width_cm=chest_width_cm,
            waist_width_cm=waist_width_cm,
            hip_silhouette_cm=hip_silhouette_cm,
            estimated_upper_size=estimated_upper_size,
            estimated_lower_size=estimated_lower_size,
            chest_span_pts=chest_span_pts,
            waist_span_pts=waist_span_pts,
            hip_span_pts=hip_span_pts,
        )

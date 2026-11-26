from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any

from body.landmarks import LandmarkPoint, PoseLandmarkIndex

MIN_HEIGHT_CM: float = 140.0
MAX_HEIGHT_CM: float = 210.0
MIN_WEIGHT_KG: float = 40.0
MAX_WEIGHT_KG: float = 150.0

# Standard human anthropometric proportion: vertical eye-to-heel distance is ~93.6%
# of total standing stature (crown-to-heel), leaving ~6.4% above the eye line.
EYE_TO_CROWN_STATURE_RATIO: float = 0.064 / 0.936
EYE_TO_SHOULDER_CROWN_FACTOR: float = 0.52
DEFAULT_FRAME_CROWN_RATIO: float = 0.04


@dataclass(frozen=True)
class BodyHeightEstimate:
    """Detailed breakdown of how full-body pixel height and cm/pixel scale were derived."""

    full_body_px: float
    crown_y_px: float
    lowest_foot_y_px: float
    skeletal_chain_px: float
    method: str
    confidence: float


def validate_user_anthropometrics(
    height_cm: float,
    weight_kg: float = 70.0,
) -> tuple[float, float]:
    """Validate user height and weight against FR-1 bounds (140-210 cm, 40-150 kg)."""
    if (
        isinstance(height_cm, bool)
        or not isinstance(height_cm, (int, float))
        or not math.isfinite(height_cm)
        or not (MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM)
    ):
        raise ValueError(
            f"height_cm must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f} cm, "
            f"got {height_cm!r}."
        )
    if (
        isinstance(weight_kg, bool)
        or not isinstance(weight_kg, (int, float))
        or not math.isfinite(weight_kg)
        or not (MIN_WEIGHT_KG <= float(weight_kg) <= MAX_WEIGHT_KG)
    ):
        raise ValueError(
            f"weight_kg must be between {MIN_WEIGHT_KG:.0f} and {MAX_WEIGHT_KG:.0f} kg, "
            f"got {weight_kg!r}."
        )
    return float(height_cm), float(weight_kg)


def _find_mask_top_y(
    mask: Any,
    x1: int,
    x2: int,
    y_start: int,
    y_end: int,
) -> int | None:
    """Find the highest foreground row y in [y_start, y_end] within horizontal band [x1, x2]."""
    if mask is None or not hasattr(mask, "shape"):
        return None
    h, w = mask.shape[:2]
    x1_c, x2_c = max(0, x1), min(w - 1, x2)
    y1_c, y2_c = max(0, y_start), min(h - 1, y_end)
    if x2_c <= x1_c or y2_c < y1_c:
        return None

    # Fast vectorized path for NumPy ndarrays
    if hasattr(mask, "ndim") and hasattr(mask, "any"):
        roi = mask[y1_c : y2_c + 1, x1_c : x2_c + 1] > 127
        row_hits = roi.any(axis=1)
        if not bool(row_hits.any()):
            return None
        first_offset = int(row_hits.argmax())
        return y1_c + first_offset

    for y in range(y1_c, y2_c + 1):
        row = mask[y, x1_c : x2_c + 1]
        if any(int(v) > 127 for v in row):
            return y
    return None


def _find_mask_bottom_y(
    mask: Any,
    x1: int,
    x2: int,
    y_start: int,
    y_end: int,
) -> int | None:
    """Find the lowest foreground row y in [y_start, y_end] within horizontal band [x1, x2]."""
    if mask is None or not hasattr(mask, "shape"):
        return None
    h, w = mask.shape[:2]
    x1_c, x2_c = max(0, x1), min(w - 1, x2)
    y1_c, y2_c = max(0, y_start), min(h - 1, y_end)
    if x2_c <= x1_c or y2_c < y1_c:
        return None

    if hasattr(mask, "ndim") and hasattr(mask, "any"):
        roi = mask[y1_c : y2_c + 1, x1_c : x2_c + 1] > 127
        row_hits = roi.any(axis=1)
        if not bool(row_hits.any()):
            return None
        rev_offset = int(row_hits[::-1].argmax())
        return y2_c - rev_offset

    for y in range(y2_c, y1_c - 1, -1):
        row = mask[y, x1_c : x2_c + 1]
        if any(int(v) > 127 for v in row):
            return y
    return None


def _compute_skeletal_chain_height_px(
    landmarks: dict[int, LandmarkPoint],
    crown_y_px: float,
    lowest_foot_y_px: float,
) -> float:
    """Compute piecewise Euclidean length along the body midline from crown to feet."""
    l_sh = landmarks.get(PoseLandmarkIndex.LEFT_SHOULDER)
    r_sh = landmarks.get(PoseLandmarkIndex.RIGHT_SHOULDER)
    l_hip = landmarks.get(PoseLandmarkIndex.LEFT_HIP)
    r_hip = landmarks.get(PoseLandmarkIndex.RIGHT_HIP)
    l_knee = landmarks.get(PoseLandmarkIndex.LEFT_KNEE)
    r_knee = landmarks.get(PoseLandmarkIndex.RIGHT_KNEE)
    l_ank = landmarks.get(PoseLandmarkIndex.LEFT_ANKLE)
    r_ank = landmarks.get(PoseLandmarkIndex.RIGHT_ANKLE)
    nose = landmarks.get(PoseLandmarkIndex.NOSE)

    if not (nose and l_sh and r_sh and l_hip and r_hip and (l_ank or r_ank)):
        return max(1.0, lowest_foot_y_px - crown_y_px)

    mid_sh_x = 0.5 * (l_sh.px + r_sh.px)
    mid_sh_y = 0.5 * (l_sh.py + r_sh.py)
    mid_hip_x = 0.5 * (l_hip.px + r_hip.px)
    mid_hip_y = 0.5 * (l_hip.py + r_hip.py)

    crown_to_sh = max(0.0, mid_sh_y - crown_y_px)
    sh_to_hip = math.hypot(mid_sh_x - mid_hip_x, mid_sh_y - mid_hip_y)

    if l_knee and r_knee and l_ank and r_ank:
        left_leg = math.hypot(l_hip.px - l_knee.px, l_hip.py - l_knee.py) + math.hypot(
            l_knee.px - l_ank.px, l_knee.py - l_ank.py
        )
        right_leg = math.hypot(r_hip.px - r_knee.px, r_hip.py - r_knee.py) + math.hypot(
            r_knee.px - r_ank.px, r_knee.py - r_ank.py
        )
        hip_to_ankle = 0.5 * (left_leg + right_leg)
        ankle_y = 0.5 * (l_ank.py + r_ank.py)
    else:
        ank = l_ank or r_ank
        assert ank is not None
        hip_to_ankle = math.hypot(mid_hip_x - ank.px, mid_hip_y - ank.py)
        ankle_y = float(ank.py)

    ankle_to_sole = max(0.0, lowest_foot_y_px - ankle_y)
    return max(1.0, crown_to_sh + sh_to_hip + hip_to_ankle + ankle_to_sole)


def estimate_body_height_details(
    landmarks: dict[int, LandmarkPoint],
    frame_height: int,
    segmentation_mask: Any = None,
) -> BodyHeightEstimate:
    """Estimate full-body height in pixels using crown projection and dual-zone mask refinement."""
    if frame_height <= 0:
        raise ValueError(f"frame_height must be positive, got {frame_height!r}.")
    if not landmarks:
        raise ValueError("landmarks dictionary cannot be empty.")

    nose = landmarks.get(PoseLandmarkIndex.NOSE)
    if nose is None:
        raise ValueError("Nose landmark is required to estimate full-body pixel height.")

    l_eye = landmarks.get(PoseLandmarkIndex.LEFT_EYE)
    r_eye = landmarks.get(PoseLandmarkIndex.RIGHT_EYE)
    l_ear = landmarks.get(PoseLandmarkIndex.LEFT_EAR)
    r_ear = landmarks.get(PoseLandmarkIndex.RIGHT_EAR)
    l_sh = landmarks.get(PoseLandmarkIndex.LEFT_SHOULDER)
    r_sh = landmarks.get(PoseLandmarkIndex.RIGHT_SHOULDER)

    foot_candidates = [
        landmarks.get(idx)
        for idx in (
            PoseLandmarkIndex.LEFT_HEEL,
            PoseLandmarkIndex.RIGHT_HEEL,
            PoseLandmarkIndex.LEFT_FOOT_INDEX,
            PoseLandmarkIndex.RIGHT_FOOT_INDEX,
            PoseLandmarkIndex.LEFT_ANKLE,
            PoseLandmarkIndex.RIGHT_ANKLE,
        )
    ]
    valid_feet = [pt for pt in foot_candidates if pt is not None]
    if not valid_feet:
        raise ValueError(
            "At least one ankle, heel, or foot landmark is required to estimate body height."
        )

    head_pts = [pt for pt in (l_eye, r_eye, l_ear, r_ear, nose) if pt is not None]
    eye_py = float(
        min(
            l_eye.py if l_eye is not None else nose.py,
            r_eye.py if r_eye is not None else nose.py,
        )
    )
    lowest_foot_py = float(max(pt.py for pt in valid_feet))
    highest_foot_py = float(min(pt.py for pt in valid_feet))
    eye_to_foot_span = max(1.0, lowest_foot_py - eye_py)

    # Anthropometric crown offset scales with the person's apparent pixel size in frame
    if l_sh is not None and r_sh is not None:
        mid_sh_y = 0.5 * (l_sh.py + r_sh.py)
        eye_to_shoulder = max(1.0, mid_sh_y - eye_py)
        proportional_offset = max(
            eye_to_foot_span * EYE_TO_CROWN_STATURE_RATIO,
            eye_to_shoulder * EYE_TO_SHOULDER_CROWN_FACTOR,
        )
        method = "anthropometric_proportional"
    else:
        mid_sh_y = eye_py + 2.0 * DEFAULT_FRAME_CROWN_RATIO * float(frame_height)
        proportional_offset = max(
            eye_to_foot_span * EYE_TO_CROWN_STATURE_RATIO,
            DEFAULT_FRAME_CROWN_RATIO * float(frame_height),
        )
        method = "eye_to_heel_ratio"

    crown_y_px = max(0.0, eye_py - proportional_offset)
    sole_y_px = lowest_foot_py

    # Dual-zone mask refinement: scan head column for crown and full foot span for sole
    if segmentation_mask is not None:
        head_center_x = int(round(nose.px))
        if l_sh is not None and r_sh is not None:
            head_half_w = max(12, int(round(0.40 * abs(l_sh.px - r_sh.px))))
        else:
            head_half_w = 40

        crown_y_min = max(0, int(round(eye_py - 2.2 * proportional_offset)))
        crown_y_max = int(round(mid_sh_y))
        mask_top_y = _find_mask_top_y(
            segmentation_mask,
            x1=head_center_x - head_half_w,
            x2=head_center_x + head_half_w,
            y_start=crown_y_min,
            y_end=crown_y_max,
        )
        if mask_top_y is not None and mask_top_y <= eye_py:
            crown_y_px = float(mask_top_y)
            method = "segmentation_mask_refined"

        foot_pad_x = max(16, head_half_w // 2)
        foot_x1 = min(pt.px for pt in valid_feet) - foot_pad_x
        foot_x2 = max(pt.px for pt in valid_feet) + foot_pad_x
        sole_y_min = max(0, int(round(highest_foot_py - 12)))
        sole_y_max = min(frame_height - 1, int(round(lowest_foot_py + 0.05 * frame_height)))
        mask_bottom_y = _find_mask_bottom_y(
            segmentation_mask,
            x1=foot_x1,
            x2=foot_x2,
            y_start=sole_y_min,
            y_end=sole_y_max,
        )
        if mask_bottom_y is not None and mask_bottom_y >= lowest_foot_py - 12:
            sole_y_px = float(max(lowest_foot_py, mask_bottom_y))
            method = "segmentation_mask_refined"

    full_body_px = sole_y_px - crown_y_px
    if full_body_px <= 0.0:
        raise ValueError(
            f"Computed full-body pixel height must be positive, got {full_body_px:.2f}."
        )

    skeletal_chain_px = _compute_skeletal_chain_height_px(
        landmarks, crown_y_px=crown_y_px, lowest_foot_y_px=sole_y_px
    )

    visibilities = [pt.visibility for pt in [*head_pts, *valid_feet]]
    mean_vis = sum(visibilities) / max(1, len(visibilities))
    confidence = round(min(1.0, max(0.0, mean_vis)), 4)

    return BodyHeightEstimate(
        full_body_px=full_body_px,
        crown_y_px=crown_y_px,
        lowest_foot_y_px=sole_y_px,
        skeletal_chain_px=skeletal_chain_px,
        method=method,
        confidence=confidence,
    )


def estimate_full_body_height_px(
    landmarks: dict[int, LandmarkPoint],
    frame_height: int,
    head_crown_offset_ratio: float | None = None,
    segmentation_mask: Any = None,
) -> float:
    """Estimate full-body vertical pixel height from head crown to lowest heel/ankle."""
    if head_crown_offset_ratio is not None:
        if frame_height <= 0:
            raise ValueError(f"frame_height must be positive, got {frame_height!r}.")
        if not landmarks:
            raise ValueError("landmarks dictionary cannot be empty.")
        nose = landmarks.get(PoseLandmarkIndex.NOSE)
        if nose is None:
            raise ValueError("Nose landmark is required to estimate full-body pixel height.")
        l_eye = landmarks.get(PoseLandmarkIndex.LEFT_EYE)
        r_eye = landmarks.get(PoseLandmarkIndex.RIGHT_EYE)
        foot_points = [
            landmarks.get(idx)
            for idx in (
                PoseLandmarkIndex.LEFT_HEEL,
                PoseLandmarkIndex.RIGHT_HEEL,
                PoseLandmarkIndex.LEFT_ANKLE,
                PoseLandmarkIndex.RIGHT_ANKLE,
            )
        ]
        valid_feet = [pt for pt in foot_points if pt is not None]
        if not valid_feet:
            raise ValueError("At least one ankle or heel landmark is required.")
        eye_py = min(
            l_eye.py if l_eye is not None else nose.py,
            r_eye.py if r_eye is not None else nose.py,
        )
        head_top_py = max(0.0, float(eye_py) - head_crown_offset_ratio * float(frame_height))
        lowest_foot_py = float(max(pt.py for pt in valid_feet))
        full_body_px = lowest_foot_py - head_top_py
        if full_body_px <= 0.0:
            raise ValueError(
                f"Computed full-body pixel height must be positive, got {full_body_px:.2f}."
            )
        return full_body_px

    details = estimate_body_height_details(
        landmarks,
        frame_height=frame_height,
        segmentation_mask=segmentation_mask,
    )
    return details.full_body_px


def compute_cm_per_pixel(
    height_cm: float,
    full_body_px: float,
    *,
    validate_height_range: bool = True,
) -> float:
    """Compute the centimetre-per-pixel scale factor from user height (cm) and body span (px)."""
    if (
        isinstance(height_cm, bool)
        or not isinstance(height_cm, (int, float))
        or not math.isfinite(height_cm)
        or height_cm <= 0.0
    ):
        raise ValueError(f"height_cm must be a positive number, got {height_cm!r}.")
    if validate_height_range and not (MIN_HEIGHT_CM <= float(height_cm) <= MAX_HEIGHT_CM):
        raise ValueError(
            f"height_cm must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f} cm, "
            f"got {height_cm!r}."
        )
    if (
        isinstance(full_body_px, bool)
        or not isinstance(full_body_px, (int, float))
        or not math.isfinite(full_body_px)
        or full_body_px <= 0.0
    ):
        raise ValueError(f"full_body_px must be a positive number, got {full_body_px!r}.")
    return float(height_cm) / float(full_body_px)


def compute_cm_per_pixel_from_landmarks(
    landmarks: dict[int, LandmarkPoint],
    height_cm: float,
    frame_height: int,
    *,
    segmentation_mask: Any = None,
    validate_height_range: bool = True,
) -> float:
    """Estimate full-body height in pixels from landmarks/mask and return the cm/pixel scale."""
    details = estimate_body_height_details(
        landmarks,
        frame_height=frame_height,
        segmentation_mask=segmentation_mask,
    )
    return compute_cm_per_pixel(
        height_cm=height_cm,
        full_body_px=details.full_body_px,
        validate_height_range=validate_height_range,
    )


class ScaleCalibrator:
    """Temporal stabilizer for cm/pixel scale across consecutive camera frames.

    Uses a rolling window median with outlier rejection so single-frame MediaPipe
    landmark jitter does not cause centimetre measurements to fluctuate wildly.
    """

    def __init__(self, window_size: int = 15, max_relative_jump: float = 0.20) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if max_relative_jump <= 0.0:
            raise ValueError("max_relative_jump must be positive.")
        self.window_size = window_size
        self.max_relative_jump = max_relative_jump
        self._samples: deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def current_cm_per_pixel(self) -> float | None:
        if not self._samples:
            return None
        return float(median(self._samples))

    def update(self, cm_per_pixel: float) -> float:
        if not math.isfinite(cm_per_pixel) or cm_per_pixel <= 0.0:
            raise ValueError(f"cm_per_pixel must be positive, got {cm_per_pixel!r}.")

        current = self.current_cm_per_pixel
        if current is not None and len(self._samples) >= 3:
            rel_diff = abs(cm_per_pixel - current) / current
            if rel_diff > self.max_relative_jump:
                # Dampen extreme single-frame spike toward current median
                direction = 1.0 if cm_per_pixel > current else -1.0
                cm_per_pixel = current * (1.0 + direction * self.max_relative_jump)

        self._samples.append(float(cm_per_pixel))
        return float(median(self._samples))

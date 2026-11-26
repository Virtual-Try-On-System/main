from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from body.landmarks import LandmarkPoint, PoseLandmarkIndex
from body.scale import EYE_TO_CROWN_STATURE_RATIO


@dataclass
class RuleConfig:
    min_visibility: float = 0.60
    frame_margin: float = 0.02
    min_body_fill_ratio: float = 0.62
    max_body_fill_ratio: float = 0.92
    max_center_offset_x: float = 0.18
    max_shoulder_tilt_deg: float = 5.0
    min_arm_spread_deg: float = 14.0
    max_arm_spread_deg: float = 65.0
    # Measured in isotropic pixel space (shoulder_width_px / torso_length_px).
    # Frontal human stance is typically ~0.65-0.90; profile/sideways drops below 0.45.
    min_shoulder_to_torso_ratio: float = 0.45
    max_shoulder_depth_diff: float = 0.35
    max_nose_off_center_ratio: float = 0.45
    message_smoothing_window: int = 10
    ready_hold_frames: int = 45


@dataclass
class RuleCheckItem:
    key: str
    label: str
    passed: bool
    metric_text: str
    guidance_message: str | None = None


@dataclass
class PoseEvaluationResult:
    person_detected: bool
    all_rules_passed: bool
    is_locked_ready: bool
    hold_progress: float
    primary_message: str
    rules: list[RuleCheckItem] = field(default_factory=list)
    head_visible: bool = False
    ankles_visible: bool = False
    heels_visible: bool = False
    front_facing: bool = False
    body_fill_ratio: float = 0.0
    shoulder_tilt_deg: float = 0.0
    left_arm_angle_deg: float = 0.0
    right_arm_angle_deg: float = 0.0
    fist_gesture_detected: bool = False

    @property
    def full_body_visible(self) -> bool:
        return self.head_visible and self.ankles_visible and self.heels_visible


def _metric_xy(lm: LandmarkPoint) -> tuple[float, float]:
    """Return isotropic 2D coordinates (preferring pixel coordinates over normalized [0,1]).

    Normalized coordinates distort angles on non-square frames (e.g. 16:9 1280x720).
    Using pixel coordinates ensures tilt and abduction angles are aspect-ratio invariant.
    """
    if lm.px != 0 or lm.py != 0:
        return float(lm.px), float(lm.py)
    return float(lm.x), float(lm.y)


def _calculate_angle_between_vectors(
    vx1: float, vy1: float, vx2: float, vy2: float
) -> float:
    mag1 = math.hypot(vx1, vy1)
    mag2 = math.hypot(vx2, vy2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_theta = (vx1 * vx2 + vy1 * vy2) / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def detect_fist_or_hand_trigger(
    landmarks: dict[int, LandmarkPoint] | None,
    min_visibility: float = 0.55,
) -> bool:
    """Detect a raised-hand or closed-fist trigger gesture from pose hand landmarks (FR-8).

    Checks whether either hand (wrist + index/pinky/thumb cluster) is raised above elbow level
    with compact finger-to-wrist proximity indicative of a fist gesture, measured in isotropic
    pixel coordinates.
    """
    if not landmarks:
        return False

    hand_triplets = [
        (
            landmarks.get(PoseLandmarkIndex.LEFT_SHOULDER),
            landmarks.get(PoseLandmarkIndex.LEFT_ELBOW),
            landmarks.get(PoseLandmarkIndex.LEFT_WRIST),
            landmarks.get(PoseLandmarkIndex.LEFT_INDEX),
            landmarks.get(PoseLandmarkIndex.LEFT_PINKY),
            landmarks.get(PoseLandmarkIndex.LEFT_THUMB),
        ),
        (
            landmarks.get(PoseLandmarkIndex.RIGHT_SHOULDER),
            landmarks.get(PoseLandmarkIndex.RIGHT_ELBOW),
            landmarks.get(PoseLandmarkIndex.RIGHT_WRIST),
            landmarks.get(PoseLandmarkIndex.RIGHT_INDEX),
            landmarks.get(PoseLandmarkIndex.RIGHT_PINKY),
            landmarks.get(PoseLandmarkIndex.RIGHT_THUMB),
        ),
    ]

    for sh, elb, wr, idx_pt, pky_pt, thb_pt in hand_triplets:
        if sh is None or elb is None or wr is None:
            continue
        if wr.visibility < min_visibility or elb.visibility < min_visibility:
            continue

        elb_x, elb_y = _metric_xy(elb)
        wr_x, wr_y = _metric_xy(wr)
        forearm_len = max(1e-3, math.hypot(wr_x - elb_x, wr_y - elb_y))

        # Hand must be clearly raised above elbow (y increases downward)
        if wr_y >= elb_y - 0.12 * forearm_len:
            continue

        finger_pts = [
            pt for pt in (idx_pt, pky_pt, thb_pt) if pt is not None and pt.visibility >= 0.4
        ]
        if len(finger_pts) >= 2:
            mean_finger_spread = sum(
                math.hypot(_metric_xy(pt)[0] - wr_x, _metric_xy(pt)[1] - wr_y)
                for pt in finger_pts
            ) / len(finger_pts)
            if mean_finger_spread <= forearm_len * 0.48:
                return True
    return False


class PoseRuleEngine:
    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        self._message_history: deque[str] = deque(
            maxlen=self.config.message_smoothing_window
        )
        self._consecutive_valid_frames: int = 0

    def reset(self) -> None:
        self._message_history.clear()
        self._consecutive_valid_frames = 0

    def _is_point_visible(self, lm: LandmarkPoint | None) -> bool:
        if lm is None:
            return False
        return (
            lm.visibility >= self.config.min_visibility
            and lm.is_within_frame(self.config.frame_margin)
        )

    def _check_front_facing(
        self,
        nose: LandmarkPoint | None,
        l_shoulder: LandmarkPoint | None,
        r_shoulder: LandmarkPoint | None,
        l_hip: LandmarkPoint | None,
        r_hip: LandmarkPoint | None,
    ) -> tuple[bool, str, str | None]:
        """Verify that the user is facing the camera directly in isotropic coordinates (FR-9)."""
        if not (
            self._is_point_visible(l_shoulder)
            and self._is_point_visible(r_shoulder)
            and l_shoulder is not None
            and r_shoulder is not None
        ):
            return False, "Shoulders unseen", "Face the camera directly"

        ls_x, ls_y = _metric_xy(l_shoulder)
        rs_x, rs_y = _metric_xy(r_shoulder)
        shoulder_span = math.hypot(ls_x - rs_x, ls_y - rs_y)
        shoulder_span_x = abs(ls_x - rs_x)
        depth_diff = abs(l_shoulder.z - r_shoulder.z)

        if l_hip is not None and r_hip is not None:
            lh_x, lh_y = _metric_xy(l_hip)
            rh_x, rh_y = _metric_xy(r_hip)
            mid_sh_x, mid_sh_y = 0.5 * (ls_x + rs_x), 0.5 * (ls_y + rs_y)
            mid_hip_x, mid_hip_y = 0.5 * (lh_x + rh_x), 0.5 * (lh_y + rh_y)
            torso_len = max(1e-3, math.hypot(mid_hip_x - mid_sh_x, mid_hip_y - mid_sh_y))
            shoulder_to_torso = shoulder_span / torso_len
            if shoulder_to_torso < self.config.min_shoulder_to_torso_ratio:
                return (
                    False,
                    f"Sideways ({shoulder_to_torso:.2f})",
                    "Face the camera directly",
                )

        if depth_diff > self.config.max_shoulder_depth_diff:
            return (
                False,
                f"Rotated (dz={depth_diff:.2f})",
                "Face the camera directly",
            )

        if nose is not None and self._is_point_visible(nose) and shoulder_span_x > 1e-4:
            nose_x, _ = _metric_xy(nose)
            mid_sh_x = 0.5 * (ls_x + rs_x)
            nose_offset_ratio = abs(nose_x - mid_sh_x) / (0.5 * shoulder_span_x)
            if nose_offset_ratio > self.config.max_nose_off_center_ratio:
                return (
                    False,
                    "Head turned",
                    "Face the camera directly",
                )

        return True, "Front-facing", None

    def evaluate(
        self, landmarks: dict[int, LandmarkPoint] | None
    ) -> PoseEvaluationResult:
        if not landmarks:
            self._consecutive_valid_frames = 0
            msg = self._smooth_message("Step into the camera view")
            return PoseEvaluationResult(
                person_detected=False,
                all_rules_passed=False,
                is_locked_ready=False,
                hold_progress=0.0,
                primary_message=msg,
            )

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
        l_ankle = landmarks.get(PoseLandmarkIndex.LEFT_ANKLE)
        r_ankle = landmarks.get(PoseLandmarkIndex.RIGHT_ANKLE)
        l_heel = landmarks.get(PoseLandmarkIndex.LEFT_HEEL)
        r_heel = landmarks.get(PoseLandmarkIndex.RIGHT_HEEL)

        bottom_candidates = [
            pt.y
            for pt in (l_heel, r_heel, l_ankle, r_ankle)
            if pt is not None
        ]
        lowest_y = max(bottom_candidates) if bottom_candidates else 1.0

        estimated_crown_y = 0.0
        if nose is not None:
            eye_y = min(
                l_eye.y if l_eye is not None else nose.y,
                r_eye.y if r_eye is not None else nose.y,
            )
            eye_to_foot = max(0.05, lowest_y - eye_y)
            crown_offset = eye_to_foot * EYE_TO_CROWN_STATURE_RATIO
            estimated_crown_y = eye_y - crown_offset

        crown_inside_top = estimated_crown_y >= (self.config.frame_margin * 0.5)
        head_visible = (
            self._is_point_visible(nose)
            and (self._is_point_visible(l_eye) or self._is_point_visible(r_eye))
            and crown_inside_top
        )
        ankles_visible = self._is_point_visible(l_ankle) and self._is_point_visible(
            r_ankle
        )
        heels_visible = self._is_point_visible(l_heel) and self._is_point_visible(
            r_heel
        )

        visibility_passed = head_visible and ankles_visible and heels_visible
        missing_parts: list[str] = []
        if not head_visible:
            missing_parts.append("Head")
        if not ankles_visible:
            missing_parts.append("Ankles")
        if not heels_visible:
            missing_parts.append("Heels")

        visibility_metric = (
            "All visible"
            if visibility_passed
            else f"Missing: {', '.join(missing_parts)}"
        )
        visibility_msg = (
            None
            if visibility_passed
            else "Take a step back so head, ankles & heels are visible"
        )

        body_fill_ratio = 0.0
        center_offset_x = 0.0
        fill_passed = False
        fill_msg: str | None = None

        if nose and bottom_candidates:
            head_top_y = max(0.0, estimated_crown_y)
            body_fill_ratio = max(0.0, lowest_y - head_top_y)

            if l_shoulder and r_shoulder:
                mid_x = 0.5 * (l_shoulder.x + r_shoulder.x)
                center_offset_x = abs(mid_x - 0.5)

            if not visibility_passed:
                fill_passed = False
                fill_msg = "Take a step back"
            elif center_offset_x > self.config.max_center_offset_x:
                fill_passed = False
                fill_msg = "Move to the center of the camera"
            elif body_fill_ratio < self.config.min_body_fill_ratio:
                fill_passed = False
                fill_msg = "Take a step closer"
            elif body_fill_ratio > self.config.max_body_fill_ratio:
                fill_passed = False
                fill_msg = "Take a step back"
            else:
                fill_passed = True

        fill_metric = f"{body_fill_ratio * 100:.0f}% (Target: 62%-92%)"

        front_facing_passed, front_facing_metric, front_facing_msg = (
            self._check_front_facing(
                nose=nose,
                l_shoulder=l_shoulder,
                r_shoulder=r_shoulder,
                l_hip=l_hip,
                r_hip=r_hip,
            )
        )

        shoulder_tilt_deg = 0.0
        shoulder_passed = False
        shoulder_msg: str | None = None

        if (
            self._is_point_visible(l_shoulder)
            and self._is_point_visible(r_shoulder)
            and l_shoulder is not None
            and r_shoulder is not None
        ):
            ls_x, ls_y = _metric_xy(l_shoulder)
            rs_x, rs_y = _metric_xy(r_shoulder)
            dx = abs(ls_x - rs_x)
            dy = abs(ls_y - rs_y)
            shoulder_tilt_deg = math.degrees(math.atan2(dy, max(dx, 1e-6)))
            shoulder_passed = shoulder_tilt_deg <= self.config.max_shoulder_tilt_deg
            if not shoulder_passed:
                shoulder_msg = "Level your shoulders horizontally"
        else:
            shoulder_msg = "Ensure both shoulders are clearly visible"

        shoulder_metric = (
            f"{shoulder_tilt_deg:.1f} deg (Max {self.config.max_shoulder_tilt_deg:.0f} deg)"
        )

        left_arm_angle_deg = 0.0
        right_arm_angle_deg = 0.0
        arm_spread_passed = False
        arm_msg: str | None = None

        if (
            l_shoulder
            and r_shoulder
            and l_hip
            and r_hip
            and (l_wrist or l_elbow)
            and (r_wrist or r_elbow)
        ):
            l_arm_target = l_wrist if self._is_point_visible(l_wrist) else l_elbow
            r_arm_target = r_wrist if self._is_point_visible(r_wrist) else r_elbow

            if l_arm_target and r_arm_target:
                ls_x, ls_y = _metric_xy(l_shoulder)
                rs_x, rs_y = _metric_xy(r_shoulder)
                lh_x, lh_y = _metric_xy(l_hip)
                rh_x, rh_y = _metric_xy(r_hip)
                lt_x, lt_y = _metric_xy(l_arm_target)
                rt_x, rt_y = _metric_xy(r_arm_target)

                left_arm_angle_deg = _calculate_angle_between_vectors(
                    lh_x - ls_x,
                    lh_y - ls_y,
                    lt_x - ls_x,
                    lt_y - ls_y,
                )
                right_arm_angle_deg = _calculate_angle_between_vectors(
                    rh_x - rs_x,
                    rh_y - rs_y,
                    rt_x - rs_x,
                    rt_y - rs_y,
                )

                min_angle = min(left_arm_angle_deg, right_arm_angle_deg)
                max_angle = max(left_arm_angle_deg, right_arm_angle_deg)

                if min_angle < self.config.min_arm_spread_deg:
                    arm_spread_passed = False
                    arm_msg = "Spread your arms a little"
                elif max_angle > self.config.max_arm_spread_deg:
                    arm_spread_passed = False
                    arm_msg = "Lower your arms slightly"
                else:
                    arm_spread_passed = True
            else:
                arm_msg = "Keep both arms visible in frame"
        else:
            arm_msg = "Keep both arms visible in frame"

        arm_metric = f"L:{left_arm_angle_deg:.0f} deg / R:{right_arm_angle_deg:.0f} deg"

        rules = [
            RuleCheckItem(
                key="visibility",
                label="Head, Ankle & Heel Visibility",
                passed=visibility_passed,
                metric_text=visibility_metric,
                guidance_message=visibility_msg,
            ),
            RuleCheckItem(
                key="body_fill",
                label="Body Frame Fill",
                passed=fill_passed,
                metric_text=fill_metric,
                guidance_message=fill_msg,
            ),
            RuleCheckItem(
                key="front_facing",
                label="Facing the Camera",
                passed=front_facing_passed,
                metric_text=front_facing_metric,
                guidance_message=front_facing_msg,
            ),
            RuleCheckItem(
                key="arm_spread",
                label="Arm Spread (A-Pose)",
                passed=arm_spread_passed,
                metric_text=arm_metric,
                guidance_message=arm_msg,
            ),
            RuleCheckItem(
                key="shoulders",
                label="Horizontal Shoulder Line",
                passed=shoulder_passed,
                metric_text=shoulder_metric,
                guidance_message=shoulder_msg,
            ),
        ]

        all_passed = all(r.passed for r in rules)
        fist_detected = detect_fist_or_hand_trigger(landmarks)

        if all_passed:
            self._consecutive_valid_frames += 1
        else:
            self._consecutive_valid_frames = 0

        hold_progress = min(
            1.0,
            self._consecutive_valid_frames / max(1, self.config.ready_hold_frames),
        )
        is_locked_ready = hold_progress >= 1.0

        if is_locked_ready:
            smoothed_message = "POSE VERIFIED - READY FOR CAPTURE"
        elif all_passed:
            countdown_num = max(1, math.ceil((1.0 - hold_progress) * 3.0))
            smoothed_message = f"Perfect pose! Hold still... {countdown_num}"
        else:
            raw_primary_message = next(
                (r.guidance_message for r in rules if not r.passed and r.guidance_message),
                "Adjust your pose",
            )
            smoothed_message = self._smooth_message(raw_primary_message)

        return PoseEvaluationResult(
            person_detected=True,
            all_rules_passed=all_passed,
            is_locked_ready=is_locked_ready,
            hold_progress=hold_progress,
            primary_message=smoothed_message,
            rules=rules,
            head_visible=head_visible,
            ankles_visible=ankles_visible,
            heels_visible=heels_visible,
            front_facing=front_facing_passed,
            body_fill_ratio=body_fill_ratio,
            shoulder_tilt_deg=shoulder_tilt_deg,
            left_arm_angle_deg=left_arm_angle_deg,
            right_arm_angle_deg=right_arm_angle_deg,
            fist_gesture_detected=fist_detected,
        )

    def _smooth_message(self, new_message: str) -> str:
        self._message_history.append(new_message)
        counts = Counter(self._message_history)
        most_common_msg, _ = counts.most_common(1)[0]
        return most_common_msg

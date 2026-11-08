from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from body.landmarks import LandmarkPoint, PoseLandmarkIndex


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
    body_fill_ratio: float = 0.0
    shoulder_tilt_deg: float = 0.0
    left_arm_angle_deg: float = 0.0
    right_arm_angle_deg: float = 0.0


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


class PoseRuleEngine:
    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        self._message_history: deque[str] = deque(
            maxlen=self.config.message_smoothing_window
        )
        self._consecutive_valid_frames: int = 0

    def _is_point_visible(self, lm: LandmarkPoint | None) -> bool:
        if lm is None:
            return False
        return (
            lm.visibility >= self.config.min_visibility
            and lm.is_within_frame(self.config.frame_margin)
        )

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

        head_visible = self._is_point_visible(nose) and (
            self._is_point_visible(l_eye) or self._is_point_visible(r_eye)
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

        if nose and (l_heel or r_heel or l_ankle or r_ankle):
            eye_y = min(
                l_eye.y if l_eye else nose.y,
                r_eye.y if r_eye else nose.y,
            )
            head_top_y = max(0.0, eye_y - 0.04)

            bottom_candidates = [
                pt.y
                for pt in (l_heel, r_heel, l_ankle, r_ankle)
                if pt is not None
            ]
            lowest_y = max(bottom_candidates) if bottom_candidates else 1.0
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

        shoulder_tilt_deg = 0.0
        shoulder_passed = False
        shoulder_msg: str | None = None

        if (
            self._is_point_visible(l_shoulder)
            and self._is_point_visible(r_shoulder)
            and l_shoulder is not None
            and r_shoulder is not None
        ):
            dx = abs(l_shoulder.x - r_shoulder.x)
            dy = abs(l_shoulder.y - r_shoulder.y)
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
                left_arm_angle_deg = _calculate_angle_between_vectors(
                    l_hip.x - l_shoulder.x,
                    l_hip.y - l_shoulder.y,
                    l_arm_target.x - l_shoulder.x,
                    l_arm_target.y - l_shoulder.y,
                )
                right_arm_angle_deg = _calculate_angle_between_vectors(
                    r_hip.x - r_shoulder.x,
                    r_hip.y - r_shoulder.y,
                    r_arm_target.x - r_shoulder.x,
                    r_arm_target.y - r_shoulder.y,
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
            body_fill_ratio=body_fill_ratio,
            shoulder_tilt_deg=shoulder_tilt_deg,
            left_arm_angle_deg=left_arm_angle_deg,
            right_arm_angle_deg=right_arm_angle_deg,
        )

    def _smooth_message(self, new_message: str) -> str:
        self._message_history.append(new_message)
        counts = Counter(self._message_history)
        most_common_msg, _ = counts.most_common(1)[0]
        return most_common_msg

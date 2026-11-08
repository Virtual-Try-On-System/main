from __future__ import annotations

import cv2
import numpy as np

from body.anthropometry import AnthropometricMeasurements
from body.pose_rules import PoseEvaluationResult


class UIOverlay:
    def __init__(
        self,
        show_checklist: bool = True,
        show_frame_guides: bool = True,
        auto_capture_enabled: bool = True,
    ) -> None:
        self.show_checklist = show_checklist
        self.show_frame_guides = show_frame_guides
        self.auto_capture_enabled = auto_capture_enabled

    @staticmethod
    def _draw_translucent_box(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color_bgr: tuple[int, int, int] = (20, 20, 25),
        alpha: float = 0.68,
        border_color: tuple[int, int, int] | None = None,
        border_thickness: int = 1,
    ) -> None:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, color_bgr, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

        if border_color is not None:
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA
            )

    def render(
        self,
        frame: np.ndarray,
        fps: float,
        evaluation: PoseEvaluationResult,
        measurements: AnthropometricMeasurements | None = None,
        capture_toast_text: str | None = None,
    ) -> np.ndarray:
        if self.show_frame_guides:
            self._draw_target_guides(frame, evaluation)

        if measurements is not None:
            self._draw_silhouette_width_lines(frame, measurements)

        self._draw_fps_badge(frame, fps)
        self._draw_guidance_banner(frame, evaluation)

        if self.show_checklist and evaluation.rules:
            self._draw_rule_checklist(frame, evaluation)

        if measurements is not None:
            self._draw_measurements_panel(frame, measurements, evaluation.all_rules_passed)

        if capture_toast_text:
            self._draw_capture_toast(frame, capture_toast_text)

        self._draw_footer_help(frame)
        return frame

    def _draw_silhouette_width_lines(
        self, frame: np.ndarray, m: AnthropometricMeasurements
    ) -> None:
        spans = [
            ("Chest", m.chest_span_pts, m.chest_width_cm, (255, 220, 80)),
            ("Waist", m.waist_span_pts, m.waist_width_cm, (80, 240, 255)),
            ("Hip", m.hip_span_pts, m.hip_silhouette_cm, (255, 140, 255)),
        ]
        for label, pts, cm_val, color in spans:
            if pts is None:
                continue
            lx, rx, y = pts
            cv2.line(frame, (lx, y), (rx, y), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (lx, y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (rx, y), 4, color, -1, cv2.LINE_AA)
            if cm_val is not None:
                cv2.putText(
                    frame,
                    f"{label}: {cm_val:.1f}cm",
                    (rx + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_target_guides(
        self, frame: np.ndarray, evaluation: PoseEvaluationResult
    ) -> None:
        h, w = frame.shape[:2]
        margin_x = int(w * 0.18)
        top_y = int(h * 0.05)
        bottom_y = int(h * 0.95)
        corner_len = 32

        if evaluation.is_locked_ready:
            color = (60, 230, 80)
        elif evaluation.all_rules_passed:
            color = (0, 220, 255)
        else:
            color = (180, 180, 190)

        t = 2
        cv2.line(frame, (margin_x, top_y), (margin_x + corner_len, top_y), color, t, cv2.LINE_AA)
        cv2.line(frame, (margin_x, top_y), (margin_x, top_y + corner_len), color, t, cv2.LINE_AA)
        cv2.line(
            frame,
            (w - margin_x, top_y),
            (w - margin_x - corner_len, top_y),
            color,
            t,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (w - margin_x, top_y),
            (w - margin_x, top_y + corner_len),
            color,
            t,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (margin_x, bottom_y),
            (margin_x + corner_len, bottom_y),
            color,
            t,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (margin_x, bottom_y),
            (margin_x, bottom_y - corner_len),
            color,
            t,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (w - margin_x, bottom_y),
            (w - margin_x - corner_len, bottom_y),
            color,
            t,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (w - margin_x, bottom_y),
            (w - margin_x, bottom_y - corner_len),
            color,
            t,
            cv2.LINE_AA,
        )

    def _draw_fps_badge(self, frame: np.ndarray, fps: float) -> None:
        self._draw_translucent_box(
            frame,
            16,
            16,
            175,
            62,
            color_bgr=(20, 24, 30),
            alpha=0.75,
            border_color=(80, 200, 120),
            border_thickness=1,
        )
        fps_text = f"FPS: {fps:4.1f}"
        cv2.putText(
            frame,
            fps_text,
            (28, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (100, 255, 140),
            2,
            cv2.LINE_AA,
        )

    def _draw_guidance_banner(
        self, frame: np.ndarray, evaluation: PoseEvaluationResult
    ) -> None:
        _, w = frame.shape[:2]
        message = evaluation.primary_message

        if evaluation.is_locked_ready:
            border_color = (50, 230, 90)
            text_color = (120, 255, 150)
            bg_color = (15, 45, 20)
        elif evaluation.all_rules_passed:
            border_color = (0, 215, 255)
            text_color = (100, 240, 255)
            bg_color = (20, 35, 45)
        else:
            border_color = (50, 120, 255)
            text_color = (255, 255, 255)
            bg_color = (25, 25, 40)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.80
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(message, font, font_scale, thickness)

        box_w = max(440, text_w + 56)
        box_h = 58
        x1 = (w - box_w) // 2
        y1 = 16
        x2 = x1 + box_w
        y2 = y1 + box_h

        self._draw_translucent_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            color_bgr=bg_color,
            alpha=0.82,
            border_color=border_color,
            border_thickness=2,
        )

        text_x = x1 + (box_w - text_w) // 2
        text_y = y1 + (box_h + text_h) // 2 - 4
        cv2.putText(
            frame,
            message,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        if evaluation.hold_progress > 0.0:
            bar_y1 = y2 - 6
            bar_w = int((box_w - 4) * evaluation.hold_progress)
            cv2.rectangle(
                frame,
                (x1 + 2, bar_y1),
                (x1 + 2 + bar_w, y2 - 2),
                (60, 240, 100),
                -1,
            )

    def _draw_rule_checklist(
        self, frame: np.ndarray, evaluation: PoseEvaluationResult
    ) -> None:
        h, _ = frame.shape[:2]
        panel_w = 390
        row_h = 30
        panel_h = 40 + len(evaluation.rules) * row_h
        x1 = 16
        y2 = h - 42
        y1 = y2 - panel_h
        x2 = x1 + panel_w

        self._draw_translucent_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            color_bgr=(18, 20, 26),
            alpha=0.76,
            border_color=(90, 95, 110),
            border_thickness=1,
        )

        cv2.putText(
            frame,
            "POSE VERIFICATION RULES",
            (x1 + 14, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (190, 200, 220),
            1,
            cv2.LINE_AA,
        )

        for idx, rule in enumerate(evaluation.rules):
            row_y = y1 + 52 + idx * row_h
            status_tag = "[OK]" if rule.passed else "[X] "
            status_color = (70, 235, 100) if rule.passed else (70, 110, 255)

            cv2.putText(
                frame,
                status_tag,
                (x1 + 14, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{rule.label}: {rule.metric_text}",
                (x1 + 60, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (235, 235, 240),
                1,
                cv2.LINE_AA,
            )

    def _draw_measurements_panel(
        self,
        frame: np.ndarray,
        m: AnthropometricMeasurements,
        is_valid: bool,
    ) -> None:
        h, w = frame.shape[:2]
        lines = [
            f"Ref Height: {m.reference_height_cm or 175.0:.0f} cm ([+/-] adjust)",
            f"Shoulder:   {m.shoulder_width_cm or 0:.1f} cm ({m.shoulder_width_px:.0f} px)",
            f"Torso/Leg:  {m.torso_length_cm or 0:.1f} / {m.leg_length_cm or 0:.1f} cm",
        ]
        if m.chest_width_cm is not None and m.waist_width_cm is not None:
            lines.append(
                f"Chest/Waist:{m.chest_width_cm:5.1f} / {m.waist_width_cm:.1f} cm"
            )
        if m.hip_silhouette_cm is not None:
            lines.append(f"Hip Width:  {m.hip_silhouette_cm:.1f} cm")
        else:
            lines.append(f"Hip Joint:  {m.hip_width_cm or 0:.1f} cm")

        upper_sz = m.estimated_upper_size or "-"
        lower_sz = m.estimated_lower_size or "-"
        lines.append(f"Est. Size:  Upper [{upper_sz}] | Lower [{lower_sz}]")

        panel_w = 325
        row_h = 25
        panel_h = 38 + len(lines) * row_h
        x2 = w - 16
        x1 = x2 - panel_w
        y2 = h - 42
        y1 = y2 - panel_h

        border_col = (70, 225, 110) if is_valid else (100, 140, 180)
        self._draw_translucent_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            color_bgr=(18, 24, 22),
            alpha=0.78,
            border_color=border_col,
            border_thickness=1,
        )

        cv2.putText(
            frame,
            "ANTHROPOMETRIC & SIZE DATA",
            (x1 + 14, y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (120, 245, 150) if is_valid else (180, 210, 240),
            1,
            cv2.LINE_AA,
        )

        for idx, text in enumerate(lines):
            row_y = y1 + 48 + idx * row_h
            is_size_row = idx == len(lines) - 1
            color = (100, 255, 160) if is_size_row else (235, 240, 235)
            thickness = 2 if is_size_row else 1
            cv2.putText(
                frame,
                text,
                (x1 + 14, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_capture_toast(self, frame: np.ndarray, text: str) -> None:
        _, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.65, 2)
        box_w = tw + 40
        x1 = (w - box_w) // 2
        y1 = 84
        x2 = x1 + box_w
        y2 = y1 + 40

        self._draw_translucent_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            color_bgr=(20, 75, 30),
            alpha=0.88,
            border_color=(80, 255, 120),
            border_thickness=2,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 20, y1 + th + 10),
            font,
            0.65,
            (220, 255, 225),
            2,
            cv2.LINE_AA,
        )

    def _draw_footer_help(self, frame: np.ndarray) -> None:
        h, _ = frame.shape[:2]
        auto_state = "ON" if self.auto_capture_enabled else "OFF"
        help_text = (
            f"[Q] Quit  [S] Snapshot  [A] Auto-Capture:{auto_state}  "
            "[M] Mask  [+/-] Height  [D] Rules"
        )
        cv2.putText(
            frame,
            help_text,
            (18, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 185, 195),
            1,
            cv2.LINE_AA,
        )

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from body.anthropometry import AnthropometricExtractor, AnthropometricMeasurements
from body.camera import CameraStream, FPSCounter
from body.landmarks import LandmarkPoint
from body.pose_detector import PoseDetector
from body.pose_rules import PoseEvaluationResult, PoseRuleEngine, RuleConfig
from body.ui_overlay import UIOverlay


def save_capture_snapshot(
    frame_bgr: np.ndarray,
    landmarks: dict[int, LandmarkPoint] | None,
    evaluation: PoseEvaluationResult,
    measurements: AnthropometricMeasurements | None,
    segmentation_mask: np.ndarray | None,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = output_dir / f"capture_{timestamp}.jpg"
    json_path = output_dir / f"capture_{timestamp}_landmarks.json"

    cv2.imwrite(str(img_path), frame_bgr)

    if segmentation_mask is not None:
        mask_path = output_dir / f"capture_{timestamp}_mask.png"
        cv2.imwrite(str(mask_path), segmentation_mask)

    body_width_cm = None
    if measurements is not None:
        body_width_cm = (
            measurements.waist_width_cm
            or measurements.chest_width_cm
            or measurements.hip_width_cm
        )

    body_schema_payload = None
    if measurements is not None and landmarks:
        body_schema_payload = {
            "schema_version": "1.0",
            "height_cm": round(measurements.reference_height_cm or 175.0, 2),
            "weight_kg": 70.0,
            "cm_per_pixel": round(measurements.cm_per_pixel or 0.0, 5),
            "shoulder_width_cm": round(measurements.shoulder_width_cm or 0.0, 2),
            "torso_length_cm": round(measurements.torso_length_cm or 0.0, 2),
            "body_width_cm": round(body_width_cm or 0.0, 2),
            "pose_keypoints": [
                {
                    "name": lm.name.lower(),
                    "x_px": float(max(0, lm.px)),
                    "y_px": float(max(0, lm.py)),
                    "confidence": round(min(1.0, max(0.0, lm.visibility)), 4),
                }
                for lm in landmarks.values()
            ],
        }

    payload = {
        "timestamp": timestamp,
        "all_rules_passed": evaluation.all_rules_passed,
        "metrics": {
            "head_visible": evaluation.head_visible,
            "ankles_visible": evaluation.ankles_visible,
            "heels_visible": evaluation.heels_visible,
            "body_fill_ratio": round(evaluation.body_fill_ratio, 4),
            "shoulder_tilt_deg": round(evaluation.shoulder_tilt_deg, 2),
            "left_arm_angle_deg": round(evaluation.left_arm_angle_deg, 2),
            "right_arm_angle_deg": round(evaluation.right_arm_angle_deg, 2),
        },
        "anthropometry": measurements.to_dict() if measurements else None,
        "body_schema": body_schema_payload,
        "landmarks": {
            idx: {
                "name": lm.name,
                "x": round(lm.x, 5),
                "y": round(lm.y, 5),
                "z": round(lm.z, 5),
                "visibility": round(lm.visibility, 4),
                "px": lm.px,
                "py": lm.py,
            }
            for idx, lm in (landmarks or {}).items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return img_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Virtual Try-On Pose Verification System"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height (default: 720)",
    )
    parser.add_argument(
        "--user-height",
        type=float,
        default=175.0,
        help="User reference height in cm (default: 175.0)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror flipping of camera feed",
    )
    parser.add_argument(
        "--no-auto-capture",
        action="store_true",
        help="Disable automatic hands-free capture when pose is verified",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera = CameraStream(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
    )
    if not camera.start():
        print(
            f"[ERROR] Could not open camera with index {args.camera}. "
            "Please check your webcam connection."
        )
        return 1

    fps_counter = FPSCounter(window_size=30)
    detector = PoseDetector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
        enable_segmentation=True,
    )
    rule_engine = PoseRuleEngine(config=RuleConfig())
    extractor = AnthropometricExtractor(reference_height_cm=args.user_height)
    ui = UIOverlay(
        show_checklist=True,
        show_frame_guides=True,
        auto_capture_enabled=not args.no_auto_capture,
    )

    window_title = "Virtual Try-On Pose Verification"
    outputs_dir = Path(__file__).resolve().parent / "outputs"

    show_mask_contour = True
    has_auto_captured_current_lock = False
    last_capture_time = 0.0
    toast_until = 0.0
    toast_text: str | None = None

    print("[INFO] Camera started. Press 'Q' or 'ESC' to exit, 'S' to save snapshot.")

    try:
        while True:
            ok, raw_frame = camera.read()
            if not ok or raw_frame is None:
                print("[WARNING] Failed to grab frame from camera.")
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            fps = fps_counter.tick()
            display_frame = raw_frame.copy()

            landmarks = detector.detect(raw_frame)
            evaluation = rule_engine.evaluate(landmarks)
            measurements = extractor.extract(
                landmarks,
                raw_frame.shape[0],
                detector.last_segmentation_mask,
            )

            if not evaluation.is_locked_ready:
                has_auto_captured_current_lock = False

            if (
                ui.auto_capture_enabled
                and evaluation.is_locked_ready
                and not has_auto_captured_current_lock
                and (now - last_capture_time) >= 4.0
            ):
                saved_path = save_capture_snapshot(
                    raw_frame,
                    landmarks,
                    evaluation,
                    measurements,
                    detector.last_segmentation_mask,
                    outputs_dir,
                )
                has_auto_captured_current_lock = True
                last_capture_time = now
                toast_text = f"AUTO-CAPTURED: {saved_path.name}"
                toast_until = now + 2.5
                print(f"[INFO] Auto-captured snapshot and data to: {saved_path}")

            if show_mask_contour:
                detector.draw_silhouette_contour(
                    display_frame,
                    is_pose_valid=evaluation.all_rules_passed,
                )

            detector.draw_skeleton(
                display_frame,
                landmarks,
                is_pose_valid=evaluation.all_rules_passed,
            )

            active_toast = toast_text if now < toast_until else None
            ui.render(
                display_frame,
                fps=fps,
                evaluation=evaluation,
                measurements=measurements,
                capture_toast_text=active_toast,
            )

            cv2.imshow(window_title, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("d"), ord("D")):
                ui.show_checklist = not ui.show_checklist
            elif key in (ord("a"), ord("A")):
                ui.auto_capture_enabled = not ui.auto_capture_enabled
            elif key in (ord("m"), ord("M")):
                show_mask_contour = not show_mask_contour
            elif key in (ord("+"), ord("=")):
                extractor.reference_height_cm = min(
                    230.0, (extractor.reference_height_cm or 175.0) + 1.0
                )
            elif key in (ord("-"), ord("_")):
                extractor.reference_height_cm = max(
                    120.0, (extractor.reference_height_cm or 175.0) - 1.0
                )
            elif key in (ord("s"), ord("S")):
                saved_path = save_capture_snapshot(
                    raw_frame,
                    landmarks,
                    evaluation,
                    measurements,
                    detector.last_segmentation_mask,
                    outputs_dir,
                )
                last_capture_time = now
                toast_text = f"SAVED: {saved_path.name}"
                toast_until = now + 2.5
                print(f"[INFO] Saved snapshot and landmarks to: {saved_path}")

    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

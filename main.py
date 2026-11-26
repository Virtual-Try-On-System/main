from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from body.anthropometry import (
    MAX_HEIGHT_CM,
    MIN_HEIGHT_CM,
    AnthropometricMeasurements,
    validate_user_anthropometrics,
)
from body.camera import CameraStream, FPSCounter
from body.landmarks import LandmarkPoint
from body.pipeline import BodyAnalysisPipeline
from body.pose_detector import PoseDetector
from body.pose_rules import PoseEvaluationResult, RuleConfig
from body.ui_overlay import UIOverlay
from schemas.body import Body


def save_capture_snapshot(
    frame_bgr: np.ndarray,
    landmarks: dict[int, LandmarkPoint] | None,
    evaluation: PoseEvaluationResult,
    measurements: AnthropometricMeasurements | None,
    segmentation_mask: np.ndarray | None,
    output_dir: Path,
    user_weight_kg: float = 70.0,
    body_schema: Body | None = None,
    metadata_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = output_dir / f"capture_{timestamp}.jpg"
    json_path = output_dir / f"capture_{timestamp}_landmarks.json"

    if not metadata_only:
        cv2.imwrite(str(img_path), frame_bgr)
        if segmentation_mask is not None:
            mask_path = output_dir / f"capture_{timestamp}_mask.png"
            cv2.imwrite(str(mask_path), segmentation_mask)

    body_schema_payload = None
    if body_schema is not None:
        body_schema_payload = body_schema.model_dump()
    elif measurements is not None and landmarks:
        body_model = measurements.to_body_schema(
            landmarks,
            weight_kg=user_weight_kg,
        )
        body_schema_payload = body_model.model_dump()

    payload = {
        "timestamp": timestamp,
        "all_rules_passed": evaluation.all_rules_passed,
        "metrics": {
            "head_visible": evaluation.head_visible,
            "ankles_visible": evaluation.ankles_visible,
            "heels_visible": evaluation.heels_visible,
            "full_body_visible": evaluation.full_body_visible,
            "front_facing": evaluation.front_facing,
            "fist_gesture_detected": evaluation.fist_gesture_detected,
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
    return json_path if metadata_only else img_path


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
        help="User reference height in cm, 140-210 (default: 175.0)",
    )
    parser.add_argument(
        "--user-weight",
        type=float,
        default=70.0,
        help="User weight in kg, 40-150 (default: 70.0)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Analyze a static full-body photo instead of opening a live webcam (FR-2)",
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
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Save only JSON body descriptor without writing user photos to disk (NFR-7)",
    )
    args = parser.parse_args()
    try:
        validate_user_anthropometrics(args.user_height, args.user_weight)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _run_static_image_mode(args: argparse.Namespace) -> int:
    image_path: Path = args.image
    if not image_path.is_file():
        print(f"[ERROR] Image file not found: {image_path}")
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not decode image file: {image_path}")
        return 1

    detector = PoseDetector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1,
        enable_segmentation=True,
        static_image_mode=True,
    )
    pipeline = BodyAnalysisPipeline(
        height_cm=args.user_height,
        weight_kg=args.user_weight,
        rule_config=RuleConfig(message_smoothing_window=1),
        stabilize_scale=False,
    )
    outputs_dir = Path(__file__).resolve().parent / "outputs"

    try:
        landmarks = detector.detect(frame)
        analysis = pipeline.analyze_landmarks(
            landmarks,
            frame_height=frame.shape[0],
            segmentation_mask=detector.last_segmentation_mask,
        )
        saved_path = save_capture_snapshot(
            frame,
            landmarks,
            analysis.evaluation,
            analysis.measurements,
            detector.last_segmentation_mask,
            outputs_dir,
            user_weight_kg=pipeline.weight_kg,
            body_schema=analysis.body_schema,
            metadata_only=args.metadata_only,
        )
        print(f"[INFO] Static photo analyzed and saved to: {saved_path}")
        return 0
    finally:
        detector.close()


def main() -> int:
    args = parse_args()
    if args.image is not None:
        return _run_static_image_mode(args)

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
    pipeline = BodyAnalysisPipeline(
        height_cm=args.user_height,
        weight_kg=args.user_weight,
        rule_config=RuleConfig(),
        stabilize_scale=True,
    )
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
            analysis = pipeline.analyze_landmarks(
                landmarks,
                frame_height=raw_frame.shape[0],
                segmentation_mask=detector.last_segmentation_mask,
            )
            evaluation = analysis.evaluation
            measurements = analysis.measurements

            trigger_ready = evaluation.is_locked_ready or (
                evaluation.all_rules_passed and evaluation.fist_gesture_detected
            )
            if not trigger_ready:
                has_auto_captured_current_lock = False

            if (
                ui.auto_capture_enabled
                and trigger_ready
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
                    user_weight_kg=pipeline.weight_kg,
                    body_schema=analysis.body_schema,
                    metadata_only=args.metadata_only,
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
                new_h = min(MAX_HEIGHT_CM, pipeline.height_cm + 1.0)
                pipeline.update_user_anthropometrics(new_h, pipeline.weight_kg)
            elif key in (ord("-"), ord("_")):
                new_h = max(MIN_HEIGHT_CM, pipeline.height_cm - 1.0)
                pipeline.update_user_anthropometrics(new_h, pipeline.weight_kg)
            elif key in (ord("s"), ord("S")):
                saved_path = save_capture_snapshot(
                    raw_frame,
                    landmarks,
                    evaluation,
                    measurements,
                    detector.last_segmentation_mask,
                    outputs_dir,
                    user_weight_kg=pipeline.weight_kg,
                    body_schema=analysis.body_schema,
                    metadata_only=args.metadata_only,
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

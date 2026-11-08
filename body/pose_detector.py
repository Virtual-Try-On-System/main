from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from body.landmarks import (
    CRITICAL_LANDMARK_INDICES,
    POSE_CONNECTIONS,
    LandmarkPoint,
    PoseLandmarkIndex,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


class PoseDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
        enable_segmentation: bool = True,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation

        self._use_solutions_api = False
        self._pose_solutions = None
        self._pose_landmarker = None
        self._mp = None
        self._last_timestamp_ms: int = 0
        self.last_segmentation_mask: np.ndarray | None = None

        self._init_backend()

    def _init_backend(self) -> None:
        import mediapipe as mp

        self._mp = mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._use_solutions_api = True
            self._pose_solutions = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=self.enable_segmentation,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = self._ensure_task_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=self.enable_segmentation,
            )
            self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def _ensure_task_model(self) -> Path:
        checkpoints_dir = Path(__file__).resolve().parent.parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoints_dir / "pose_landmarker_lite.task"
        if not model_path.exists():
            print(f"[INFO] Downloading MediaPipe Pose model to {model_path}...")
            urllib.request.urlretrieve(MODEL_URL, str(model_path))
        return model_path

    def detect(self, frame_bgr: np.ndarray) -> dict[int, LandmarkPoint] | None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        raw_landmarks = None
        self.last_segmentation_mask = None

        if self._use_solutions_api and self._pose_solutions is not None:
            results = self._pose_solutions.process(frame_rgb)
            if results.pose_landmarks:
                raw_landmarks = results.pose_landmarks.landmark
            if results.segmentation_mask is not None:
                self.last_segmentation_mask = (
                    (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                )
        elif self._pose_landmarker is not None:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=frame_rgb,
            )
            now_ms = int(time.perf_counter() * 1000)
            self._last_timestamp_ms = max(self._last_timestamp_ms + 1, now_ms)
            result = self._pose_landmarker.detect_for_video(
                mp_image, self._last_timestamp_ms
            )
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                raw_landmarks = result.pose_landmarks[0]
            if result.segmentation_masks and len(result.segmentation_masks) > 0:
                mask_view = result.segmentation_masks[0].numpy_view()
                if mask_view.ndim == 3:
                    mask_view = mask_view[:, :, 0]
                self.last_segmentation_mask = (
                    (mask_view > 0.5).astype(np.uint8) * 255
                )

        if not raw_landmarks:
            return None

        landmarks: dict[int, LandmarkPoint] = {}
        for idx, lm in enumerate(raw_landmarks):
            try:
                name = PoseLandmarkIndex(idx).name
            except ValueError:
                name = f"LANDMARK_{idx}"

            visibility = float(getattr(lm, "visibility", 1.0) or 0.0)
            px = int(round(lm.x * w))
            py = int(round(lm.y * h))
            landmarks[idx] = LandmarkPoint(
                index=idx,
                name=name,
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=visibility,
                px=px,
                py=py,
            )

        return landmarks

    def draw_silhouette_contour(
        self,
        frame_bgr: np.ndarray,
        is_pose_valid: bool = False,
    ) -> np.ndarray:
        if self.last_segmentation_mask is None:
            return frame_bgr

        contours, _ = cv2.findContours(
            self.last_segmentation_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return frame_bgr

        contour_color = (120, 255, 140) if is_pose_valid else (255, 190, 80)
        cv2.drawContours(frame_bgr, contours, -1, contour_color, 1, cv2.LINE_AA)
        return frame_bgr

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        landmarks: dict[int, LandmarkPoint] | None,
        is_pose_valid: bool = False,
        min_visibility: float = 0.5,
    ) -> np.ndarray:
        if not landmarks:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        bone_color = (80, 230, 100) if is_pose_valid else (255, 200, 60)
        shoulder_color = (0, 255, 255)

        for idx_a, idx_b in POSE_CONNECTIONS:
            lm_a = landmarks.get(idx_a)
            lm_b = landmarks.get(idx_b)
            if lm_a is None or lm_b is None:
                continue
            if lm_a.visibility < min_visibility or lm_b.visibility < min_visibility:
                continue

            color = shoulder_color if (idx_a, idx_b) == (11, 12) else bone_color
            thickness = 3 if (idx_a, idx_b) == (11, 12) else 2
            cv2.line(
                frame_bgr,
                (lm_a.px, lm_a.py),
                (lm_b.px, lm_b.py),
                color,
                thickness,
                cv2.LINE_AA,
            )

        for idx, lm in landmarks.items():
            if lm.visibility < min_visibility:
                continue
            if not (0 <= lm.px < w and 0 <= lm.py < h):
                continue

            if idx in CRITICAL_LANDMARK_INDICES:
                joint_color = (40, 240, 80) if lm.visibility >= 0.65 else (0, 140, 255)
                cv2.circle(frame_bgr, (lm.px, lm.py), 6, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (lm.px, lm.py), 4, joint_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, (lm.px, lm.py), 3, (220, 220, 220), -1, cv2.LINE_AA)

        return frame_bgr

    def close(self) -> None:
        if self._pose_solutions is not None:
            self._pose_solutions.close()
            self._pose_solutions = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None

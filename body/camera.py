from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = max(1, window_size)
        self._timestamps: deque[float] = deque(maxlen=self.window_size)
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class CameraStream:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.mirror = mirror
        self.cap: cv2.VideoCapture | None = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

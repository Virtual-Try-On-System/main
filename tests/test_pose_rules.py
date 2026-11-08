from __future__ import annotations

from body.anthropometry import AnthropometricExtractor
from body.landmarks import LandmarkPoint, PoseLandmarkIndex
from body.pose_rules import PoseRuleEngine, RuleConfig


class _DummyMask:
    def __init__(self, height: int, width: int, y1: int, y2: int, x1: int, x2: int) -> None:
        self.shape = (height, width)
        self._y1 = y1
        self._y2 = y2
        self._x1 = x1
        self._x2 = x2

    def __getitem__(self, item: tuple[int, slice]) -> list[int]:
        row_y, col_slice = item
        start = col_slice.start or 0
        stop = col_slice.stop or self.shape[1]
        if not (self._y1 <= row_y < self._y2):
            return [0] * (stop - start)
        return [
            255 if (self._x1 <= x < self._x2) else 0
            for x in range(start, stop)
        ]


def _make_valid_pose_landmarks() -> dict[int, LandmarkPoint]:
    coords = {
        PoseLandmarkIndex.NOSE: (0.50, 0.14),
        PoseLandmarkIndex.LEFT_EYE: (0.48, 0.12),
        PoseLandmarkIndex.RIGHT_EYE: (0.52, 0.12),
        PoseLandmarkIndex.LEFT_SHOULDER: (0.42, 0.26),
        PoseLandmarkIndex.RIGHT_SHOULDER: (0.58, 0.26),
        PoseLandmarkIndex.LEFT_ELBOW: (0.35, 0.40),
        PoseLandmarkIndex.RIGHT_ELBOW: (0.65, 0.40),
        PoseLandmarkIndex.LEFT_WRIST: (0.30, 0.52),
        PoseLandmarkIndex.RIGHT_WRIST: (0.70, 0.52),
        PoseLandmarkIndex.LEFT_HIP: (0.45, 0.54),
        PoseLandmarkIndex.RIGHT_HIP: (0.55, 0.54),
        PoseLandmarkIndex.LEFT_ANKLE: (0.45, 0.86),
        PoseLandmarkIndex.RIGHT_ANKLE: (0.55, 0.86),
        PoseLandmarkIndex.LEFT_HEEL: (0.45, 0.89),
        PoseLandmarkIndex.RIGHT_HEEL: (0.55, 0.89),
    }

    landmarks: dict[int, LandmarkPoint] = {}
    for idx, (x, y) in coords.items():
        landmarks[int(idx)] = LandmarkPoint(
            index=int(idx),
            name=idx.name,
            x=x,
            y=y,
            z=0.0,
            visibility=0.95,
            px=int(x * 1280),
            py=int(y * 720),
        )
    return landmarks


def test_valid_pose_passes_all_rules() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    result = engine.evaluate(landmarks)

    assert result.person_detected is True
    assert result.head_visible is True
    assert result.ankles_visible is True
    assert result.heels_visible is True
    assert result.all_rules_passed is True
    assert "Perfect pose" in result.primary_message


def test_missing_heels_triggers_step_back() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_HEEL].y = 1.03
    landmarks[PoseLandmarkIndex.RIGHT_HEEL].y = 1.03

    result = engine.evaluate(landmarks)
    assert result.heels_visible is False
    assert result.all_rules_passed is False
    assert "step back" in result.primary_message.lower()


def test_arms_too_close_triggers_spread_arms_message() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.LEFT_WRIST].x = 0.44
    landmarks[PoseLandmarkIndex.LEFT_WRIST].y = 0.54
    landmarks[PoseLandmarkIndex.RIGHT_WRIST].x = 0.56
    landmarks[PoseLandmarkIndex.RIGHT_WRIST].y = 0.54

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.primary_message == "Spread your arms a little"


def test_tilted_shoulders_triggers_shoulder_message() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.RIGHT_SHOULDER].y = 0.31

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.shoulder_tilt_deg > 5.0
    assert "shoulders" in result.primary_message.lower()


def test_user_too_far_triggers_step_closer() -> None:
    engine = PoseRuleEngine(RuleConfig(message_smoothing_window=1))
    landmarks = _make_valid_pose_landmarks()
    landmarks[PoseLandmarkIndex.NOSE].y = 0.32
    landmarks[PoseLandmarkIndex.LEFT_EYE].y = 0.30
    landmarks[PoseLandmarkIndex.RIGHT_EYE].y = 0.30
    landmarks[PoseLandmarkIndex.LEFT_ANKLE].y = 0.65
    landmarks[PoseLandmarkIndex.RIGHT_ANKLE].y = 0.65
    landmarks[PoseLandmarkIndex.LEFT_HEEL].y = 0.67
    landmarks[PoseLandmarkIndex.RIGHT_HEEL].y = 0.67

    result = engine.evaluate(landmarks)
    assert result.all_rules_passed is False
    assert result.primary_message == "Take a step closer"


def test_anthropometric_extraction_and_silhouette() -> None:
    extractor = AnthropometricExtractor(reference_height_cm=175.0)
    landmarks = _make_valid_pose_landmarks()
    mask = _DummyMask(720, 1280, 180, 420, 520, 760)

    m = extractor.extract(landmarks, frame_height=720, segmentation_mask=mask)
    assert m is not None
    assert m.full_body_px > 400
    assert m.shoulder_width_cm is not None and m.shoulder_width_cm > 30
    assert m.chest_silhouette_px is not None and m.chest_silhouette_px > 200
    assert m.estimated_upper_size in {"XS", "S", "M", "L", "XL", "XXL"}
    assert m.estimated_lower_size in {"XS", "S", "M", "L", "XL", "XXL"}

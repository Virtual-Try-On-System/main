from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from body.anthropometry import (
    AnthropometricExtractor,
    AnthropometricMeasurements,
    validate_user_anthropometrics,
)
from body.landmarks import LandmarkPoint
from body.pose_rules import PoseEvaluationResult, PoseRuleEngine, RuleConfig
from schemas.body import Body


@dataclass
class BodyAnalysisResult:
    """Unified output of a single frame or uploaded photo body analysis pass."""

    landmarks: dict[int, LandmarkPoint] | None
    evaluation: PoseEvaluationResult
    measurements: AnthropometricMeasurements | None
    body_schema: Body | None


class BodyAnalysisPipeline:
    """End-to-end body analysis coordinator combining pose rule evaluation,
    centimetre-per-pixel scaling, anthropometric extraction, and Body schema generation.
    """

    def __init__(
        self,
        height_cm: float = 175.0,
        weight_kg: float = 70.0,
        rule_config: RuleConfig | None = None,
        stabilize_scale: bool = False,
    ) -> None:
        h_cm, w_kg = validate_user_anthropometrics(height_cm, weight_kg)
        self.height_cm = h_cm
        self.weight_kg = w_kg
        self.rule_engine = PoseRuleEngine(config=rule_config)
        self.extractor = AnthropometricExtractor(
            reference_height_cm=self.height_cm,
            reference_weight_kg=self.weight_kg,
            stabilize_scale=stabilize_scale,
        )

    def update_user_anthropometrics(self, height_cm: float, weight_kg: float) -> None:
        h_cm, w_kg = validate_user_anthropometrics(height_cm, weight_kg)
        self.height_cm = h_cm
        self.weight_kg = w_kg
        self.extractor.reference_height_cm = h_cm
        self.extractor.reference_weight_kg = w_kg
        self.extractor.reset_calibration()

    def analyze_landmarks(
        self,
        landmarks: dict[int, LandmarkPoint] | None,
        frame_height: int,
        segmentation_mask: Any = None,
    ) -> BodyAnalysisResult:
        """Evaluate pose rules, compute cm/pixel and measurements, and build Body schema."""
        evaluation = self.rule_engine.evaluate(landmarks)
        measurements = self.extractor.extract(
            landmarks=landmarks,
            frame_height=frame_height,
            segmentation_mask=segmentation_mask,
        )
        body_schema: Body | None = None
        if measurements is not None and landmarks:
            body_schema = measurements.to_body_schema(
                landmarks=landmarks,
                weight_kg=self.weight_kg,
                height_cm=self.height_cm,
            )

        return BodyAnalysisResult(
            landmarks=landmarks,
            evaluation=evaluation,
            measurements=measurements,
            body_schema=body_schema,
        )

"""JSON contracts between the garment, body and fit modules."""

from schemas.body import Body, PoseKeypoint
from schemas.common import SCHEMA_VERSION
from schemas.fit_result import FitResult
from schemas.garment import FitClass, Garment, GarmentCategory, GarmentRatios, SizeLabel

__all__ = [
    "SCHEMA_VERSION",
    "Body",
    "FitClass",
    "FitResult",
    "Garment",
    "GarmentCategory",
    "GarmentRatios",
    "PoseKeypoint",
    "SizeLabel",
]

from typing import Annotated

from pydantic import Field

from schemas.common import PositiveFloat, SchemaModel, VersionedSchema


class PoseKeypoint(SchemaModel):
    name: Annotated[str, Field(min_length=1)]
    x_px: Annotated[float, Field(ge=0)]
    y_px: Annotated[float, Field(ge=0)]
    confidence: Annotated[float, Field(ge=0, le=1)]


class Body(VersionedSchema):
    height_cm: Annotated[float, Field(ge=140, le=210)]
    weight_kg: Annotated[float, Field(ge=40, le=150)]
    cm_per_pixel: PositiveFloat
    shoulder_width_cm: PositiveFloat
    torso_length_cm: PositiveFloat
    body_width_cm: PositiveFloat
    pose_keypoints: Annotated[list[PoseKeypoint], Field(min_length=1)]

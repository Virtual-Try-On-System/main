from schemas.common import PositiveFloat, RelativePath, VersionedSchema


class FitResult(VersionedSchema):
    """Where the garment ends on the body. Ease is negative when the garment is tight."""

    hem_from_shoulder_cm: PositiveFloat
    sleeve_end_from_shoulder_cm: PositiveFloat
    side_width_cm: PositiveFloat
    width_ease_cm: float
    length_ease_cm: float
    target_mask_path: RelativePath

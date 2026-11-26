from enum import StrEnum

from schemas.common import PositiveFloat, RelativePath, SchemaModel, VersionedSchema


class GarmentCategory(StrEnum):
    TSHIRT = "tshirt"
    SWEATSHIRT = "sweatshirt"
    SHIRT = "shirt"


class FitClass(StrEnum):
    CROP = "crop"
    REGULAR = "regular"
    OVERSIZE = "oversize"


class SizeLabel(StrEnum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


class GarmentRatios(SchemaModel):
    """Proportions measured on the garment mask, so they have no unit."""

    length_to_chest_width: PositiveFloat
    sleeve_to_length: PositiveFloat
    shoulder_to_chest_width: PositiveFloat


class Garment(VersionedSchema):
    category: GarmentCategory
    fit_class: FitClass
    size_label: SizeLabel
    mask_path: RelativePath
    ratios: GarmentRatios

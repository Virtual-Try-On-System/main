"""Garment Descriptor: the output contract of garment analysis (Task 3, Goal 1).

From a garment image (flat-lay or on-model) this message carries the category,
per-axis cut classes with confidences, the segmentation mask, the ratio vector
and the size label. It extends the draft ``schemas/garment.py`` from the
``feature/json-schemas`` branch: the single ``fit_class`` is replaced by three
ordinal axes (length, fit, sleeve) plus two optional nominal axes (neckline,
closure), and confidence values are added throughout.

Style follows ``schemas/common.py``: unknown fields are rejected, types are
strict and messages never embed images (only relative paths or coordinates).

Run this file directly to (re)generate ``garment/schema/garment_descriptor.schema.json``.
"""

import json
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "garment_descriptor.schema.json"

Confidence = Annotated[float, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class SchemaModel(BaseModel):
    """Base for every message exchanged between modules."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _check_relative_path(value: str) -> str:
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute() or windows_path.drive or PurePosixPath(value).is_absolute():
        raise ValueError("path must be relative")
    if ".." in windows_path.parts:
        raise ValueError("path must not leave its base folder")
    return value


RelativePath = Annotated[str, Field(min_length=1), AfterValidator(_check_relative_path)]


class GarmentCategory(StrEnum):
    TSHIRT = "tshirt"
    SWEATSHIRT = "sweatshirt"
    SHIRT = "shirt"


class Domain(StrEnum):
    FLAT = "flat"
    ON_MODEL = "on_model"


class LengthClass(StrEnum):
    CROP = "crop"
    SHORT = "short"
    REGULAR = "regular"
    LONG = "long"


class FitClass(StrEnum):
    TIGHT = "tight"
    REGULAR = "regular"
    LOOSE = "loose"
    OVERSIZE = "oversize"


class SleeveClass(StrEnum):
    SLEEVELESS = "sleeveless"
    SHORT = "short"
    THREE_QUARTER = "three_quarter"
    LONG = "long"


class NecklineClass(StrEnum):
    ROUND = "round"
    V_NECK = "v_neck"
    POLO = "polo"
    SHIRT_COLLAR = "shirt_collar"
    HOODED = "hooded"
    MANDARIN = "mandarin"
    BOAT = "boat"
    SQUARE = "square"
    OFF_SHOULDER = "off_shoulder"
    COLLARLESS = "collarless"
    OTHER = "other"


class ClosureClass(StrEnum):
    BUTTON = "button"
    ZIP = "zip"
    NONE = "none"
    OTHER = "other"


class SizeLabel(StrEnum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"
    UNKNOWN = "unknown"


class Axis(StrEnum):
    LENGTH = "length"
    FIT = "fit"
    SLEEVE = "sleeve"
    NECKLINE = "neckline"
    CLOSURE = "closure"


# Single source of truth for per-axis labels; taxonomy.json and the generated
# JSON schema are checked against this mapping in tests.
AXIS_LABELS = {
    Axis.LENGTH: tuple(c.value for c in LengthClass),
    Axis.FIT: tuple(c.value for c in FitClass),
    Axis.SLEEVE: tuple(c.value for c in SleeveClass),
    Axis.NECKLINE: tuple(c.value for c in NecklineClass),
    Axis.CLOSURE: tuple(c.value for c in ClosureClass),
}
ORDINAL_AXES = {Axis.LENGTH, Axis.FIT, Axis.SLEEVE}
REQUIRED_AXES = {Axis.LENGTH, Axis.FIT, Axis.SLEEVE}


class CategoryPrediction(SchemaModel):
    label: GarmentCategory
    confidence: Confidence


class CutAxis(SchemaModel):
    """One cut axis prediction. Ordinal axes carry an ordered label set."""

    axis: Axis
    label: str = Field(min_length=1)
    confidence: Confidence
    ordinal: bool
    low_confidence: bool

    @model_validator(mode="after")
    def _check_label_and_ordinality(self):
        allowed = AXIS_LABELS[self.axis]
        if self.label not in allowed:
            raise ValueError(f"label {self.label!r} is not valid for axis {self.axis}: {allowed}")
        expected = self.axis in ORDINAL_AXES
        if self.ordinal != expected:
            raise ValueError(f"ordinal must be {expected} for axis {self.axis}")
        return self


class MaskFormat(StrEnum):
    POLYGON = "polygon"
    RLE = "rle"
    PNG_PATH = "png_path"


class MaskSource(StrEnum):
    MODEL = "model"
    DATASET = "dataset"
    MANUAL = "manual"


class Mask(SchemaModel):
    """Segmentation output: inline polygon/RLE data or a relative PNG path."""

    format: MaskFormat
    data: str = Field(min_length=1)
    source: MaskSource

    @model_validator(mode="after")
    def _check_path_when_png(self):
        if self.format == MaskFormat.PNG_PATH:
            _check_relative_path(self.data)
        return self


class RatioVector(SchemaModel):
    """Mask geometry ratios; only meaningful on flat-lay images.

    length_to_chest_width = garment length / chest width,
    sleeve_to_chest_width = sleeve length / chest width,
    shoulder_to_chest_width = shoulder width / chest width.
    On on-model images all values are None and valid is False.
    """

    length_to_chest_width: NonNegativeFloat | None
    sleeve_to_chest_width: NonNegativeFloat | None
    shoulder_to_chest_width: NonNegativeFloat | None
    valid: bool


class GarmentDescriptor(SchemaModel):
    schema_version: Literal["1.0"]
    image_id: str = Field(min_length=1)
    domain: Domain
    category: CategoryPrediction
    cut: list[CutAxis] = Field(min_length=3, max_length=5)
    mask: Mask
    ratios: RatioVector
    size_label: SizeLabel
    low_confidence: bool
    runtime_ms: NonNegativeFloat

    @model_validator(mode="after")
    def _check_axes(self):
        axes = [c.axis for c in self.cut]
        if len(set(axes)) != len(axes):
            raise ValueError("each axis may appear at most once in cut")
        missing = REQUIRED_AXES - set(axes)
        if missing:
            raise ValueError(f"missing required cut axes: {sorted(a.value for a in missing)}")
        return self

    @model_validator(mode="after")
    def _check_ratios_match_domain(self):
        if self.domain == Domain.ON_MODEL:
            values = (
                self.ratios.length_to_chest_width,
                self.ratios.sleeve_to_chest_width,
                self.ratios.shoulder_to_chest_width,
            )
            if self.ratios.valid or any(v is not None for v in values):
                raise ValueError("on_model images must have valid=False and null ratios")
        return self


def export_json_schema(path: Path = SCHEMA_PATH) -> Path:
    """Write the JSON Schema artifact generated from the Pydantic model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = GarmentDescriptor.model_json_schema()
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"JSON schema written to {export_json_schema()}")

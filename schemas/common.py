from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

PositiveFloat = Annotated[float, Field(gt=0)]


def _check_relative_path(value: str) -> str:
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute() or windows_path.drive or PurePosixPath(value).is_absolute():
        raise ValueError("path must be relative")
    if ".." in windows_path.parts:
        raise ValueError("path must not leave its base folder")
    return value


RelativePath = Annotated[str, Field(min_length=1), AfterValidator(_check_relative_path)]


class SchemaModel(BaseModel):
    """Base for every message exchanged between modules."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class VersionedSchema(SchemaModel):
    schema_version: Literal["1.0"]

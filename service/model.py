"""The single interface every try-on model implements.

The service only talks to this interface, so a model can be replaced without changing the
service code. A model gets the person, the garment and the target mask and returns an image.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from schemas import FitResult


@dataclass(frozen=True)
class TryOnInput:
    person: Image.Image
    """RGB photo of the person."""

    garment: Image.Image
    """RGB photo of the garment."""

    target_mask: Image.Image
    """Single channel mask, same size as the person photo. White marks where the garment goes."""

    fit: FitResult
    """Output of the fit engine for this person and garment."""


@runtime_checkable
class TryOnModel(Protocol):
    name: str

    def generate(self, request: TryOnInput) -> Image.Image:
        """Return an RGB image of the person wearing the garment, same size as the person photo."""
        ...

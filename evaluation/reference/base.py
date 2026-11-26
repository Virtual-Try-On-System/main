"""Common interface for reference try-on models."""

from abc import ABC, abstractmethod

from PIL import Image

CUTS = ("crop", "regular", "oversize")
CATEGORIES = ("tops", "bottoms", "one-pieces")


class ReferenceModel(ABC):
    """A third-party try-on model used only as a comparison baseline."""

    name: str = "reference"
    dtype: str = "unknown"

    def load(self) -> None:  # noqa: B027 - optional hook
        """Load weights. Called once before the first generate()."""

    def device_info(self) -> dict:
        """Extra facts about the run, written to run.json."""
        return {}

    @abstractmethod
    def generate(self, person: Image.Image, garment: Image.Image, category: str) -> Image.Image:
        """Dress the person in the garment and return the result image."""

"""Adapter for FASHN VTON v1.5 (Apache-2.0). The library is imported lazily."""

from pathlib import Path

from PIL import Image

from evaluation.reference.base import ReferenceModel

INSTALL_HINT = (
    "FASHN VTON is not installed. Install it into this environment with:\n"
    "  git clone https://github.com/fashn-AI/fashn-vton-1.5.git\n"
    "  uv pip install -e ./fashn-vton-1.5\n"
    "See docs/reference-model.md."
)


class ReferenceModelUnavailable(RuntimeError):
    """The reference library or its weights are not available."""


class FashnReference(ReferenceModel):
    name = "fashn-vton-1.5"

    def __init__(
        self,
        weights_dir: str | Path = "weights/fashn",
        device: str | None = None,
        garment_photo_type: str = "flat-lay",
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        seed: int = 42,
    ):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self.garment_photo_type = garment_photo_type
        self.options = {
            "num_timesteps": num_timesteps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        self.dtype = "unknown (library picks bf16 on Ampere+ GPUs, else fp32)"
        self._pipeline = None

    def load(self) -> None:
        if not (self.weights_dir / "model.safetensors").is_file():
            raise ReferenceModelUnavailable(
                f"FASHN weights not found in {self.weights_dir}. Run in the FASHN repo:\n"
                f"  python scripts/download_weights.py --weights-dir {self.weights_dir}"
            )
        try:
            from fashn_vton import TryOnPipeline
        except ImportError as exc:
            raise ReferenceModelUnavailable(INSTALL_HINT) from exc
        self._pipeline = TryOnPipeline(weights_dir=str(self.weights_dir), device=self.device)
        self.dtype = self._detect_dtype()

    @staticmethod
    def _detect_dtype() -> str:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "bfloat16"
        except ImportError:
            pass
        return "float32"

    def device_info(self) -> dict:
        return {
            "garment_photo_type": self.garment_photo_type,
            "fashn_options": self.options,
            "weights_dir": str(self.weights_dir),
        }

    def generate(self, person: Image.Image, garment: Image.Image, category: str) -> Image.Image:
        if self._pipeline is None:
            self.load()
        result = self._pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=self.garment_photo_type,
            **self.options,
        )
        return result.images[0]

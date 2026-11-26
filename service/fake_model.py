from PIL import Image

from service.model import TryOnInput


class FakeTryOnModel:
    """Placeholder until a real model is trained.

    It stretches the garment photo over the person photo inside the target mask, so the service
    can be built and tested without a GPU or model weights.
    """

    name = "fake"

    def generate(self, request: TryOnInput) -> Image.Image:
        size = request.person.size
        garment = request.garment.resize(size)
        return Image.composite(garment, request.person, request.target_mask)

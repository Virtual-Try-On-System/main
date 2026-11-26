"""Try-on service.

Run it locally with: uv run uvicorn service.app:create_app --factory
Images are kept in memory and never written to disk.
"""

import io
import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.formparsers import MultiPartParser

from schemas import SCHEMA_VERSION, FitResult
from service.limits import BodySizeLimit
from service.model import TryOnInput, TryOnModel
from service.registry import load_model

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 3 * MAX_UPLOAD_BYTES + 1024 * 1024
MAX_IMAGE_SIDE = 4096
# MPO is how Pillow reports JPEGs from many phones that carry extra pictures.
ALLOWED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP"}
# 16 bit and float images would turn white when converted, so they are rejected.
REJECTED_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}

# Starlette writes uploaded files above 1 MB to a temporary file on disk. Raising the limit to
# the largest accepted request keeps every upload in memory. This setting is process wide.
MultiPartParser.spool_max_size = MAX_REQUEST_BYTES


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Put a transparent photo on a white background."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba)
    return image


def _read_image(upload: UploadFile, field: str, is_mask: bool = False) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"{field}: file is larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(422, detail=f"{field}: only PNG, JPEG and WEBP are accepted")
        if max(image.size) > MAX_IMAGE_SIDE:
            raise HTTPException(422, detail=f"{field}: image side is above {MAX_IMAGE_SIDE} px")
        if image.mode in REJECTED_MODES:
            raise HTTPException(422, detail=f"{field}: 16 bit and float images are not accepted")
        image.load()
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, detail=f"{field}: not a readable image") from None

    if is_mask:
        # A mask with transparency marks the garment area in its alpha channel.
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            return image.convert("RGBA").getchannel("A")
        return image.convert("L")
    return _flatten_alpha(image).convert("RGB")


def _parse_fit_result(raw: str) -> FitResult:
    try:
        return FitResult.model_validate_json(raw)
    except ValidationError as error:
        raise HTTPException(422, detail=json.loads(error.json(include_url=False))) from None


def create_app(model: TryOnModel | None = None) -> FastAPI:
    app = FastAPI(title="VTON try-on service", version="0.1.0")
    app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)
    app.state.model = model if model is not None else load_model()

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {
            "status": "ok",
            "model": request.app.state.model.name,
            "schema_version": SCHEMA_VERSION,
        }

    @app.post(
        "/tryon",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}, "description": "Try-on result"}},
    )
    def tryon(
        request: Request,
        person_image: Annotated[UploadFile, File(description="Photo of the person")],
        garment_image: Annotated[UploadFile, File(description="Photo of the garment")],
        target_mask: Annotated[UploadFile, File(description="Mask, same size as the person")],
        fit_result: Annotated[str, Form(description="FitResult JSON from the fit engine")],
    ) -> Response:
        fit = _parse_fit_result(fit_result)
        person = _read_image(person_image, "person_image")
        garment = _read_image(garment_image, "garment_image")
        mask = _read_image(target_mask, "target_mask", is_mask=True)
        if mask.size != person.size:
            raise HTTPException(
                422,
                detail=f"target_mask: size {mask.size} differs from person_image {person.size}",
            )

        current_model: TryOnModel = request.app.state.model
        result = current_model.generate(
            TryOnInput(person=person, garment=garment, target_mask=mask, fit=fit)
        )
        if not isinstance(result, Image.Image) or result.size != person.size:
            raise HTTPException(500, detail="the model returned an invalid image")

        buffer = io.BytesIO()
        result.convert("RGB").save(buffer, format="PNG")
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"X-Model-Name": current_model.name},
        )

    return app

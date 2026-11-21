import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import service.app as service_app
from schemas import SCHEMA_VERSION
from service.app import create_app
from service.fake_model import FakeTryOnModel
from service.model import TryOnInput, TryOnModel
from service.registry import MODEL_ENV_VAR, load_model

EXAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "examples" / "fit_result"
VALID_FIT = EXAMPLES / "valid" / "loose.json"
INVALID_FITS = sorted((EXAMPLES / "invalid").glob("*.json"))
PERSON_SIZE = (48, 64)


def png(size=PERSON_SIZE, mode="RGB", color="red", fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def half_mask(size=PERSON_SIZE):
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, size[0], size[1] // 2))
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


def request_parts(person=None, garment=None, mask=None, fit=None):
    files = {
        "person_image": ("person.png", person if person is not None else png(), "image/png"),
        "garment_image": ("garment.png", garment if garment is not None else png(color="blue"),
                          "image/png"),
        "target_mask": ("mask.png", mask if mask is not None else half_mask(), "image/png"),
    }
    data = {"fit_result": fit if fit is not None else VALID_FIT.read_text(encoding="utf-8")}
    return {"files": files, "data": data}


@pytest.fixture
def client():
    return TestClient(create_app(FakeTryOnModel()))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake", "schema_version": SCHEMA_VERSION}


def test_tryon_returns_png_of_person_size(client):
    response = client.post("/tryon", **request_parts())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-model-name"] == "fake"
    result = Image.open(io.BytesIO(response.content))
    assert result.format == "PNG"
    assert result.size == PERSON_SIZE
    # The fake model puts the garment (blue) inside the mask and keeps the person (red) outside.
    assert result.convert("RGB").getpixel((10, 5)) == (0, 0, 255)
    assert result.convert("RGB").getpixel((10, 60)) == (255, 0, 0)


def test_tryon_accepts_jpeg(client):
    response = client.post("/tryon", **request_parts(person=png(fmt="JPEG")))
    assert response.status_code == 200


@pytest.mark.parametrize("path", INVALID_FITS, ids=lambda path: path.stem)
def test_tryon_rejects_invalid_fit_result(client, path):
    response = client.post("/tryon", **request_parts(fit=path.read_text(encoding="utf-8")))
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_tryon_rejects_broken_json(client):
    response = client.post("/tryon", **request_parts(fit="{not json"))
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["person_image", "garment_image", "target_mask"])
def test_tryon_rejects_missing_file(client, field):
    parts = request_parts()
    del parts["files"][field]
    response = client.post("/tryon", **parts)
    assert response.status_code == 422


def test_tryon_rejects_missing_fit_result(client):
    parts = request_parts()
    del parts["data"]["fit_result"]
    response = client.post("/tryon", **parts)
    assert response.status_code == 422


def test_tryon_rejects_file_that_is_not_an_image(client):
    response = client.post("/tryon", **request_parts(person=b"this is not an image"))
    assert response.status_code == 422
    assert "person_image" in response.json()["detail"]


def test_tryon_rejects_unsupported_format(client):
    response = client.post("/tryon", **request_parts(garment=png(fmt="BMP")))
    assert response.status_code == 422
    assert "garment_image" in response.json()["detail"]


def test_tryon_rejects_too_large_image(client):
    response = client.post("/tryon", **request_parts(person=png(size=(4097, 8))))
    assert response.status_code == 422


def test_tryon_rejects_mask_of_other_size(client):
    response = client.post("/tryon", **request_parts(mask=half_mask((10, 10))))
    assert response.status_code == 422
    assert "target_mask" in response.json()["detail"]


class GreyModel:
    """A second model, to show that the service does not depend on the model behind it."""

    name = "grey"

    def __init__(self):
        self.calls = []

    def generate(self, request: TryOnInput) -> Image.Image:
        self.calls.append(request)
        return Image.new("RGB", request.person.size, (128, 128, 128))


def test_other_model_works_without_service_changes():
    model = GreyModel()
    assert isinstance(model, TryOnModel)
    client = TestClient(create_app(model))

    assert client.get("/health").json()["model"] == "grey"
    response = client.post("/tryon", **request_parts())
    assert response.status_code == 200
    assert response.headers["x-model-name"] == "grey"
    assert Image.open(io.BytesIO(response.content)).getpixel((0, 0)) == (128, 128, 128)

    (call,) = model.calls
    assert call.person.mode == "RGB"
    assert call.garment.mode == "RGB"
    assert call.target_mask.mode == "L"
    assert call.fit.schema_version == SCHEMA_VERSION


def test_model_returning_wrong_size_is_a_server_error():
    class WrongSizeModel:
        name = "wrong"

        def generate(self, request):
            return Image.new("RGB", (1, 1))

    client = TestClient(create_app(WrongSizeModel()))
    response = client.post("/tryon", **request_parts())
    assert response.status_code == 500


def spy_client():
    model = GreyModel()
    return model, TestClient(create_app(model))


def test_tryon_accepts_webp_and_phone_jpeg(client):
    phone_jpeg = io.BytesIO()
    Image.new("RGB", PERSON_SIZE, "red").save(
        phone_jpeg, format="MPO", save_all=True, append_images=[Image.new("RGB", (8, 8))]
    )
    assert Image.open(io.BytesIO(phone_jpeg.getvalue())).format == "MPO"
    response = client.post(
        "/tryon", **request_parts(person=phone_jpeg.getvalue(), garment=png(fmt="WEBP"))
    )
    assert response.status_code == 200


def test_tryon_rotates_photo_by_exif():
    sideways = Image.new("RGB", (PERSON_SIZE[1], PERSON_SIZE[0]), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees
    buffer = io.BytesIO()
    sideways.save(buffer, format="JPEG", exif=exif)

    model, client = spy_client()
    response = client.post("/tryon", **request_parts(person=buffer.getvalue()))
    assert response.status_code == 200
    assert model.calls[0].person.size == PERSON_SIZE


def test_tryon_rejects_16_bit_image(client):
    response = client.post("/tryon", **request_parts(mask=png(mode="I;16", color=1000)))
    assert response.status_code == 422
    assert "16 bit" in response.json()["detail"]


def test_tryon_puts_transparent_garment_on_white_and_reads_mask_alpha():
    garment = png(mode="RGBA", color=(255, 0, 0, 0))
    mask = Image.new("RGBA", PERSON_SIZE, (0, 0, 0, 0))
    mask.paste((0, 0, 0, 255), (0, 0, PERSON_SIZE[0], 10))
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")

    model, client = spy_client()
    response = client.post("/tryon", **request_parts(garment=garment, mask=mask_bytes.getvalue()))
    assert response.status_code == 200
    call = model.calls[0]
    assert call.garment.getpixel((0, 0)) == (255, 255, 255)
    assert call.target_mask.getpixel((0, 0)) == 255
    assert call.target_mask.getpixel((0, 20)) == 0


def test_large_upload_stays_in_memory(monkeypatch):
    rolled = []
    original = service_app._read_image

    def spy(upload, field, is_mask=False):
        rolled.append(upload.file._rolled)
        return original(upload, field, is_mask)

    monkeypatch.setattr(service_app, "_read_image", spy)
    size = (1100, 1100)
    noise = io.BytesIO()
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(noise, format="PNG")
    assert len(noise.getvalue()) > 1024 * 1024

    client = TestClient(create_app(FakeTryOnModel()))
    response = client.post(
        "/tryon", **request_parts(person=noise.getvalue(), mask=half_mask(size))
    )
    assert response.status_code == 200
    assert rolled == [False, False, False]


def test_tryon_rejects_too_large_file(client, monkeypatch):
    monkeypatch.setattr(service_app, "MAX_UPLOAD_BYTES", 100)
    response = client.post("/tryon", **request_parts())
    assert response.status_code == 413


def test_request_above_limit_is_rejected_by_content_length(monkeypatch):
    monkeypatch.setattr(service_app, "MAX_REQUEST_BYTES", 1000)
    client = TestClient(create_app(FakeTryOnModel()))
    response = client.post("/tryon", **request_parts())
    assert response.status_code == 413


def test_streamed_request_above_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(service_app, "MAX_REQUEST_BYTES", 1000)
    client = TestClient(create_app(FakeTryOnModel()))

    def chunks():
        for _ in range(10):
            yield b"x" * 500

    response = client.post(
        "/tryon",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=abc"},
    )
    assert response.status_code == 413
    assert client.get("/health").status_code == 200


def test_registry_reads_model_name_from_environment(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "fake")
    assert load_model().name == "fake"
    monkeypatch.delenv(MODEL_ENV_VAR)
    assert load_model().name == "fake"


def test_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown try-on model"):
        load_model("does-not-exist")

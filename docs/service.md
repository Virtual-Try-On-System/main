# Try-on service

A FastAPI service with two endpoints. The service talks to the try-on model only through one
interface, so a model can be replaced without changing the service code. Until a real model is
trained, a fake model is plugged in.

## Run it

```bash
uv run uvicorn service.app:create_app --factory
```

`--factory` makes uvicorn call `create_app()`, so the model is loaded when the service starts and
not when the module is imported.

The service listens on `http://127.0.0.1:8000`. Interactive API docs are at
`http://127.0.0.1:8000/docs`, where you can also send test requests.

With Docker:

```bash
docker compose --profile service up --build api
```

The port is published on `127.0.0.1` only, so the service is not reachable from other machines.

## Endpoints

### `GET /health`

```json
{"status": "ok", "model": "fake", "schema_version": "1.0"}
```

### `POST /tryon`

A `multipart/form-data` request with four parts:

| Part | Type | Content |
|---|---|---|
| `person_image` | file | Photo of the person, PNG, JPEG or WEBP |
| `garment_image` | file | Photo of the garment, PNG, JPEG or WEBP |
| `target_mask` | file | Mask from the fit engine, same size as the upright person photo, white where the garment goes. If the mask has transparency, its alpha channel is used instead |
| `fit_result` | text | FitResult JSON, see [schemas.md](schemas.md) |

The answer is a PNG image of the same size as the person photo. The `X-Model-Name` header says
which model produced it.

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/tryon -F person_image=@person.png -F garment_image=@garment.png -F target_mask=@mask.png -F "fit_result=<schemas/examples/fit_result/valid/loose.json" -o result.png
```

`fit_result` is a text part, so curl reads the file with `<`. With `@` it would be sent as a file
and rejected.

Errors:

| Status | When |
|---|---|
| 422 | A part is missing, `fit_result` does not match the schema, a file is not a readable PNG, JPEG or WEBP, an image is 16 bit or float, an image side is above 4096 px, or the mask size differs from the person photo |
| 413 | A file is larger than 20 MB, or the whole request is larger than 61 MB. The request size is checked while it arrives, so a huge upload is stopped early |
| 500 | The model returned something other than an image of the person photo's size |

For a schema error, `detail` lists each failing field, in the same format as Pydantic.

How images are read:

- Phone photos are turned upright using their EXIF orientation, so the mask must be made from the
  upright photo.
- Transparent parts of the person and garment photos become white.
- Images are kept in memory and never written to disk. By default Starlette writes uploads above
  1 MB to a temporary file, so the service raises that limit. The test
  `test_large_upload_stays_in_memory` checks this.

## The model interface

Every model implements `TryOnModel` in [`service/model.py`](../service/model.py):

```python
class TryOnModel(Protocol):
    name: str

    def generate(self, request: TryOnInput) -> Image.Image: ...
```

`TryOnInput` holds the person photo (RGB), the garment photo (RGB), the target mask (single
channel, same size as the person photo) and the `FitResult`. The model returns an RGB image of
the same size as the person photo.

The fake model in [`service/fake_model.py`](../service/fake_model.py) stretches the garment photo
over the person inside the mask. It is only for building and testing the service.

## Adding a model

1. Write a class with a `name` and a `generate` method, for example in `service/my_model.py`.
2. Add it to `MODELS` in [`service/registry.py`](../service/registry.py).
3. Choose it with the `VTON_TRYON_MODEL` variable in `.env`, for example `VTON_TRYON_MODEL=my_model`.

Nothing in `service/app.py` changes. The test `test_other_model_works_without_service_changes`
checks this.

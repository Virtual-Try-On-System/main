"""Chooses the try-on model by name. A new model is added here, not in the service code."""

import os
from collections.abc import Callable

from service.fake_model import FakeTryOnModel
from service.model import TryOnModel

MODEL_ENV_VAR = "VTON_TRYON_MODEL"
DEFAULT_MODEL = "fake"

MODELS: dict[str, Callable[[], TryOnModel]] = {
    "fake": FakeTryOnModel,
}


def load_model(name: str | None = None) -> TryOnModel:
    name = name or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    try:
        factory = MODELS[name]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown try-on model {name!r}, known models: {known}") from None
    return factory()

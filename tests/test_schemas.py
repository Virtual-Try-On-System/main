from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas import Body, FitResult, Garment

EXAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "examples"
MODELS = {"garment": Garment, "body": Body, "fit_result": FitResult}


def example_cases(kind):
    return [
        pytest.param(MODELS[schema], path, id=f"{schema}-{path.stem}")
        for schema in MODELS
        for path in sorted((EXAMPLES / schema / kind).glob("*.json"))
    ]


@pytest.mark.parametrize("schema", MODELS)
@pytest.mark.parametrize("kind", ["valid", "invalid"])
def test_every_schema_has_examples(schema, kind):
    assert list((EXAMPLES / schema / kind).glob("*.json"))


@pytest.mark.parametrize(("model", "path"), example_cases("valid"))
def test_valid_example_is_accepted(model, path):
    model.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("model", "path"), example_cases("valid"))
def test_valid_example_survives_a_json_round_trip(model, path):
    parsed = model.model_validate_json(path.read_text(encoding="utf-8"))

    assert model.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(("model", "path"), example_cases("invalid"))
def test_invalid_example_is_rejected_for_its_named_field(model, path):
    """Invalid files are named <field>__<reason>.json and break exactly that field."""
    field = path.stem.split("__")[0]

    with pytest.raises(ValidationError) as caught:
        model.model_validate_json(path.read_text(encoding="utf-8"))

    errors = caught.value.errors()
    assert len(errors) == 1
    assert field in errors[0]["loc"]

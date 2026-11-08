"""Goal 1 acceptance: the garment descriptor accepts valid examples and rejects
invalid ones. Also checks that taxonomy.json and the generated JSON schema stay
in sync with garment/descriptor.py (single source of truth).
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from garment import descriptor as gd

EXAMPLES = Path(__file__).resolve().parent.parent / "garment" / "examples"
TAXONOMY = Path(__file__).resolve().parent.parent / "garment" / "taxonomy" / "taxonomy.json"


def _load(path):
    return path.read_text(encoding="utf-8")


def _valid_paths(kind):
    return sorted((EXAMPLES / kind).glob("*.json"))


@pytest.mark.parametrize("path", _valid_paths("valid"), ids=lambda p: p.name)
def test_valid_examples_accepted(path):
    assert len(_valid_paths("valid")) >= 5
    gd.GarmentDescriptor.model_validate_json(_load(path))


@pytest.mark.parametrize("path", _valid_paths("invalid"), ids=lambda p: p.name)
def test_invalid_examples_rejected(path):
    assert len(_valid_paths("invalid")) >= 5
    with pytest.raises(ValidationError):
        gd.GarmentDescriptor.model_validate_json(_load(path))


def test_duplicate_axis_rejected():
    example = json.loads(_load(EXAMPLES / "valid" / "tshirt_flat.json"))
    example["cut"].append(dict(example["cut"][0]))
    with pytest.raises(ValidationError, match="at most once"):
        gd.GarmentDescriptor.model_validate_json(json.dumps(example))


def test_taxonomy_matches_model():
    taxonomy = json.loads(_load(TAXONOMY))
    for axis_name, axis_def in taxonomy["axes"].items():
        axis = gd.Axis(axis_name)
        labels = [c["label"] for c in axis_def["classes"]]
        assert labels == list(gd.AXIS_LABELS[axis]), f"label mismatch on axis {axis_name}"
        assert axis_def["ordinal"] == (axis in gd.ORDINAL_AXES)
    taxonomy_categories = {c["label"] for c in taxonomy["categories"]}
    assert taxonomy_categories == {c.value for c in gd.GarmentCategory}


def test_json_schema_artifact_is_fresh():
    committed = json.loads(_load(gd.SCHEMA_PATH))
    assert committed == gd.GarmentDescriptor.model_json_schema(), (
        "garment/schema/garment_descriptor.schema.json is outdated; "
        "regenerate it with: python -m garment.descriptor"
    )

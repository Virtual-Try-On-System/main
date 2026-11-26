"""Tests for the reference model runner, using a fake model and generated images."""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from evaluation.reference.base import ReferenceModel
from evaluation.reference.fashn import FashnReference, ReferenceModelUnavailable
from evaluation.reference.gallery import build_gallery, read_marks
from evaluation.reference.manifest import ManifestError, read_manifest
from evaluation.reference.run import main
from evaluation.reference.runner import run_reference


class FakeModel(ReferenceModel):
    name = "fake"
    dtype = "float32"

    def __init__(self):
        self.calls = []

    def load(self):
        self.loaded = True

    def generate(self, person, garment, category):
        self.calls.append(category)
        return Image.new("RGB", (32, 48), (10, 200, 10))


def make_image(path: Path, color=(120, 120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 24), color).save(path)
    return path


def write_manifest(tmp_path, rows, header=("person", "garment", "cut", "category")):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return manifest


@pytest.fixture
def inputs(tmp_path):
    person = make_image(tmp_path / "in" / "person.png")
    rows = []
    for cut in ("crop", "regular", "oversize"):
        garment = make_image(tmp_path / "in" / f"{cut}_top.png", (200, 50, 50))
        rows.append((str(person), str(garment), cut, "tops"))
    return write_manifest(tmp_path, rows)


def test_read_manifest_parses_rows(inputs):
    rows = read_manifest(inputs)
    assert [r.cut for r in rows] == ["crop", "regular", "oversize"]
    assert rows[0].category == "tops"
    assert rows[0].output_name == "person__crop_top.png"


def test_relative_paths_resolve_against_manifest_folder(tmp_path):
    make_image(tmp_path / "p.png")
    make_image(tmp_path / "g.png")
    manifest = write_manifest(tmp_path, [("p.png", "g.png", "crop", "tops")])
    assert read_manifest(manifest)[0].person == tmp_path / "p.png"


def test_manifest_rejects_unknown_cut(tmp_path):
    make_image(tmp_path / "p.png")
    make_image(tmp_path / "g.png")
    manifest = write_manifest(tmp_path, [("p.png", "g.png", "baggy", "tops")])
    with pytest.raises(ManifestError, match="cut"):
        read_manifest(manifest)


def test_manifest_rejects_unknown_category(tmp_path):
    make_image(tmp_path / "p.png")
    make_image(tmp_path / "g.png")
    manifest = write_manifest(tmp_path, [("p.png", "g.png", "crop", "hats")])
    with pytest.raises(ManifestError, match="category"):
        read_manifest(manifest)


def test_manifest_rejects_missing_file(tmp_path):
    make_image(tmp_path / "p.png")
    manifest = write_manifest(tmp_path, [("p.png", "nope.png", "crop", "tops")])
    with pytest.raises(ManifestError, match="nope.png"):
        read_manifest(manifest)


def test_manifest_rejects_missing_column(tmp_path):
    manifest = write_manifest(tmp_path, [("p.png", "g.png", "crop")], ("person", "garment", "cut"))
    with pytest.raises(ManifestError, match="category"):
        read_manifest(manifest)


def test_manifest_rejects_duplicate_output_names(tmp_path):
    make_image(tmp_path / "p.png")
    make_image(tmp_path / "g.png")
    manifest = write_manifest(
        tmp_path,
        [("p.png", "g.png", "crop", "tops"), ("p.png", "g.png", "regular", "tops")],
    )
    with pytest.raises(ManifestError, match="duplicate"):
        read_manifest(manifest)


def test_run_writes_outputs_table_and_record(inputs, tmp_path):
    out = tmp_path / "out"
    model = FakeModel()
    records = run_reference(model, read_manifest(inputs), out)

    assert len(records) == 3
    assert model.calls == ["tops"] * 3
    for name in ("person__crop_top.png", "person__regular_top.png", "person__oversize_top.png"):
        assert (out / name).is_file()

    with (out / "timings.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert [r["cut"] for r in table] == ["crop", "regular", "oversize"]
    assert float(table[0]["seconds"]) >= 0
    assert "peak_vram_mb" in table[0]
    assert table[0]["output"] == "person__crop_top.png"

    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["model"] == "fake"
    assert run_json["dtype"] == "float32"
    assert len(run_json["inputs"]) == 3
    assert len(run_json["inputs"][0]["person_sha256"]) == 64


def test_cli_runs_with_injected_model(inputs, tmp_path):
    out = tmp_path / "cli_out"
    code = main(["--inputs", str(inputs), "--out", str(out)], model=FakeModel())
    assert code == 0
    assert (out / "run.json").is_file()


def test_cli_reports_manifest_error(tmp_path, capsys):
    manifest = write_manifest(tmp_path, [("a.png", "b.png", "crop", "tops")])
    code = main(["--inputs", str(manifest), "--out", str(tmp_path / "o")], model=FakeModel())
    assert code == 2
    assert "a.png" in capsys.readouterr().err


def test_fashn_missing_library_gives_clear_error(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fashn_vton", None)  # forces ImportError
    (tmp_path / "model.safetensors").write_bytes(b"")
    model = FashnReference(weights_dir=tmp_path)
    with pytest.raises(ReferenceModelUnavailable, match="pip install"):
        model.load()


def test_fashn_missing_weights_gives_clear_error(tmp_path):
    model = FashnReference(weights_dir=tmp_path / "missing")
    with pytest.raises(ReferenceModelUnavailable, match="download_weights"):
        model.load()


def test_gallery_shows_marks_per_cut(inputs, tmp_path):
    out = tmp_path / "out"
    run_reference(FakeModel(), read_manifest(inputs), out)
    marks = tmp_path / "marks.csv"
    marks.write_text(
        "output,hem_shifted,note\n"
        "person__crop_top.png,yes,hem rides up\n"
        "person__regular_top.png,no,\n",
        encoding="utf-8",
    )
    assert read_marks(marks)["person__crop_top.png"] == ("yes", "hem rides up")

    html_path = build_gallery(out, tmp_path / "gallery.html", marks)
    html = html_path.read_text(encoding="utf-8")
    for cut in ("crop", "regular", "oversize"):
        assert f"<h2>{cut}" in html
    assert "hem rides up" in html
    assert 'class="mark yes"' in html
    assert 'class="mark no"' in html
    assert 'class="mark unmarked"' in html  # oversize has no mark yet
    assert html.count("<img") == 9


def test_gallery_escapes_notes(inputs, tmp_path):
    out = tmp_path / "out"
    run_reference(FakeModel(), read_manifest(inputs), out)
    marks = tmp_path / "marks.csv"
    marks.write_text(
        'output,hem_shifted,note\nperson__crop_top.png,yes,"<script>x</script>"\n',
        encoding="utf-8",
    )
    html = build_gallery(out, tmp_path / "g.html", marks).read_text(encoding="utf-8")
    assert "<script>" not in html


def test_gallery_rejects_bad_mark_value(tmp_path):
    marks = tmp_path / "marks.csv"
    marks.write_text("output,hem_shifted,note\na.png,maybe,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maybe"):
        read_marks(marks)

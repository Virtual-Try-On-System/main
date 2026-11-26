import csv
import json

import numpy as np
import pytest
from PIL import Image

from evaluation.compare import _parse_resolution, main, run_comparison
from evaluation.metrics import (
    FolderPair,
    Metric,
    MetricResult,
    PairingError,
    default_metrics,
    list_metrics,
    load_image,
    pair_folders,
    register_metric,
    unregister_metric,
)

SIZE = 64


def make_folder(path, seed, names=("a.png", "b.png", "c.png"), noise=0.0):
    """Write small random images. With the same seed the images are identical."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        base = rng.integers(0, 256, (SIZE, SIZE, 3)).astype(np.float32)
        if noise:
            noise_rng = np.random.default_rng(seed + 1000)
            base = base + noise_rng.normal(0, noise, base.shape)
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(path / name)
    return path


class FakeLpipsNet:
    """Stand-in for the LPIPS network: mean absolute difference, no weights needed."""

    def __call__(self, a, b):
        return (a - b).abs().mean(dim=(1, 2, 3))


class FakeExtractor:
    """Stand-in for the Inception network: average colour plus a few pooled values."""

    def __call__(self, batch):
        x = batch.float() / 255.0
        return x.mean(dim=(2, 3))


def fake_metrics(fid_min_samples=1):
    return default_metrics(
        lpips_net=FakeLpipsNet(), fid_extractor=FakeExtractor(), fid_min_samples=fid_min_samples
    )


@pytest.fixture
def folders(tmp_path):
    ref = make_folder(tmp_path / "ref", seed=1)
    same = make_folder(tmp_path / "same", seed=1)
    noisy = make_folder(tmp_path / "noisy", seed=1, noise=40)
    return ref, same, noisy


def test_identical_folders_have_perfect_ssim(folders):
    ref, same, _ = folders
    pairs = pair_folders(ref, same)
    result = fake_metrics()["ssim"].compute(pairs)
    assert result.values["ssim"] == pytest.approx(1.0)
    assert result.count == 3


def test_noisy_folder_scores_lower_ssim(folders):
    ref, _, noisy = folders
    result = fake_metrics()["ssim"].compute(pair_folders(ref, noisy))
    assert result.values["ssim"] < 0.9


def test_missing_and_extra_files_are_reported(tmp_path):
    ref = make_folder(tmp_path / "ref", 1, names=("a.png", "b.png"))
    model = make_folder(tmp_path / "model", 1, names=("b.png", "z.png"))
    with pytest.raises(PairingError) as err:
        pair_folders(ref, model)
    message = str(err.value)
    assert "a.png" in message and "z.png" in message
    assert "missing" in message and "extra" in message


def test_allow_mismatch_uses_common_files_and_records_it(tmp_path):
    ref = make_folder(tmp_path / "ref", 1, names=("a.png", "b.png"))
    model = make_folder(tmp_path / "model", 1, names=("b.png", "z.png"))
    pairs = pair_folders(ref, model, allow_mismatch=True)
    assert isinstance(pairs, FolderPair)
    assert pairs.names == ["b.png"]
    assert pairs.missing == ["a.png"]
    assert pairs.extra == ["z.png"]


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    ref = make_folder(tmp_path / "ref", 1)
    with pytest.raises(PairingError):
        pair_folders(ref, tmp_path / "empty")


def test_table_has_one_row_per_model(folders):
    ref, same, noisy = folders
    table = run_comparison(ref, {"ours": same, "other": noisy}, metrics=fake_metrics())
    assert [row["model"] for row in table.rows] == ["ours", "other"]
    row = table.rows[0]
    for key in ("ssim", "ssim_std", "lpips", "lpips_std", "fid", "n_images"):
        assert key in row
    assert row["n_images"] == 3
    assert table.rows[0]["ssim"] > table.rows[1]["ssim"]


def test_results_are_identical_across_runs(folders):
    ref, same, noisy = folders
    models = {"ours": same, "other": noisy}
    first = run_comparison(ref, models, metrics=fake_metrics())
    second = run_comparison(ref, models, metrics=fake_metrics())
    assert first.rows == second.rows


def test_new_metric_can_be_registered_without_changing_runner(folders):
    ref, same, _ = folders

    class FileCount(Metric):
        name = "file_count"

        def compute(self, pairs):
            n = len(pairs.names)
            return MetricResult(values={"file_count": float(n)}, count=n)

    register_metric(FileCount())
    try:
        assert "file_count" in list_metrics()
        metrics = {**fake_metrics(), "file_count": FileCount()}
        table = run_comparison(ref, {"ours": same}, metrics=metrics)
        assert table.rows[0]["file_count"] == 3.0
    finally:
        unregister_metric("file_count")
    assert "file_count" not in list_metrics()


def test_registered_metric_is_used_by_build_metrics():
    from evaluation.compare import build_metrics

    class Dummy(Metric):
        name = "dummy"

        def compute(self, pairs):
            return MetricResult(values={"dummy": 1.0}, count=len(pairs))

    register_metric(Dummy())
    try:
        assert "dummy" in build_metrics()
    finally:
        unregister_metric("dummy")


def test_small_sample_fid_warns(folders):
    ref, same, _ = folders
    table = run_comparison(ref, {"ours": same}, metrics=fake_metrics(2048))
    assert any("FID" in w and "unreliable" in w for w in table.warnings)


def test_metadata_records_settings_and_versions(folders):
    ref, same, _ = folders
    table = run_comparison(ref, {"ours": same}, metrics=fake_metrics())
    meta = table.metadata
    assert meta["resolution_hw"] == [256, 192]
    assert meta["seed"] == 0
    assert "versions" in meta and "numpy" in meta["versions"]
    assert "ssim" in meta["metric_settings"]


def test_cli_writes_csv_markdown_json(folders, tmp_path, capsys, monkeypatch):
    ref, same, noisy = folders
    monkeypatch.setattr("evaluation.compare.build_metrics", fake_metrics)
    out = tmp_path / "out" / "table.csv"
    code = main(
        [
            "--reference", str(ref),
            "--model", f"ours={same}",
            "--model", f"other={noisy}",
            "--out", str(out),
            "--fid-min-samples", "1",
        ]
    )
    assert code == 0
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["ours", "other"]
    assert out.with_suffix(".md").is_file()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["resolution_hw"] == [256, 192]
    assert "ours" in capsys.readouterr().out


def test_cli_reports_pairing_error(tmp_path, capsys, monkeypatch):
    ref = make_folder(tmp_path / "ref", 1, names=("a.png",))
    model = make_folder(tmp_path / "m", 1, names=("b.png",))
    monkeypatch.setattr("evaluation.compare.build_metrics", fake_metrics)
    code = main(["--reference", str(ref), "--model", f"m={model}"])
    assert code == 2
    assert "a.png" in capsys.readouterr().err


@pytest.mark.real_weights
def test_real_weights_smoke(folders):
    ref, same, _ = folders
    from evaluation.compare import build_metrics

    table = run_comparison(ref, {"ours": same}, metrics=build_metrics())
    assert table.rows[0]["ssim"] == pytest.approx(1.0)
    assert table.rows[0]["lpips"] == pytest.approx(0.0, abs=1e-4)


def test_portrait_images_keep_their_shape(tmp_path):
    path = tmp_path / "portrait.png"
    Image.new("RGB", (768, 1024), "red").save(path)
    assert load_image(path).shape == (256, 192, 3)
    assert load_image(path, (512, 384)).shape == (512, 384, 3)


def test_resolution_option_is_height_by_width():
    import argparse

    assert _parse_resolution("512x384") == (512, 384)
    for bad in ("512", "0x384", "ax384"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_resolution(bad)

"""Tests for the training data pipeline, using a tiny synthetic VITON-HD-like dataset.

All images are generated here in ``tmp_path``; no image is ever committed.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from training.data import check, splits
from training.data.dresscode import DressCodeDataset
from training.data.viton_hd import DatasetError, VitonHDDataset, verify_dataset

NATIVE = (64, 48)  # (height, width) of the synthetic "full resolution" images
TARGET = (32, 24)
N_TRAIN = 20
N_TEST = 4
# Rectangle (top, bottom, left, right) where the garment sits in every synthetic person image.
GARMENT_BOX = (16, 40, 12, 36)


def _person_array() -> np.ndarray:
    arr = np.zeros((*NATIVE, 3), dtype=np.uint8)
    arr[:] = (200, 30, 30)  # red person, clearly not gray
    return arr


def _write_sample(root: Path, phase: str, name: str, *, with_agnostic: bool) -> None:
    top, bottom, left, right = GARMENT_BOX
    base = root / phase
    for folder in ("image", "cloth", "cloth-mask", "image-parse-v3", "agnostic-v3.2"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem

    person = _person_array()
    Image.fromarray(person).save(base / "image" / name)

    cloth = np.full((*NATIVE, 3), 255, dtype=np.uint8)
    cloth[top:bottom, left:right] = (10, 200, 10)
    Image.fromarray(cloth).save(base / "cloth" / name)

    mask = np.zeros(NATIVE, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    Image.fromarray(mask, mode="L").save(base / "cloth-mask" / name)

    parse = np.zeros(NATIVE, dtype=np.uint8)
    parse[top:bottom, left:right] = 5  # upper clothes
    Image.fromarray(parse, mode="L").save(base / "image-parse-v3" / f"{stem}.png")

    if with_agnostic:
        agnostic = person.copy()
        agnostic[top:bottom, left:right] = 128
        Image.fromarray(agnostic).save(base / "agnostic-v3.2" / name)
    else:
        shutil.rmtree(base / "agnostic-v3.2", ignore_errors=True)


def make_viton_hd(root: Path, *, with_agnostic: bool = True) -> Path:
    train_lines, test_lines = [], []
    # Like the real release, the published lists are unpaired: each person is listed with the
    # next person's garment.
    train_names = [f"{i:05d}_00.jpg" for i in range(N_TRAIN)]
    test_names = [f"{100 + i:05d}_00.jpg" for i in range(N_TEST)]
    phases = (("train", train_names, train_lines), ("test", test_names, test_lines))
    for phase, names, lines in phases:
        for i, name in enumerate(names):
            _write_sample(root, phase, name, with_agnostic=with_agnostic)
            lines.append(f"{name} {names[(i + 1) % len(names)]}")
    (root / "train_pairs.txt").write_text("\n".join(train_lines) + "\n")
    (root / "test_pairs.txt").write_text("\n".join(test_lines) + "\n")
    return root


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    return make_viton_hd(tmp_path / "viton-hd")


@pytest.fixture
def split_dir(dataset_root: Path, tmp_path: Path) -> Path:
    out = tmp_path / "splits"
    splits.create_splits(dataset_root, out, val_fraction=0.25, seed=7)
    return out


# ---------------------------------------------------------------- splits


def test_splits_are_deterministic(dataset_root, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    splits.create_splits(dataset_root, a, val_fraction=0.25, seed=7)
    splits.create_splits(dataset_root, b, val_fraction=0.25, seed=7)
    for name in ("train.txt", "val.txt", "test.txt", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_different_seed_changes_split(dataset_root, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    splits.create_splits(dataset_root, a, val_fraction=0.25, seed=1)
    splits.create_splits(dataset_root, b, val_fraction=0.25, seed=2)
    assert (a / "val.txt").read_text() != (b / "val.txt").read_text()


def test_splits_do_not_overlap_and_cover_official_train(split_dir, dataset_root):
    train = splits.read_pairs(split_dir / "train.txt")
    val = splits.read_pairs(split_dir / "val.txt")
    assert len(val) == N_TRAIN // 4
    assert not {p for p, _ in train} & {p for p, _ in val}
    assert sorted(train + val) == sorted(splits.read_pairs(dataset_root / "train_pairs.txt"))


def test_test_split_equals_official_test_pairs(split_dir, dataset_root):
    official = splits.read_pairs(dataset_root / "test_pairs.txt")
    assert splits.read_pairs(split_dir / "test.txt") == official


def test_manifest_records_sha256_and_verifies(split_dir):
    manifest = json.loads((split_dir / "manifest.json").read_text())
    assert manifest["seed"] == 7
    assert set(manifest["files"]) == {"train.txt", "val.txt", "test.txt"}
    assert all(len(v["sha256"]) == 64 for v in manifest["files"].values())
    assert splits.verify_splits(split_dir) == []


def test_verify_splits_detects_edit(split_dir):
    with (split_dir / "val.txt").open("a") as handle:
        handle.write("99999_00.jpg 99999_00.jpg\n")
    problems = splits.verify_splits(split_dir)
    assert any("val.txt" in p for p in problems)


def test_splits_cli_writes_files(dataset_root, tmp_path):
    out = tmp_path / "cli-splits"
    code = splits.main(["--root", str(dataset_root), "--out", str(out), "--val-fraction", "0.25"])
    assert code == 0
    assert (out / "train.txt").is_file() and (out / "manifest.json").is_file()


def test_split_missing_pairs_file_is_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_pairs.txt"):
        splits.create_splits(tmp_path, tmp_path / "out")


# ---------------------------------------------------------------- dataset


def test_sample_shapes_dtypes_ranges(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    sample = ds[0]
    h, w = TARGET
    for key in ("person", "garment", "agnostic"):
        assert sample[key].shape == (3, h, w)
        assert sample[key].dtype == torch.float32
        assert -1.0 <= sample[key].min() and sample[key].max() <= 1.0
    assert sample["garment_mask"].shape == (1, h, w)
    assert sample["garment_mask"].dtype == torch.float32
    assert isinstance(sample["person_id"], str) and isinstance(sample["garment_id"], str)


def test_lengths_match_split_files(dataset_root, split_dir):
    train = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    val = VitonHDDataset(dataset_root, "val", resolution=TARGET, split_dir=split_dir)
    test = VitonHDDataset(dataset_root, "test", resolution=TARGET, split_dir=split_dir)
    assert len(train) + len(val) == N_TRAIN
    assert len(test) == N_TEST


def test_mask_is_binary(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    mask = ds[0]["garment_mask"]
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0


def test_native_resolution_is_kept_when_requested(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "test", resolution=NATIVE, split_dir=split_dir)
    assert ds[0]["person"].shape == (3, *NATIVE)


def test_agnostic_from_folder_hides_garment_region(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=NATIVE, split_dir=split_dir)
    sample = ds[0]
    top, bottom, left, right = GARMENT_BOX
    # JPEG blurs the gray/red border, so look at the inner part of the box only.
    inside = sample["agnostic"][:, top + 4 : bottom - 4, left + 4 : right - 4]
    assert torch.allclose(inside, torch.zeros_like(inside), atol=0.08)  # gray 128 -> ~0
    assert not torch.allclose(sample["person"], sample["agnostic"])


def test_agnostic_from_parse_hides_garment_region(tmp_path):
    root = make_viton_hd(tmp_path / "no-agn", with_agnostic=False)
    out = tmp_path / "s"
    splits.create_splits(root, out, val_fraction=0.25, seed=1)
    ds = VitonHDDataset(root, "train", resolution=NATIVE, split_dir=out, agnostic="parse")
    sample = ds[0]
    top, bottom, left, right = GARMENT_BOX
    hidden = sample["agnostic"][:, top:bottom, left:right]
    assert torch.allclose(hidden, torch.zeros_like(hidden), atol=0.02)
    # Outside the hidden region the person is kept.
    assert torch.allclose(sample["agnostic"][:, :top], sample["person"][:, :top])


def test_auto_falls_back_to_parse_when_agnostic_folder_missing(tmp_path):
    root = make_viton_hd(tmp_path / "no-agn", with_agnostic=False)
    out = tmp_path / "s"
    splits.create_splits(root, out, val_fraction=0.25, seed=1)
    ds = VitonHDDataset(root, "train", resolution=TARGET, split_dir=out)  # default: auto
    assert ds.agnostic_source == "parse"


def test_folder_mode_without_agnostic_folder_is_clear_error(tmp_path):
    root = make_viton_hd(tmp_path / "no-agn", with_agnostic=False)
    out = tmp_path / "s"
    splits.create_splits(root, out, val_fraction=0.25, seed=1)
    with pytest.raises(DatasetError, match="agnostic-v3.2"):
        VitonHDDataset(root, "train", resolution=TARGET, split_dir=out, agnostic="folder")


def test_missing_image_file_names_the_file(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    person_id = ds[0]["person_id"]
    (dataset_root / "train" / "cloth-mask" / person_id).unlink()
    with pytest.raises(DatasetError, match="cloth-mask"):
        ds[0]


def test_missing_split_files_gives_hint(dataset_root, tmp_path):
    with pytest.raises(DatasetError, match="training.data.splits"):
        VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=tmp_path / "nothing")


def test_missing_root_is_clear_error(tmp_path, split_dir):
    with pytest.raises(DatasetError, match="not found"):
        VitonHDDataset(tmp_path / "nope", "train", resolution=TARGET, split_dir=split_dir)


def test_unknown_split_rejected(dataset_root, split_dir):
    with pytest.raises(ValueError, match="split"):
        VitonHDDataset(dataset_root, "dev", resolution=TARGET, split_dir=split_dir)


def test_same_index_returns_identical_tensors(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    a, b = ds[3], ds[3]
    for key in ("person", "garment", "garment_mask", "agnostic"):
        assert torch.equal(a[key], b[key])


def test_dataloader_iterates_whole_set(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    seen = 0
    for batch in loader:
        assert batch["person"].shape[1:] == (3, *TARGET)
        assert len(batch["person_id"]) == batch["person"].shape[0]
        seen += batch["person"].shape[0]
    assert seen == len(ds)


# ---------------------------------------------------------------- verify


def test_verify_clean_dataset_reports_no_problems(dataset_root):
    assert verify_dataset(dataset_root, expected_size=NATIVE) == []


def test_verify_reports_missing_and_wrong_size(dataset_root):
    (dataset_root / "train" / "cloth" / "00001_00.jpg").unlink()
    Image.new("RGB", (10, 10)).save(dataset_root / "train" / "image" / "00002_00.jpg")
    problems = verify_dataset(dataset_root, expected_size=NATIVE)
    assert any("00001_00.jpg" in p and "cloth" in p for p in problems)
    assert any("00002_00.jpg" in p and "size" in p for p in problems)


# ---------------------------------------------------------------- check CLI


def test_check_cli_succeeds_and_cycles_to_requested_count(dataset_root, split_dir, capsys):
    code = check.main(
        [
            "--root", str(dataset_root), "--split-dir", str(split_dir), "--split", "train",
            "--samples", "30", "--resolution", "32x24", "--batch-size", "4",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "30 samples" in out and "OK" in out


def test_check_cli_fails_on_broken_pair(dataset_root, split_dir, capsys):
    for f in (dataset_root / "train" / "cloth").glob("*.jpg"):
        f.unlink()
    code = check.main(
        [
            "--root", str(dataset_root), "--split-dir", str(split_dir), "--samples", "5",
            "--resolution", "32x24",
        ]
    )
    assert code == 1
    assert "FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------- DressCode stub


def test_dresscode_adapter_is_a_documented_stub(tmp_path):
    with pytest.raises(NotImplementedError, match="DressCode"):
        DressCodeDataset(tmp_path)


def test_paired_mode_gives_each_person_their_own_garment(dataset_root, split_dir):
    ds = VitonHDDataset(dataset_root, "train", resolution=TARGET, split_dir=split_dir)
    assert ds.pairing == "paired"
    for index in range(len(ds)):
        sample = ds[index]
        assert sample["garment_id"] == sample["person_id"]


def test_unpaired_mode_uses_the_listed_garment(dataset_root, split_dir):
    ds = VitonHDDataset(
        dataset_root, "train", resolution=TARGET, split_dir=split_dir, pairing="unpaired"
    )
    listed = dict(splits.read_pairs(split_dir / "train.txt"))
    for index in range(len(ds)):
        sample = ds[index]
        assert sample["garment_id"] == listed[sample["person_id"]]
        assert sample["garment_id"] != sample["person_id"]


def test_unknown_pairing_mode_is_rejected(dataset_root, split_dir):
    with pytest.raises(ValueError, match="pairing"):
        VitonHDDataset(dataset_root, "train", split_dir=split_dir, pairing="mixed")

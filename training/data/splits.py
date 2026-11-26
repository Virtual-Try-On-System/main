"""Deterministic train/val/test split lists for VITON-HD.

The official ``train_pairs.txt`` is split into our ``train`` and ``val`` lists with a fixed seed.
The split is made by person image, so one person never appears in both lists. The official
``test_pairs.txt`` stays the test set unchanged.

Output (plain text, one ``person_image garment_image`` pair per line, no images):

    train.txt  val.txt  test.txt  manifest.json   (manifest has seed, counts and SHA256 of each)

Create the files once the dataset is downloaded::

    uv run python -m training.data.splits --root data/viton-hd
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent / "splits" / "viton_hd"
DEFAULT_SEED = 1234
DEFAULT_VAL_FRACTION = 0.05
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")
MANIFEST = "manifest.json"

Pair = tuple[str, str]


def read_pairs(path: Path) -> list[Pair]:
    """Read a pairs file: one ``person garment`` pair per line, blank lines ignored."""
    pairs: list[Pair] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected 'person garment', got {line!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def find_pairs_file(root: Path, phase: str) -> Path:
    """Locate ``<phase>_pairs.txt`` in the dataset root (or in the phase folder)."""
    name = f"{phase}_pairs.txt"
    for candidate in (Path(root) / name, Path(root) / phase / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{name} not found in {root} (also looked in {Path(root) / phase}). "
        "Is --root the folder that contains train/ and test/?"
    )


def split_pairs(
    pairs: list[Pair], val_fraction: float, seed: int
) -> tuple[list[Pair], list[Pair]]:
    """Split pairs into (train, val) by person image. Same input and seed give same output."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    persons = sorted({person for person, _ in pairs})
    random.Random(seed).shuffle(persons)
    n_val = max(1, round(len(persons) * val_fraction))
    val_persons = set(persons[:n_val])
    train = [p for p in pairs if p[0] not in val_persons]
    val = [p for p in pairs if p[0] in val_persons]
    return train, val


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pairs(path: Path, pairs: list[Pair]) -> None:
    # newline="\n" keeps the file bytes (and the SHA256) identical on Windows and Linux.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for person, garment in pairs:
            handle.write(f"{person} {garment}\n")


def create_splits(
    root: Path,
    out_dir: Path = DEFAULT_OUT,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Write train/val/test lists and a manifest with SHA256 of each file. Returns the manifest."""
    train_all = read_pairs(find_pairs_file(root, "train"))
    test = read_pairs(find_pairs_file(root, "test"))
    train, val = split_pairs(train_all, val_fraction, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contents = {"train.txt": train, "val.txt": val, "test.txt": test}
    files = {}
    for name, pairs in contents.items():
        _write_pairs(out_dir / name, pairs)
        files[name] = {"count": len(pairs), "sha256": _sha256(out_dir / name)}
    manifest = {
        "dataset": "viton-hd",
        "seed": seed,
        "val_fraction": val_fraction,
        "official_train_pairs": len(train_all),
        "files": files,
    }
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def verify_splits(out_dir: Path) -> list[str]:
    """Compare split files with the SHA256 and counts recorded in the manifest."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST
    if not manifest_path.is_file():
        return [f"{manifest_path} is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for name, info in manifest["files"].items():
        path = out_dir / name
        if not path.is_file():
            problems.append(f"{name}: file is missing")
        elif _sha256(path) != info["sha256"]:
            problems.append(f"{name}: SHA256 differs from the manifest (file was edited)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create fixed VITON-HD split files.")
    parser.add_argument("--root", type=Path, required=True, help="folder with train/ and test/")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the lists")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        manifest = create_splits(args.root, args.out, args.val_fraction, args.seed)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote split files to {args.out} (seed {manifest['seed']})")
    for name, info in manifest["files"].items():
        print(f"  {name}: {info['count']} pairs, sha256 {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

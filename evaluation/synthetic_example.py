"""Create synthetic folders to try the comparison without real data.

    uv run python -m evaluation.synthetic_example --root outputs/eval_demo

Makes ``ground_truth``, ``model_a`` (small noise) and ``model_b`` (blurred and shifted).
The images are random smooth patterns, not people. Nothing here is committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_synthetic_folders(root: Path, count: int = 12, size: int = 256, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    folders = {name: Path(root) / name for name in ("ground_truth", "model_a", "model_b")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        low = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        truth = Image.fromarray(low).resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(truth, dtype=np.float32)
        noisy = arr + rng.normal(0, 8, arr.shape)
        model_a = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
        model_b = truth.filter(ImageFilter.GaussianBlur(6)).transform(
            truth.size, Image.Transform.AFFINE, (1, 0, 6, 0, 1, 4)
        )
        name = f"img_{i:03d}.png"
        truth.save(folders["ground_truth"] / name)
        model_a.save(folders["model_a"] / name)
        model_b.save(folders["model_b"] / name)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/eval_demo"))
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    for name, path in make_synthetic_folders(args.root, args.count).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

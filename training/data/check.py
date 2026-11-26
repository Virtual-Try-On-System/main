"""Acceptance check: read N samples through a DataLoader and report timing and shapes.

    uv run python -m training.data.check --root data/viton-hd --samples 1000

Add ``--verify`` to scan every file of the dataset first (missing files, wrong sizes).
Exit code 0 means success, 1 means a problem was found.
"""

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from training.data import splits
from training.data.viton_hd import NATIVE_SIZE, DatasetError, VitonHDDataset, verify_dataset


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use HxW, for example 512x384") from error
    return height, width


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iterate samples through a DataLoader.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split-dir", type=Path, default=splits.DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--resolution", type=_parse_resolution, default=(512, 384))
    parser.add_argument("--agnostic", default="auto", choices=["auto", "folder", "parse"])
    parser.add_argument(
        "--pairing",
        default="paired",
        choices=["paired", "unpaired"],
        help="paired: own garment (for training); unpaired: listed garment",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--verify", action="store_true", help="scan all files first")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify_dataset(args.root, expected_size=NATIVE_SIZE)
        print(f"verify: {len(problems)} problem(s)")
        for problem in problems[:50]:
            print(f"  {problem}")
        if problems:
            print("FAILED: dataset has problems")
            return 1

    problems = splits.verify_splits(args.split_dir)
    if problems:
        for problem in problems:
            print(f"split files: {problem}")
        print(
            "FAILED: split files are missing or differ from the manifest. Create them with: "
            f"uv run python -m training.data.splits --root {args.root}"
        )
        return 1

    try:
        dataset = VitonHDDataset(
            args.root,
            args.split,
            args.resolution,
            args.agnostic,
            args.split_dir,
            pairing=args.pairing,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        seen = 0
        shapes: dict[str, tuple[int, ...]] = {}
        start = time.perf_counter()
        while seen < args.samples:  # cycle through the split if it has fewer samples
            for batch in loader:
                shapes = {k: tuple(v.shape[1:]) for k, v in batch.items() if hasattr(v, "shape")}
                seen += batch["person"].shape[0]
                if seen >= args.samples:
                    break
        seconds = time.perf_counter() - start
    except (DatasetError, OSError, ValueError) as error:
        print(f"FAILED: {error}")
        return 1

    print(f"split {args.split}: {len(dataset)} pairs, agnostic from {dataset.agnostic_source}")
    print(f"shapes per sample: {shapes}")
    print(f"read {seen} samples in {seconds:.1f} s ({seen / seconds:.1f} samples/s)")
    print(f"OK: {args.samples} samples read without error" if seen >= args.samples else "FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare the output folders of several models against one reference folder.

Example:
    uv run python -m evaluation.compare --reference data/test/ground_truth \
        --model ours=outputs/ours --model fashn=outputs/fashn --out outputs/eval/table.csv

Images are matched by file name. The table is printed, and written as CSV, Markdown
and JSON (settings, library versions, warnings) next to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evaluation.metrics import (
    RESOLUTION,
    SEED,
    Metric,
    PairingError,
    default_metrics,
    list_metrics,
    pair_folders,
)

LIBRARIES = ["numpy", "Pillow", "scikit-image", "scipy", "torch", "lpips", "torchmetrics"]


@dataclass
class ComparisonTable:
    rows: list[dict]
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        cols: list[str] = []
        for row in self.rows:
            cols += [k for k in row if k not in cols]
        return cols


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions


def set_determinism(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_metrics(fid_min_samples: int | None = None) -> dict[str, Metric]:
    """Built-in metrics (real pretrained networks, downloaded on first use) plus any
    metric added with ``register_metric``."""
    metrics = default_metrics(fid_min_samples=fid_min_samples)
    for name, metric in list_metrics().items():
        metrics.setdefault(name, metric)
    return metrics


def run_comparison(
    reference: Path,
    models: dict[str, Path],
    metrics: dict[str, Metric] | None = None,
    allow_mismatch: bool = False,
    resolution: tuple[int, int] = RESOLUTION,
) -> ComparisonTable:
    """Score every model folder against the reference folder with every metric."""
    metrics = metrics if metrics is not None else build_metrics()
    set_determinism()
    rows, warnings = [], []
    for model_name, model_dir in models.items():
        pairs = pair_folders(
            reference, model_dir, allow_mismatch=allow_mismatch, resolution=resolution
        )
        row: dict = {"model": model_name, "n_images": len(pairs)}
        if pairs.missing or pairs.extra:
            warnings.append(
                f"{model_name}: scored {len(pairs)} common images; "
                f"{len(pairs.missing)} missing from model, {len(pairs.extra)} extra in model."
            )
        for metric in metrics.values():
            result = metric.compute(pairs)
            row.update(result.values)
            warnings += [f"{model_name}: {w}" for w in result.warnings]
        rows.append(row)
    metadata = {
        "reference": str(reference),
        "models": {name: str(path) for name, path in models.items()},
        "resolution_hw": list(resolution),
        "resize_filter": "bicubic",
        "seed": SEED,
        "allow_mismatch": allow_mismatch,
        "metric_settings": {name: dict(m.settings) for name, m in metrics.items()},
        "versions": library_versions(),
    }
    return ComparisonTable(rows, warnings, metadata)


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def format_text(table: ComparisonTable) -> str:
    cols = table.columns
    cells = [cols] + [[_fmt(r.get(c, "")) for c in cols] for r in table.rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_markdown(table: ComparisonTable) -> str:
    cols = table.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in table.rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_outputs(table: ComparisonTable, out: Path) -> list[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table.columns)
        writer.writeheader()
        writer.writerows(table.rows)
    md = out.with_suffix(".md")
    md.write_text(format_markdown(table), encoding="utf-8")
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({**table.metadata, "warnings": table.warnings}, indent=2) + "\n",
        encoding="utf-8",
    )
    return [out, md, meta]


def _parse_resolution(text: str) -> tuple[int, int]:
    height, sep, width = text.lower().partition("x")
    if not sep or not height.isdigit() or not width.isdigit() or "0" in (height, width):
        raise argparse.ArgumentTypeError(f"--resolution must look like HEIGHTxWIDTH, got: {text}")
    return int(height), int(width)


def _parse_model(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--model must look like NAME=FOLDER, got: {text}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare model output folders (SSIM, LPIPS, FID).")
    parser.add_argument("--reference", type=Path, required=True, help="ground truth folder")
    parser.add_argument("--model", type=_parse_model, action="append", required=True,
                        metavar="NAME=FOLDER", help="model output folder (repeat for more models)")
    parser.add_argument("--out", type=Path, help="CSV path; .md and .json are written beside it")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="score only names present in both folders (reported as a warning)")
    parser.add_argument("--resolution", type=_parse_resolution, default=RESOLUTION,
                        metavar="HxW", help="size every image is resized to (default 256x192)")
    parser.add_argument("--fid-min-samples", type=int, default=None,
                        help="warn about FID below this many images (default 2048)")
    args = parser.parse_args(argv)

    names = [name for name, _ in args.model]
    if len(set(names)) != len(names):
        parser.error("model names must be unique")
    try:
        table = run_comparison(
            args.reference,
            dict(args.model),
            metrics=build_metrics(fid_min_samples=args.fid_min_samples),
            allow_mismatch=args.allow_mismatch,
            resolution=args.resolution,
        )
    except PairingError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(format_text(table))
    for warning in table.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if args.out:
        for path in write_outputs(table, args.out):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

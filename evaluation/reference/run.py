"""CLI: python -m evaluation.reference.run --inputs manifest.csv --out outputs/reference/fashn"""

import argparse
import sys

from evaluation.reference.base import ReferenceModel
from evaluation.reference.fashn import FashnReference, ReferenceModelUnavailable
from evaluation.reference.manifest import ManifestError, read_manifest
from evaluation.reference.runner import run_reference


def main(argv: list[str] | None = None, model: ReferenceModel | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reference try-on model on a manifest.")
    parser.add_argument("--inputs", required=True, help="manifest CSV")
    parser.add_argument("--out", default="outputs/reference/fashn", help="output folder")
    parser.add_argument("--weights-dir", default="weights/fashn", help="FASHN weights folder")
    parser.add_argument("--device", default=None, help="cuda, cpu or empty for auto")
    parser.add_argument("--garment-photo-type", default="flat-lay", choices=["model", "flat-lay"])
    args = parser.parse_args(argv)

    try:
        rows = read_manifest(args.inputs)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if model is None:
        model = FashnReference(
            weights_dir=args.weights_dir,
            device=args.device,
            garment_photo_type=args.garment_photo_type,
        )
    try:
        records = run_reference(model, rows, args.out)
    except ReferenceModelUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"wrote {len(records)} outputs, timings.csv and run.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

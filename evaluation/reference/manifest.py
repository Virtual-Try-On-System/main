"""Input manifest for reference runs.

CSV columns: person, garment, cut, category. Optional column: name (output file name).
Relative paths are resolved against the folder that holds the manifest.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from evaluation.reference.base import CATEGORIES, CUTS

REQUIRED = ("person", "garment", "cut", "category")


class ManifestError(ValueError):
    """The manifest is malformed or points to missing files."""


@dataclass(frozen=True)
class ManifestRow:
    person: Path
    garment: Path
    cut: str
    category: str
    output_name: str


def read_manifest(path: str | Path) -> list[ManifestRow]:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    base = path.parent
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest is missing column(s): {', '.join(missing)}")
        rows = []
        seen: set[str] = set()
        for line, raw in enumerate(reader, start=2):
            row = _parse_row(raw, base, line)
            if row.output_name in seen:
                raise ManifestError(f"line {line}: duplicate output name {row.output_name}")
            seen.add(row.output_name)
            rows.append(row)
    if not rows:
        raise ManifestError("manifest has no rows")
    return rows


def _resolve(value: str, base: Path) -> Path:
    p = Path(value.strip())
    return p if p.is_absolute() else base / p


def _parse_row(raw: dict, base: Path, line: int) -> ManifestRow:
    cut = (raw["cut"] or "").strip()
    category = (raw["category"] or "").strip()
    if cut not in CUTS:
        raise ManifestError(f"line {line}: cut must be one of {CUTS}, got {cut!r}")
    if category not in CATEGORIES:
        raise ManifestError(f"line {line}: category must be one of {CATEGORIES}, got {category!r}")
    person = _resolve(raw["person"] or "", base)
    garment = _resolve(raw["garment"] or "", base)
    for file in (person, garment):
        if not file.is_file():
            raise ManifestError(f"line {line}: file not found: {file}")
    name = (raw.get("name") or "").strip() or f"{person.stem}__{garment.stem}.png"
    return ManifestRow(person, garment, cut, category, name)

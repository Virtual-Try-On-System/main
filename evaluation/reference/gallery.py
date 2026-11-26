"""Build an HTML gallery: person, garment and output side by side, grouped by cut.

Marks CSV columns: output, hem_shifted (yes/no), note. Rows without a mark show "unmarked".
CLI: python -m evaluation.reference.gallery --run outputs/reference/fashn --marks marks.csv
"""

import argparse
import csv
import html
from pathlib import Path

from evaluation.reference.base import CUTS

MARK_VALUES = ("yes", "no")


def read_marks(path: str | Path) -> dict[str, tuple[str, str]]:
    """Return {output name: (hem_shifted, note)}."""
    marks = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            value = (raw.get("hem_shifted") or "").strip().lower()
            if value not in MARK_VALUES:
                raise ValueError(f"hem_shifted must be yes or no, got {value!r} for {raw}")
            marks[raw["output"].strip()] = (value, (raw.get("note") or "").strip())
    return marks


def _uri(path) -> str:
    return html.escape(Path(path).resolve().as_uri(), quote=True)


def build_gallery(
    run_dir: str | Path, out_html: str | Path, marks_csv: str | Path | None = None
) -> Path:
    run_dir, out_html = Path(run_dir), Path(out_html)
    with (run_dir / "timings.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    marks = read_marks(marks_csv) if marks_csv and Path(marks_csv).is_file() else {}

    parts = [
        "<!doctype html><meta charset='utf-8'><title>Reference model gallery</title>",
        "<style>body{font-family:sans-serif;margin:1.5rem}"
        ".row{display:flex;gap:1rem;align-items:flex-start;margin-bottom:1.5rem}"
        "figure{margin:0}img{max-height:320px}"
        ".mark{padding:.2rem .5rem;border-radius:4px}.yes{background:#f8c4c4}"
        ".no{background:#c9eccb}.unmarked{background:#eee}</style>",
        "<h1>Reference model gallery</h1>",
    ]
    for cut in CUTS:
        cut_rows = [r for r in rows if r["cut"] == cut]
        parts.append(f"<h2>{cut} ({len(cut_rows)})</h2>")
        for r in cut_rows:
            value, note = marks.get(r["output"], ("", ""))
            css = value or "unmarked"
            label = f"hem line shifted: {value or 'unmarked'}"
            stats = f"{r['seconds']} s, {r['peak_vram_mb'] or 'n/a'} MB"
            parts.append(
                "<div class='row'>"
                f"<figure><img src='{_uri(r['person'])}'><figcaption>person</figcaption></figure>"
                f"<figure><img src='{_uri(r['garment'])}'><figcaption>garment</figcaption></figure>"
                f"<figure><img src='{_uri(run_dir / r['output'])}'>"
                f"<figcaption>{html.escape(r['output'])}<br>{stats}</figcaption></figure>"
                f"<div><span class=\"mark {css}\">{label}</span><p>{html.escape(note)}</p></div>"
                "</div>"
            )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(parts), encoding="utf-8")
    return out_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the reference model gallery.")
    parser.add_argument("--run", default="outputs/reference/fashn", help="folder from run.py")
    parser.add_argument("--marks", default=None, help="marks CSV (output,hem_shifted,note)")
    parser.add_argument("--out", default=None, help="HTML path (default: <run>/gallery.html)")
    args = parser.parse_args(argv)
    out = args.out or Path(args.run) / "gallery.html"
    print(f"wrote {build_gallery(args.run, out, args.marks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

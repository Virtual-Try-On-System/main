"""Labeler agreement measurement (Goal 2 acceptance criterion).

Input: garment/artifacts/kappa_labeling_A.csv and kappa_labeling_B.csv
       Columns: image_id,length,fit,sleeve (empty cell = no label, pair excluded)
Ordinal axes (length, fit, sleeve) use quadratic-weighted Cohen's kappa,
nominal axes use plain Cohen's kappa. Pure Python, no external dependencies.
Output: garment/taxonomy/kappa_report.md
Run: python -m garment.data_prep.cohen_kappa
"""

import csv
import json
from collections import Counter

from .paths import ARTIFACTS, ROOT

TAXONOMY = ROOT / "garment" / "taxonomy" / "taxonomy.json"
REPORT = ROOT / "garment" / "taxonomy" / "kappa_report.md"
AXES = ("length", "fit", "sleeve", "neckline", "closure")


def kappa(pairs, classes, weighted):
    """pairs: [(labelA, labelB), ...]; classes: class labels (ordered if ordinal)."""
    index = {c: i for i, c in enumerate(classes)}
    invalid = [(a, b) for a, b in pairs if a not in index or b not in index]
    pairs = [(a, b) for a, b in pairs if a in index and b in index]
    n = len(pairs)
    if n == 0:
        return None, 0, len(invalid)
    k = len(classes)
    observed = [[0] * k for _ in range(k)]
    for a, b in pairs:
        observed[index[a]][index[b]] += 1
    a_marginal = Counter(a for a, _ in pairs)
    b_marginal = Counter(b for _, b in pairs)
    if weighted:
        po = 0.0
        pe = 0.0
        for i in range(k):
            for j in range(k):
                w = ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
                po += w * observed[i][j] / n
                pe += w * (a_marginal[classes[i]] / n) * (b_marginal[classes[j]] / n)
        if pe == 1.0:
            return 1.0, n, len(invalid)
        return 1.0 - po / (1.0 - pe), n, len(invalid)
    po = sum(observed[i][i] for i in range(k)) / n
    pe = sum((a_marginal[c] / n) * (b_marginal[c] / n) for c in classes)
    if pe == 1.0:
        return 1.0, n, len(invalid)
    return (po - pe) / (1.0 - pe), n, len(invalid)


def main():
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    axes = taxonomy["axes"]

    def read(person):
        path = ARTIFACTS / f"kappa_labeling_{person}.csv"
        with open(path, encoding="utf-8-sig") as f:
            return {r["image_id"]: r for r in csv.DictReader(f)}

    a_labels, b_labels = read("A"), read("B")
    shared = sorted(set(a_labels) & set(b_labels))
    lines = ["# Labeler Agreement Report (Cohen's Kappa)", ""]
    lines.append(f"Shared images: {len(shared)} "
                 "(target: 20 shared images, two independent labelers)")
    lines.append("")
    lines.append("| Axis | Kappa type | Kappa | Pair count | Excluded (invalid/empty) | Target |")
    lines.append("|---|---|---|---|---|---|")
    for axis in AXES:
        if axis not in axes:
            continue
        classes = [c["label"] for c in axes[axis]["classes"]]
        ordinal = axes[axis]["ordinal"]
        pairs = []
        empty = 0
        for image_id in shared:
            a = a_labels[image_id].get(axis, "").strip()
            b = b_labels[image_id].get(axis, "").strip()
            if not a or not b:
                empty += 1
                continue
            pairs.append((a, b))
        value, n, invalid = kappa(pairs, classes, ordinal)
        kind = "weighted (quadratic)" if ordinal else "plain"
        value_str = f"{value:.3f}" if value is not None else "not computable"
        meets = value is not None and value >= 0.6
        target = ">= 0.6" + (" OK" if meets else " (BELOW/NOT MEASURED)")
        lines.append(f"| {axis} | {kind} | {value_str} | {n} | {empty + invalid} | {target} |")
    lines.append("")
    lines.append("Note: empty cell = no label; the pair is excluded for that axis. "
                 "Kappa < 0.6 means the guide is revised and the measurement repeated.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nReport written:", REPORT)


if __name__ == "__main__":
    main()

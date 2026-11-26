# Garment Analysis Module (Task 3)

From a garment image (flat-lay or on-model), this module produces the **category,
multi-axis cut classes, segmentation mask and ratio vector** of a garment, plus the
model-side preparation for try-on training.

This PR covers **Goal 1 (Garment Descriptor contract + mock examples), Goal 2 (cut
taxonomy + labeling guide) and Goal 3 (dataset selection and preparation)**.

## Status

| Goal | Deliverable | Status |
|---|---|---|
| 1 — Descriptor contract | `descriptor.py`, `schema/garment_descriptor.schema.json`, `examples/` (5 valid + 6 invalid), `tests/test_garment_descriptor.py` | Tests pass (valid accepted / invalid rejected / taxonomy + JSON schema in sync) |
| 2 — Taxonomy + guide | `taxonomy/taxonomy.json`, `taxonomy/labeling_guide.md`, `data_prep/cohen_kappa.py` | Written; kappa measurement pending two-person labeling (`artifacts/kappa_labeling_A/B.csv`) |
| 3 — Dataset prep | `artifacts/mapping_*.csv`, `class_counts.csv`, fixed splits, `RARE_CLASS_DECISIONS.md`, `LICENSE_NOTES.md` | Split overlaps empty, SHA256 recorded |

Next: Goal 5 (zero-shot baseline), Goal 4 (domain gap), Goals 6/9 (training), Goals 7–8 (mask + ratio vector).

## The descriptor contract

`descriptor.py` is the single source of truth (Pydantic, strict, `extra="forbid"`).
`schema/garment_descriptor.schema.json` is generated from it (`python -m garment.descriptor`)
and a test fails if the two diverge.

- `category`: `tshirt`, `sweatshirt`, `shirt` (+ confidence).
- `cut`: per-axis predictions. **Ordinal** axes: `length` (crop/short/regular/long),
  `fit` (tight/regular/loose/oversize), `sleeve` (sleeveless/short/three_quarter/long).
  **Nominal, auxiliary** axes: `neckline`, `closure` (`ordinal: false`).
  Every prediction carries `confidence` and a `low_confidence` flag.
- `mask`: inline `polygon`/`rle` data or a relative `png_path` (absolute paths rejected).
- `ratios`: `length_to_chest_width`, `sleeve_to_chest_width`, `shoulder_to_chest_width`.
  Only meaningful on flat-lay images; on `on_model` they must be null with `valid: false`.
- `size_label`: `XS`–`XXL` or `unknown` (size chart is a later deliverable).
- `domain`: `flat` / `on_model`; `runtime_ms`; global `low_confidence`.

### Relationship to `schemas/garment.py` (Task 2, `feature/json-schemas`)

This contract answers the open questions in their `docs/schemas.md`: it replaces the single
`fit_class` with the three ordinal axes, adds confidences, and fixes the ratio vector to
length/sleeve/shoulder **to chest width** (their draft used `sleeve_to_length`).
`schema_version` stays `1.0` until the team locks the schema; Goal 1 acceptance requires
joint approval with Task 2 before their `schemas/garment.py` is updated to match.

## Fixed splits (seed=42, `artifacts/split_sha256.txt`)

| File | Rows | Content |
|---|---|---|
| `train.csv` | 53,123 | Fashionpedia 19,302 masks + Shopping100k 33,821 images |
| `val.csv` | 10,656 | Fashionpedia 3,407 + Shopping100k 7,249 |
| `test.csv` | 7,830 | Fashionpedia 579 (official val2020) + Shopping100k 7,251 |
| `val_flat.csv` | 201 | Golden flat pool (Shopping100k, products excluded from training) |
| `test_flat.csv` | 199 | Golden flat pool (Shopping100k, products excluded from training) |

- Fashionpedia: official train2020 -> 85/15 train/val (image-level, class-stratified);
  official val2020 -> test. Row unit = upper-garment mask (bbox included for cropping).
- Shopping100k: **product-group** split 70/15/15 (all views of a product stay together).
- Golden flat pool: 400 images, one per product, class-balanced; their labels are
  **pre-labels requiring two-person manual correction** (see `label_source` column).
- Acceptance metrics are computed on `test_flat`; `test.csv` measures the domain gap.

## Key dataset decisions

- **Datasets:** Fashionpedia (main, on-model) + Shopping100k (supplement, flat) + a golden
  flat pool split out of Shopping100k. The `dress_code` folder seen earlier is a different
  Roboflow person-detection set and is **not used**.
- **Fashionpedia has t-shirt and sweatshirt in ONE category** (`top, t-shirt, sweatshirt`);
  the split uses `nickname=hoodie`. `sweater`/`cardigan` are out of scope.
- **Sleeve-length labels effectively do not exist on Fashionpedia upper-garment masks**
  (0.1%; sleeve parts carry zero sleeve labels) -> the sleeve head learns from
  Shopping100k + manual labels. Neckline labels are joined from neckline part masks,
  only on single-upper-garment images (12,288 images).
- **Shopping100k has no `length` attribute** -> the length axis relies on Fashionpedia
  + manual labels. Trouser-specific `fit` values are OUT.
- Missing labels stay empty and are masked out of the loss (never treated as `none`).

Details: `artifacts/VERIFICATION_REPORT.md`, `artifacts/RARE_CLASS_DECISIONS.md`.

## Reproduce

See `data_prep/README.md`. Tests: `uv run pytest` (or `pytest` with the repo on the path).

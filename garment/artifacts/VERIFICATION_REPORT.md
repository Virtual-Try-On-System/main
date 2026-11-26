# Step 0 Verification Report (Task 3, Phase A)

Date: 2026-11-07

## 1. Fashionpedia annotation counts
- Source: official CVDF/S3 annotations (`instances_attributes_train2020.json`,
  `instances_attributes_val2020.json`) downloaded.
- 46 categories, 294 attributes; train: 45,623 images / 333,401 masks; official val:
  1,158 images / 8,781 masks.
- **T-shirt/sweatshirt split:** `top, t-shirt, sweatshirt` is a **single category** (id=1).
  The split uses the `nickname` attribute: `hoodie`(16) -> sweatshirt (420 train masks),
  everything else -> tshirt. `sweater`(2) and `cardigan`(3) are separate categories and
  out of scope (OUT).
- Upper-body mask counts (train/official val): shirt,blouse 6,161/102;
  top,t-shirt,sweatshirt 16,548/477; sweater 1,494/21; cardigan 1,107/12.
- Axis label coverage (main garment mask, train): length 73-95%, fit 63-87%, closure on
  shirts 91%; **sleeve ~0.1% and neckline ~0%** -> neckline labels live on neckline part
  masks (34,258 masks; usable on 12,288 single-upper-garment images), and **sleeve labels
  effectively do not exist** (874 sleeve part masks, zero sleeve labels).

## 2. Shopping100k verification
- 100,586 images + 12 attribute files local, no missing files on disk.
- **Flat/person-free check:** random 100 images (seed=42) inspected via a contact sheet
  (kept in `outputs/`, not committed): **100/100 flat and person-free**; ~3 images contain
  multi-garment sets.
- Target categories: Shirt 14,853, T-shirt 22,071, Jumper 11,797 (48,721 total).
- **No `length` attribute** -> the length axis relies on Fashionpedia + manual labels
  (a correction to the draft mapping in the planning document).
- Trouser-specific `fit` values are OUT; `sleevelen` 9 values reduced to 4 classes.
- License: Readme says "non-commercial research/education only" (see `LICENSE_NOTES.md`).

## 3. Dress Code
- The local `dress_code/` folder is not the real Dress Code; it is a Roboflow
  person-detection set. Not used. The golden flat pool was split out of Shopping100k
  (400 images, excluded from training).

## 4. Fixed splits produced (seed=42)
| File | Rows | Content |
|---|---|---|
| train.csv | 53,123 | FP 19,302 masks + S100k 33,821 images |
| val.csv | 10,656 | FP 3,407 + S100k 7,249 |
| test.csv | 7,830 | FP 579 (official val2020) + S100k 7,251 |
| val_flat.csv | 201 | Golden flat pool (S100k, products excluded from training) |
| test_flat.csv | 199 | Golden flat pool (S100k, products excluded from training) |

- FP: official train2020 -> 85/15 train/val (image-level, class-stratified);
  official val2020 -> test.
- S100k: product-group 70/15/15; golden-pool products fully removed from train/val/test.
- Overlap assertions passed; SHA256 digests in `split_sha256.txt`.

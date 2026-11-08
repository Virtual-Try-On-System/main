# Data preparation scripts (Task 3)

Offline utilities that produce every artifact in `garment/artifacts/`.
Run them **from the repository root** with `python -m`.

## Local data layout

Datasets are never committed. Scripts read them from `<repo>/data/` by default;
override with the `VTON_DATA_DIR` environment variable (e.g. `VTON_DATA_DIR=G:\VTON\datasets`).

```
data/
  fashionpedia/
    train/                         # 45,623 images
    val_test2020/test/             # official val+test images
    annotations/
      instances_attributes_train2020.json
      instances_attributes_val2020.json
  Shopping100k/
    Images/                        # 100,586 images
    Attributes/                    # 12 attribute txt files
```

Download links and licenses: `../artifacts/LICENSE_NOTES.md`.

## Commands (in order)

```bash
# Phase A: verification and counts
python -m garment.data_prep.analyze_fashionpedia          # categories, attributes, axis coverage
python -m garment.data_prep.analyze_fashionpedia_parts    # sleeve/neckline part-mask analysis
python -m garment.data_prep.analyze_shopping100k          # S100k distributions + 100-image check

# Phase C1: mapping tables (single source of truth for label mappings)
python -m garment.data_prep.generate_mappings

# Phase C2/C4: apply mappings, class counts, fixed splits (seed=42, overlap asserts, SHA256)
python -m garment.data_prep.prepare_data

# Goal 1: (re)generate the JSON schema artifact from the Pydantic model
python -m garment.descriptor

# Goal 2: labeler agreement, after the two-person labeling of the 20 shared images
python -m garment.data_prep.cohen_kappa                   # writes garment/taxonomy/kappa_report.md
```

## Notes

- Only `analyze_shopping100k` needs **Pillow**, imported lazily for the contact sheet
  (`pip install pillow`). Everything else uses the standard library.
- The contact sheet and any other images are written to `outputs/` (git-ignored);
  they must never be committed (CI blocks image files).
- Re-running `prepare_data` with the same seed reproduces identical splits;
  `artifacts/split_sha256.txt` records the digests.
- Split CSVs use English class labels that match `garment/descriptor.py`
  (`tshirt/sweatshirt/shirt`, `length/fit/sleeve/neckline/closure` axes).

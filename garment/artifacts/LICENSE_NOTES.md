# License Notes (Task 3)

All images stay local. Person and garment images are never committed; the cloud is used
only for non-personal model tests whose licenses allow it.

## Fashionpedia (main dataset)
- **License:** CC BY 4.0 (per the Hugging Face dataset card; official page:
  https://fashionpedia.github.io — image copyrights should be double-checked there).
- **Local content:** 45,623 train + 3,200 val/test images; annotations
  `annotations/instances_attributes_train2020.json` (542 MB) and
  `instances_attributes_val2020.json` (14.5 MB), downloaded from the official CVDF/S3
  source (https://s3.amazonaws.com/ifashionist-dataset/annotations/...).
- **Use:** category/mask pre-training, length/fit/neckline axes (on-model domain).
- The official val2020 (1,158 labeled images) is our **test** split; the official test
  images have no public labels and are unused.

## Shopping100k (supplement)
- **License:** the dataset Readme states explicitly: *"we do not own the copyrights to these
  images. Their use is RESTRICTED to non-commercial research and educational purposes."*
  -> **Non-commercial research/education only.**
- **Source verification:** no official download page/license URL was found; contact in the
  Readme: Kenan Ak (emir.ak@u.nus.edu), citations: Ak et al., WACV 2018 + CVPR 2018.
  Any commercial use requires written clarification first.
- **Local content:** 100,586 images (none missing on disk), 12 attribute files.
  There is **no `length` attribute** (the length axis cannot be labeled from this source).
- **Flat/person-free verification:** a random 100-image sample (seed=42) was inspected:
  **100/100 flat-lay and person-free**; ~3 images contain multi-garment sets (noise note).
- **Use:** flat-domain training; category, fit, sleeve, neckline, closure axes; the golden
  flat val/test pool (400 images, excluded from training).

## Dress Code
- The `dress_code/` folder seen in local storage is **not the Dress Code dataset** planned
  in the task documents (aimagelab, 15,366 pairs, flat upper-garment images). Its content is
  the Roboflow "Men Women Dresscode 2" set
  (https://universe.roboflow.com/legiontech/men-women-dresscode-2): 4 YOLO detection
  classes, CC BY 4.0 per the Roboflow page.
- **Decision:** that folder is **not used** for Task 3. No access application was made for
  the real Dress Code; the flat-lay val/test pool was split out of **Shopping100k** instead
  (team decision).

## General warning
If this project becomes a commercial product or a customer delivery, **the Shopping100k
license must be clarified in writing before that use.** For academic/course use the current
usage is appropriate.

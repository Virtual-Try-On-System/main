# Data card: try-on training data

Status: code is ready and tested on synthetic data. The real datasets are **not downloaded yet**, so
every number marked "to fill" must be measured after download. Person and garment images stay on
local disk under `data/` (git-ignored) and are never committed or synced to the cloud.

## Datasets

| | VITON-HD | DressCode |
|---|---|---|
| Use in this project | Main training and evaluation data | Optional extra data, upper body only |
| Content | Frontal woman photos with top garments | Person and garment pairs, 3 categories (upper body, lower body, dresses) |
| Image count | 11,647 train and 2,032 test person images, each with its own garment (checked on 26.11.2026) | 53,792 garments and 107,584 images in total (per the authors); upper-body count to fill |
| Resolution | 1024x768 (height x width) | 1024x768 |
| Download size | 4.5 GB (`zalando-hd-resized.zip`) | to fill |
| Access | Public Google Drive link in the official repository | Request form, institutional email and signed agreement required; no private companies |
| Access status | Not downloaded yet | Not requested yet |
| License | Creative Commons BY-NC 4.0: research and non-commercial use, redistribution allowed with credit | Own academic non-commercial license (LICENCE file), see below |

### Licenses and what they mean for us

- **VITON-HD**: CC BY-NC 4.0. Use is limited to non-commercial purposes, with citation of the paper
  (Choi et al., "VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization",
  CVPR 2021). Redistribution is allowed under the same terms, but the dataset collection is stated
  as "for research purposes only". Keep it on our own training machines; do not publish it.
- **DressCode**: academic non-commercial use only (research, teaching, publication). It may be
  shared unmodified only with the license attached. Do not alter the garments, do not try to identify
  the models, derivative works carry the same restrictions. The dataset is not released to private
  companies. Cite Morelli et al., "Dress Code: High-Resolution Multi-Category Virtual Try-On",
  ECCV 2022 and include the attribution text required by the LICENCE file. The authors can withdraw
  the license.
- **Cloud**: renting a GPU machine to train is a use by us, not a redistribution, but the data
  should be copied to that machine only for the training run and deleted afterwards. The images
  must not go into the cloud sync (see `training/sync.py` allow list). Ask the instructor if the
  DressCode agreement forbids third-party servers.
- Any commercial use of a model trained on this data is not covered by these licenses. Confirm
  this with the team before any non-academic use.

## Layout the loader expects (VITON-HD, `zalando-hd-resized`)

```
data/viton-hd/
  train_pairs.txt   test_pairs.txt            "person.jpg cloth.jpg" per line (unpaired)
  train/ and test/
    image/            00000_00.jpg            person photo
    cloth/            00000_00.jpg            garment photo
    cloth-mask/       00000_00.jpg            garment mask
    image-parse-v3/   00000_00.png            person label map
    agnostic-v3.2/    00000_00.jpg            person with garment and arms hidden
    (also in the release: agnostic-mask/, image-densepose/, image-parse-agnostic-v3.2/,
     openpose_img/, openpose_json/, and train/gt_cloth_warped_mask/; not used by the loader)
```

Checked on the real archive on 26.11.2026: this layout is exact, every folder in `train/` has
11,647 files and every folder in `test/` has 2,032, and `--verify` found no missing, unreadable or
wrongly sized file. The folder names and parse labels also match HR-VITON's official loader.

**The published pair lists are unpaired.** Each person is listed with another person's garment:
only 1 of 11,647 train lines and none of the 2,032 test lines pair a person with their own garment.
There is no photo of a person wearing another person's garment, so such a pair has no target
image. The loader therefore has two modes:

- `pairing="paired"` (default): each person gets their own garment (`image/X.jpg` with
  `cloth/X.jpg`). Use it for training and for SSIM/LPIPS, which need a target image.
- `pairing="unpaired"`: the garment from the list. Use it to try other garments on and for FID. If `agnostic-v3.2/` is missing, `--agnostic parse` builds
the agnostic image from the parse map by painting upper clothes, dress, coat and both arms gray.
This is an approximation of the official agnostic image.

DressCode layout (documented by the authors and community loaders, **not verified**, upper body only):
`upper_body/images/<id>_0.jpg` (person), `<id>_1.jpg` (garment), `label_maps/<id>_4.png`,
`keypoints/<id>_2.json`, and `train_pairs.txt` / test pair lists in the category folder. The adapter
`training/data/dresscode.py` is a stub until the data arrives. DressCode has no ready cloth mask or
agnostic image, so both must be derived from the label maps.

## What the loader returns

| Key | Shape and type | Meaning |
|---|---|---|
| `person` | `(3, H, W)` float32 in [-1, 1] | Person photo |
| `garment` | `(3, H, W)` float32 in [-1, 1] | Garment photo |
| `garment_mask` | `(1, H, W)` float32, values 0 or 1 | Garment mask |
| `agnostic` | `(3, H, W)` float32 in [-1, 1] | Person with garment region hidden (gray = 0.0) |
| `person_id`, `garment_id` | str | File names |

Normalisation is `pixel / 127.5 - 1`. Default resolution is 512x384 (height x width); full size is
1024x768 (`--resolution 1024x768`). Images use bicubic resizing, the mask uses bilinear resizing
and a threshold at 0.5. Nothing is random.

## Split files

Fixed seed 1234. The official `train_pairs.txt` is split by person into `train.txt` and `val.txt`
(5% of persons go to val). The official `test_pairs.txt` is copied unchanged as `test.txt`.
Files live in `training/data/splits/viton_hd/` as plain text lists of file names, with a
`manifest.json` that records seed, counts and the SHA256 of each file.

| File | Content | Count | SHA256 |
|---|---|---|---|
| `train.txt` | official train pairs minus val | 11,065 | `c029a4e739206dbbcd50c648a59e92c8210008038dd705c5bba122dbc946e978` |
| `val.txt` | 5% of train persons | 582 | `0fb65f28992d65fb08b7a0214584c92e22f3a6309d3ba95c4fc62dd7040dd109` |
| `test.txt` | official test pairs | 2,032 | `a54173857f2234ed53202e04a126f1ad9679a695f7103b398110e68b5dc052bf` |

The split files were created from the real release on 26.11.2026 and are committed in
`training/data/splits/viton_hd/`. They hold file names only, no images. Recreate them with:

```
uv run python -m training.data.splits --root data/viton-hd
```

## Acceptance check

Passed on 26.11.2026: `--verify` found 0 problems, and 1000 train samples at 512x384 were read
without error in 47.6 s (21 samples per second, one process, laptop disk).

```
uv run python -m training.data.check --root data/viton-hd --verify --samples 1000
uv run python -m training.data.check --root data/viton-hd --samples 1000 --resolution 1024x768
```

`--verify` scans every file in both phases for missing files, unreadable images and sizes other than
1024x768. The second run reads 1000 samples through a PyTorch `DataLoader` and prints shapes and
speed. Add `--workers 4` to test with worker processes.

## Still unknown

- SHA256 of the downloaded archive.
- DressCode access decision, real layout and upper-body counts.

## Questions for the model lead

- Which input resolution does the model need: 512x384, 1024x768, or both?
- Which normalisation does the model expect: [-1, 1] (current) or [0, 1]?
- Is the official agnostic image enough, or does the model need extra inputs (DensePose, pose
  keypoints, parse map)? The loader can add them.
- Should DressCode be used at all, given the access process and agreement?
- Is a 5% validation share (by person) acceptable?

## Sources

- VITON-HD repository, dataset link and license: https://github.com/shadow2496/VITON-HD
- VITON-HD paper: https://arxiv.org/abs/2103.16874
- StableVITON README (folder layout for VITON-HD): https://github.com/rlawjdghek/StableVITON
- HR-VITON (same dataset layout variant): https://github.com/sangyun884/HR-VITON
- DressCode repository, access rules and annotations: https://github.com/aimagelab/dress-code
- DressCode license: https://github.com/aimagelab/dress-code/blob/main/LICENCE
- DressCode paper: https://arxiv.org/abs/2204.08532

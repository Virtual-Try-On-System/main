# Reference model: FASHN VTON v1.5

Task 6 compares our try-on output with an existing open-weight model. FASHN VTON v1.5 is a
**comparison baseline only**. It is not part of the product. Person photos stay on the local
machine; nothing is uploaded.

Status: the runner code and tests are done. The model has **not** been installed or run yet
(no weights were downloaded). Everything below about FASHN comes from its public pages, not from
our own measurements. Fill in the measured numbers after the first real run.

## Findings (checked 2026-11-26)

| Item | Finding | Source |
| --- | --- | --- |
| License (code and weights) | Apache-2.0 | [GitHub](https://github.com/fashn-AI/fashn-vton-1.5), [Hugging Face](https://huggingface.co/fashn-ai/fashn-vton-1.5) |
| Third-party parts | DWPose and YOLOX (Apache-2.0), `fashn-human-parser` (own license, linked on the HF page) | HF page |
| Parameters | 972M, weights stored in bf16 | [FASHN blog](https://fashn.ai/blog/fashn-vton-1-5-open-source-release) |
| Download size | `model.safetensors` about 2 GB, DWPose ONNX files extra, human parser about 244 MB (auto-download to the Hugging Face cache on first use) | GitHub README |
| Inputs | Person image, garment image, `category` in `tops`, `bottoms`, `one-pieces`; `garment_photo_type` `model` or `flat-lay`. No mask or pose needed from us (maskless by default; pose and parsing run inside the pipeline) | GitHub README, `pipeline.py` |
| Output resolution | 576x864 (the HF page lists this as a limitation: lower than 1K VAE models) | HF page |
| Inference time | about 5 s per image on an H100 (vendor figure; none for RTX 3070) | HF page |
| VRAM | "about 8 GB" for inference (vendor figure, not measured by us) | HF page |
| Precision | bf16 on Ampere or newer (RTX 30xx included), otherwise fp32. The pipeline has no `dtype` argument and no offload option | GitHub README, `pipeline.py` |
| Python / torch | `python>=3.10`, `torch>=2.0`, `torchvision>=0.15`; plus `onnxruntime-gpu>=1.14`, `opencv-python`, `einops`, `matplotlib`, `safetensors`, `huggingface_hub`, `fashn-human-parser>=0.1.1` | `pyproject.toml` |
| Known limits | Imperfect body shape preservation; traces of the original garment when going long to short or bulky to slim | HF page |

### License note (academic comparison)

Apache-2.0 allows use, modification and redistribution, including commercial use, with
attribution and the license text. Using it as a comparison baseline in a university project is
fine. We do not redistribute its weights or code (we only download them locally and
`outputs/`, `weights/` are git-ignored). In the report, cite FASHN VTON v1.5 and name the license.
This is our reading of the public license text, not legal advice.

### Can it run on the RTX 3070 Laptop (8 GB)?

Recommendation: **try it locally first, fall back to the cloud GPU if it runs out of memory.**

- The card supports bf16 (Ampere), so the library will pick bf16 automatically. This matches
  what the model was released in. fp16 is not needed and cannot be selected through the API.
- Weights are about 2 GB in bf16, so the model itself fits. The vendor says "about 8 GB", which
  is the whole 8 GB of this card, and a Windows laptop uses some VRAM for the display. It is
  therefore borderline. It may also need to share memory with onnxruntime (DWPose) and the
  human parser.
- There is no official CPU offload or attention slicing option. If it runs out of memory:
  set `--device cpu` (works, fp32, very slow), or run the same command on the cloud GPU
  (`docs/cloud-training.md`). Lowering `num_timesteps` reduces time, not memory.
- The runner records the real peak VRAM per image (`torch.cuda.max_memory_allocated`). Note this
  counts torch only; onnxruntime memory is not included. Watch `nvidia-smi` during the run for
  the full number and write it in the table.
- Open risk: `onnxruntime-gpu` wheels are built for a specific CUDA major version. Our Docker
  image uses torch `+cu130`. If DWPose cannot find CUDA libraries, install `onnxruntime`
  (CPU) instead; pose detection on CPU is only a small part of the time.

## Install (not done yet)

The FASHN package is not on PyPI as far as we checked, so it is installed from a clone and
**not** added to `pyproject.toml` (keeps `uv.lock` free of onnxruntime/opencv and avoids pip
conflicts with `torch==2.14.1`). Do this once, in the project environment with GPU torch:

```bash
uv sync --extra cu130            # GPU torch 2.14.1 (CPU machine: --extra cpu)
git clone https://github.com/fashn-AI/fashn-vton-1.5.git external/fashn-vton-1.5
uv pip install -e external/fashn-vton-1.5
python external/fashn-vton-1.5/scripts/download_weights.py --weights-dir weights/fashn
```

Use `uv run --no-sync ...` afterwards, otherwise `uv run` may remove the manually installed
packages. If the install tries to replace torch, stop and install with `--no-deps` plus the
remaining packages by hand. Git ignores the `external/` and `weights/` folders, so neither the
FASHN code nor its weights can be committed by mistake.

The weights download is about 2 GB plus the human parser (about 244 MB) on first run.

## Input manifest

A CSV file. Relative paths are resolved from the manifest's folder.

```csv
person,garment,cut,category
inputs/person_01.jpg,inputs/crop_top_01.jpg,crop,tops
inputs/person_01.jpg,inputs/tee_01.jpg,regular,tops
inputs/person_01.jpg,inputs/hoodie_01.jpg,oversize,tops
```

- `cut`: `crop`, `regular` or `oversize`.
- `category`: `tops`, `bottoms` or `one-pieces`.
- optional `name`: output file name. Default is `<person stem>__<garment stem>.png`.
- Names must be unique. Use the same person photo for all cuts, at least 3 garments per cut.
- Images must not be committed (CI fails on image files). Keep them under `data/` or `inputs/`
  (`data/` is git-ignored).

## Run it

```bash
uv run --no-sync python -m evaluation.reference.run \
  --inputs data/reference/manifest.csv --out outputs/reference/fashn \
  --weights-dir weights/fashn --garment-photo-type flat-lay
```

Options: `--device cuda|cpu` (default auto), `--garment-photo-type model|flat-lay`. Seed, steps
and guidance are the library defaults (42, 30, 1.5) and are written to `run.json`.

Outputs in `outputs/reference/fashn/`:

- one image per manifest row (same name as in the manifest or the default name above)
- `timings.csv`: output, person, garment, cut, category, seconds, peak_vram_mb (the time and
  memory table; `peak_vram_mb` is empty on CPU)
- `run.json`: model name, dtype, GPU and torch version, model load time, and for every input the
  paths and SHA-256 of the person and garment files (the "inputs used" record)

The first image does not include model load time (stored separately as `load_seconds`), but it
may still be slower because of GPU warm-up.

## Mark hem shifts and build the gallery

Look at each output and write a marks file, for example `outputs/reference/fashn/marks.csv`:

```csv
output,hem_shifted,note
person_01__crop_top_01.png,yes,hem sits 4 cm lower than the garment photo
person_01__tee_01.png,no,
```

`hem_shifted` is `yes` or `no`. Rows that are missing show as "unmarked". A hem line shifts when
the lower edge of the garment in the output sits clearly higher or lower than in the garment
photo relative to the body (typical for crop tops pulled to the waist, or oversize tops cut
short). Mark every faulty example `yes` and write what happened in `note`.

```bash
uv run --no-sync python -m evaluation.reference.gallery \
  --run outputs/reference/fashn --marks outputs/reference/fashn/marks.csv
```

This writes `outputs/reference/fashn/gallery.html`: person, garment and output side by side,
grouped by cut, with time and VRAM per image and a red "yes" or green "no" mark. The page uses
`file://` links to the local images, so open it on the same machine.

Acceptance check: at least 3 examples per cut, faulty ones marked, `run.json` kept.

## Fallbacks (if FASHN does not fit or fails)

| Model | License | Memory and speed | Notes | Source |
| --- | --- | --- | --- | --- |
| Leffa | Code: MIT. Weights license not stated on the GitHub page; check the Hugging Face model card before use | Runs in fp16 by default; about 6 s on an A100; VRAM not stated | Needs a person mask and DensePose (the demo uses SCHP and DensePose), so more setup. Python 3.10 | [GitHub](https://github.com/franciszzj/Leffa), [HF](https://huggingface.co/franciszzj/Leffa) |
| CatVTON | CC BY-NC-SA 4.0 (non-commercial, share-alike): fine for a university comparison, never for the product | Third-party figure about 4 GB inference at 512x384 | Light model, mask-based | [HF card](https://huggingface.co/camenduru/CatVTON/blob/main/README.md), [paper](https://arxiv.org/abs/2407.15886) |

IDM-VTON (CC BY-NC-SA 4.0) is not recommended: reports of 16 GB or more VRAM make it
unlikely to fit in 8 GB ([repo](https://github.com/yisol/IDM-VTON)). To use a fallback, write a
new `ReferenceModel` subclass next to `evaluation/reference/fashn.py` (one `generate` method)
and add it to the CLI.

## Code layout

- `evaluation/reference/base.py`: `ReferenceModel` interface, cut and category lists
- `evaluation/reference/fashn.py`: FASHN adapter (imports the library only in `load()`; clear
  error if the library or weights are missing)
- `evaluation/reference/manifest.py`: manifest parsing and validation
- `evaluation/reference/runner.py`: runs a model, writes outputs, `timings.csv`, `run.json`
- `evaluation/reference/gallery.py`: gallery builder and marks reader
- `tests/test_reference_model.py`: tests with a fake model; no weights or committed images

## Open questions

- Real VRAM on the 3070 (8 GB): unknown until the first run.
- Whether `onnxruntime-gpu` works with the CUDA 13 setup; otherwise CPU onnxruntime.
- Which garment photos count as crop, regular, oversize: the team should fix the list so the
  same files are used for our model.

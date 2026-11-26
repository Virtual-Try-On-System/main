# Evaluation

One command compares the output folders of several models (for example our model and a
reference model) against the same ground-truth images. It reports SSIM, LPIPS and FID.

## Quick start

```bash
uv sync --extra cpu
uv run python -m evaluation.compare \
    --reference data/test/ground_truth \
    --model ours=outputs/ours \
    --model fashn=outputs/fashn \
    --out outputs/eval/table.csv
```

The table is printed and written as `table.csv`, `table.md` and `table.json` (settings,
library versions, warnings). Folders under `data/` and `outputs/` are git-ignored.

Options: `--allow-mismatch` (score only file names present in both folders, reported as a
warning), `--fid-min-samples N` (change the small-sample FID warning limit).

## Rules

- Images are paired **by file name** (`ground_truth/001.png` with `ours/001.png`).
  Missing or extra files stop the run with a clear error that lists the names, unless
  `--allow-mismatch` is used. An empty folder is also an error.
- Every image is converted to RGB and resized to 256x192 (height x width, bicubic) before scoring. This keeps the
  portrait 3:4 shape of try-on images. Change it with `--resolution HxW`, for example `512x384`.
  The resolution is fixed so that numbers are comparable between runs.

## Metrics

| Column | Meaning | Better |
|---|---|---|
| `n_images` | number of image pairs scored | - |
| `ssim`, `ssim_std` | scikit-image SSIM per pair; mean and std over pairs | higher |
| `lpips`, `lpips_std` | LPIPS (AlexNet) per pair, inputs in [-1, 1]; mean and std | lower |
| `fid` | Frechet distance of Inception-v3 (2048-d) features, model folder vs reference folder | lower |

FID compares two distributions, not single pairs. It is **unreliable with few images**
(a few thousand are recommended). The tool prints a warning when there are fewer than 2048
images. Use FID only to compare models on the same test set.

LPIPS and FID download pretrained weights on first use (AlexNet and Inception, about
100 MB in total). SSIM needs no download.

## Reproducibility

Fixed resolution and resize filter, seed 0, `torch.no_grad()` and deterministic algorithms
requested. Library versions and metric settings are saved in `table.json`. Two runs on the
same folders and machine gave byte-identical CSV files.

## Example table

Made from synthetic images (random smooth patterns, not people). `model_a` is the ground
truth plus noise, `model_b` is blurred and shifted.

```bash
uv run python -m evaluation.synthetic_example --root outputs/eval_demo
uv run python -m evaluation.compare \
    --reference outputs/eval_demo/ground_truth \
    --model model_a=outputs/eval_demo/model_a \
    --model model_b=outputs/eval_demo/model_b \
    --out outputs/eval_demo/table.csv
```

| model | n_images | ssim | ssim_std | lpips | lpips_std | fid |
|---|---|---|---|---|---|---|
| model_a | 12 | 0.8124 | 0.0088 | 0.1421 | 0.0147 | 396.7662 |
| model_b | 12 | 0.8303 | 0.0083 | 0.1490 | 0.0104 | 134.9709 |

These numbers only show that the tool works. FID with 12 images is not meaningful.

## Adding a metric (for example hem and sleeve error in cm)

Metrics live in `evaluation/metrics.py`. Write a `Metric` subclass with a `name` and a
`compute(pairs)` method that returns a `MetricResult` (a dict of column names to numbers and
an image count), then register it. The runner and the table do not change; new columns
appear automatically.

```python
from evaluation.metrics import FolderPair, Metric, MetricResult, register_metric

class HemSleeveErrorCm(Metric):
    name = "hem_sleeve_cm"

    def compute(self, pairs: FolderPair) -> MetricResult:
        ...  # compare measurements of the model images and the reference
        return MetricResult({"hem_sleeve_cm": error_cm}, count=len(pairs))

register_metric(HemSleeveErrorCm())
```

Registered metrics are picked up by the command line automatically, as long as the module that
registers them is imported (for example from `evaluation/__init__.py`).

## Tests

`tests/test_evaluation.py` uses small fake networks, so it needs no downloads. One test
with the marker `real_weights` uses the real networks and is skipped by default:

```bash
uv run pytest -m real_weights
```

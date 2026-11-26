"""Metric interface and the built-in metrics (SSIM, LPIPS, FID).

A metric is an object with a ``name`` and a ``compute(pairs)`` method. It gets a
``FolderPair`` (a model folder matched to the reference folder by file name) and
returns a ``MetricResult``. To add a new metric (for example a hem/sleeve error in
cm), write a ``Metric`` subclass and call ``register_metric``. The runner in
``evaluation/compare.py`` does not need to change.

Heavy libraries (torch, lpips, torchmetrics) are imported only when a real network
is needed. Tests inject small fake networks instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# (height, width). Try-on images are portrait 3:4 (VITON-HD and DressCode are 1024x768), so the
# default keeps that shape instead of squashing images into a square.
RESOLUTION = (256, 192)
SEED = 0


class PairingError(ValueError):
    """The model folder and the reference folder do not match."""


@dataclass
class FolderPair:
    """A model folder matched to the reference folder by file name."""

    reference_dir: Path
    model_dir: Path
    names: list[str]
    missing: list[str] = field(default_factory=list)  # in reference, not in model
    extra: list[str] = field(default_factory=list)  # in model, not in reference
    resolution: tuple[int, int] = RESOLUTION  # (height, width) every image is resized to

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class MetricResult:
    """Scores of one metric for one model. ``values`` maps column name to number."""

    values: dict[str, float]
    count: int
    warnings: list[str] = field(default_factory=list)


class Metric(ABC):
    name: str
    settings: dict = {}

    @abstractmethod
    def compute(self, pairs: FolderPair) -> MetricResult: ...


def _list_images(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise PairingError(f"Folder does not exist: {folder}")
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_folders(
    reference: Path,
    model: Path,
    allow_mismatch: bool = False,
    resolution: tuple[int, int] = RESOLUTION,
) -> FolderPair:
    """Match images by file name. Mismatches raise ``PairingError`` unless allowed."""
    reference, model = Path(reference), Path(model)
    ref_names, model_names = _list_images(reference), _list_images(model)
    if not ref_names:
        raise PairingError(f"No images found in reference folder: {reference}")
    if not model_names:
        raise PairingError(f"No images found in model folder: {model}")
    missing = sorted(set(ref_names) - set(model_names))
    extra = sorted(set(model_names) - set(ref_names))
    common = sorted(set(ref_names) & set(model_names))
    if (missing or extra) and not allow_mismatch:
        raise PairingError(
            f"File names in {model} do not match the reference folder {reference}.\n"
            f"  missing from model ({len(missing)}): {_preview(missing)}\n"
            f"  extra in model ({len(extra)}): {_preview(extra)}\n"
            "Fix the folders, or pass --allow-mismatch to score only the common files."
        )
    if not common:
        raise PairingError(f"No file names in common between {reference} and {model}")
    return FolderPair(reference, model, common, missing, extra, resolution)


def _preview(names: list[str], limit: int = 5) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", ... (+{len(names) - limit} more)" if len(names) > limit else "") or "-"


def load_image(path: Path, resolution: tuple[int, int] = RESOLUTION) -> np.ndarray:
    """Load as RGB uint8 array (H, W, 3), resized to (height, width) with a fixed filter."""
    height, width = resolution
    with Image.open(path) as img:
        img = img.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.uint8)


def _iter_pairs(pairs: FolderPair):
    for name in pairs.names:
        yield (
            load_image(pairs.reference_dir / name, pairs.resolution),
            load_image(pairs.model_dir / name, pairs.resolution),
        )


def _mean_std(scores: list[float]) -> tuple[float, float]:
    arr = np.asarray(scores, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


class SSIM(Metric):
    """Structural similarity per image pair (higher is better, 1.0 is identical)."""

    name = "ssim"
    settings = {"library": "scikit-image", "channel_axis": "rgb", "data_range": 255, "win": 7}

    def compute(self, pairs: FolderPair) -> MetricResult:
        from skimage.metrics import structural_similarity

        scores = [
            structural_similarity(ref, out, channel_axis=2, data_range=255)
            for ref, out in _iter_pairs(pairs)
        ]
        mean, std = _mean_std(scores)
        return MetricResult({"ssim": mean, "ssim_std": std}, len(scores))


class LPIPS(Metric):
    """Learned perceptual distance per image pair (lower is better)."""

    name = "lpips"

    def __init__(self, net=None, net_name: str = "alex"):
        self._net = net
        self.settings = {"library": "lpips", "net": net_name, "input_range": "[-1, 1]"}

    def _get_net(self):
        if self._net is None:
            import lpips

            self._net = lpips.LPIPS(net=self.settings["net"], verbose=False).eval()
        return self._net

    def compute(self, pairs: FolderPair) -> MetricResult:
        import torch

        net = self._get_net()
        scores: list[float] = []
        with torch.no_grad():
            for ref, out in _iter_pairs(pairs):
                a, b = (_to_tensor_pm1(x) for x in (ref, out))
                scores.append(float(net(a, b).reshape(-1).mean()))
        mean, std = _mean_std(scores)
        return MetricResult({"lpips": mean, "lpips_std": std}, len(scores))


def _to_tensor_pm1(img: np.ndarray):
    import torch

    tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0)
    return tensor / 127.5 - 1.0


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Frechet distance between two Gaussians fitted to feature sets (rows are samples)."""
    from scipy import linalg

    a, b = feats_a.astype(np.float64), feats_b.astype(np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(b, rowvar=False))
    covmean = linalg.sqrtm(cov_a.dot(cov_b))
    if not np.isfinite(covmean).all():  # numerical fix for near-singular matrices
        offset = np.eye(cov_a.shape[0]) * 1e-6
        covmean = linalg.sqrtm((cov_a + offset).dot(cov_b + offset))
    covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2.0 * np.trace(covmean))


class FID(Metric):
    """Frechet Inception Distance between the model folder and the reference folder.

    This is a distribution-level score (lower is better). It does not pair images and
    is unreliable with few images (a few thousand are normally recommended).
    """

    name = "fid"
    min_reliable_samples = 2048

    def __init__(self, extractor: Callable | None = None, min_samples: int | None = None):
        self._extractor = extractor
        if min_samples is not None:
            self.min_reliable_samples = min_samples
        self.settings = {
            "library": "torchmetrics NoTrainInceptionV3 (pt_inception-2015-12-05, 2048-d)",
            "min_reliable_samples": self.min_reliable_samples,
        }

    def _get_extractor(self):
        if self._extractor is None:
            from torchmetrics.image.fid import NoTrainInceptionV3

            self._extractor = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=["2048"]
            ).eval()
        return self._extractor

    def _features(self, images: list[np.ndarray]) -> np.ndarray:
        import torch

        extractor = self._get_extractor()
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), 16):
                batch = np.stack(images[i : i + 16])
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()  # uint8
                feats.append(extractor(tensor).reshape(len(batch), -1).double().numpy())
        return np.concatenate(feats)

    def compute(self, pairs: FolderPair) -> MetricResult:
        ref_images, out_images = [], []
        for ref, out in _iter_pairs(pairs):
            ref_images.append(ref)
            out_images.append(out)
        n = len(ref_images)
        warnings = []
        if n < self.min_reliable_samples:
            warnings.append(
                f"FID is unreliable with only {n} images (at least "
                f"{self.min_reliable_samples} recommended). Use it only to compare "
                "models on the same test set."
            )
        value = frechet_distance(self._features(ref_images), self._features(out_images))
        return MetricResult({"fid": value}, n, warnings)


# --- registry -------------------------------------------------------------------------

_REGISTRY: dict[str, Metric] = {}


def register_metric(metric: Metric) -> None:
    """Add a metric so that ``--metrics`` can select it. Names must be unique."""
    if metric.name in _REGISTRY:
        raise ValueError(f"Metric already registered: {metric.name}")
    _REGISTRY[metric.name] = metric


def unregister_metric(name: str) -> None:
    _REGISTRY.pop(name, None)


def list_metrics() -> dict[str, Metric]:
    return dict(_REGISTRY)


def default_metrics(lpips_net=None, fid_extractor=None, fid_min_samples=None) -> dict[str, Metric]:
    """Fresh instances of the built-in metrics. Fake networks can be injected for tests."""
    metrics = [SSIM(), LPIPS(net=lpips_net), FID(fid_extractor, fid_min_samples)]
    return {m.name: m for m in metrics}


def _register_builtins() -> None:
    for metric in default_metrics().values():
        register_metric(metric)


_register_builtins()

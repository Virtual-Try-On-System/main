"""PyTorch dataset for the preprocessed VITON-HD release (``zalando-hd-resized``).

Expected folder layout (confirmed from the VITON-HD, HR-VITON and StableVITON repositories)::

    <root>/
      train_pairs.txt            "person.jpg cloth.jpg" per line (official train pairs)
      test_pairs.txt             official test pairs
      train/  and  test/
        image/            00000_00.jpg   person photo, 1024x768
        cloth/            00000_00.jpg   garment photo
        cloth-mask/       00000_00.jpg   garment mask (white garment on black)
        image-parse-v3/   00000_00.png   person label map, label per pixel (LIP-style)
        agnostic-v3.2/    00000_00.jpg   person with garment and arms hidden (gray)

Which pairs go to which split is decided by the split files made with
``python -m training.data.splits`` (see ``splits.py``).

The published pair lists are *unpaired*: each person is listed with another person's garment
(checked on the real release: only 1 of 11,647 train lines and 0 of 2,032 test lines match).
There is no photo of the person wearing that other garment, so those pairs have no target
image. ``pairing="paired"`` (the default) therefore gives each person their own garment
(``image/X.jpg`` with ``cloth/X.jpg``), as HR-VITON does for training. Use it for training and
for SSIM/LPIPS evaluation. ``pairing="unpaired"`` uses the listed garment, for trying a
different garment on and for FID.

Each sample is a dict:

    person        float32 (3, H, W)  in [-1, 1]   (pixel / 127.5 - 1)
    garment       float32 (3, H, W)  in [-1, 1]
    garment_mask  float32 (1, H, W)  values 0.0 or 1.0
    agnostic      float32 (3, H, W)  in [-1, 1]   (hidden region is gray = 0.0)
    person_id     str   file name of the person image
    garment_id    str   file name of the garment image

Resolution is ``(height, width)``; images are resized with bicubic interpolation, masks and label
maps with nearest or thresholded interpolation. Nothing is random, so the same index always
returns the same tensors.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.data import splits as split_lists

NATIVE_SIZE = (1024, 768)  # (height, width) of the published images
SPLITS = ("train", "val", "test")
AGNOSTIC_MODES = ("auto", "folder", "parse")
AGNOSTIC_FOLDER = "agnostic-v3.2"
PAIRING_MODES = ("paired", "unpaired")
GRAY = 128
# LIP-style labels used by image-parse-v3: upper clothes, dress, coat, left arm, right arm.
HIDDEN_PARSE_LABELS = (5, 6, 7, 14, 15)


class DatasetError(RuntimeError):
    """A dataset folder, split file or image is missing or broken."""


def _phase_for(split: str) -> str:
    return "test" if split == "test" else "train"


def _open(path: Path, mode: str) -> Image.Image:
    if not path.is_file():
        raise DatasetError(f"missing file: {path}")
    try:
        with Image.open(path) as image:
            return image.convert(mode) if mode else image.copy()
    except OSError as error:
        raise DatasetError(f"cannot read image {path}: {error}") from error


def _to_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    image = image.resize((width, height), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


class VitonHDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        resolution: tuple[int, int] = (512, 384),
        agnostic: str = "auto",
        split_dir: Path = split_lists.DEFAULT_OUT,
        pairing: str = "paired",
    ):
        """
        Args:
            root: dataset folder that contains ``train/`` and ``test/``.
            split: ``train``, ``val`` (both read from ``train/``) or ``test``.
            resolution: output ``(height, width)``; default 512x384, full size is 1024x768.
            agnostic: ``folder`` reads ``agnostic-v3.2/``; ``parse`` builds the agnostic image
                from the parse map (gray over upper clothes and arms); ``auto`` uses the folder
                when it exists and the parse map otherwise.
            split_dir: folder with ``train.txt``, ``val.txt``, ``test.txt``.
            pairing: ``paired`` gives each person their own garment (has a target image);
                ``unpaired`` uses the garment listed in the split file.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if agnostic not in AGNOSTIC_MODES:
            raise ValueError(f"agnostic must be one of {AGNOSTIC_MODES}, got {agnostic!r}")
        if pairing not in PAIRING_MODES:
            raise ValueError(f"pairing must be one of {PAIRING_MODES}, got {pairing!r}")
        self.pairing = pairing
        self.root = Path(root)
        self.split = split
        self.resolution = (int(resolution[0]), int(resolution[1]))
        if not self.root.is_dir():
            raise DatasetError(f"dataset root not found: {self.root}")
        self.base = self.root / _phase_for(split)
        if not self.base.is_dir():
            raise DatasetError(f"folder not found: {self.base} (expected train/ and test/)")

        list_path = Path(split_dir) / f"{split}.txt"
        if not list_path.is_file():
            raise DatasetError(
                f"split file not found: {list_path}. Create it with: "
                f"uv run python -m training.data.splits --root {self.root}"
            )
        self.pairs = split_lists.read_pairs(list_path)
        if not self.pairs:
            raise DatasetError(f"split file is empty: {list_path}")

        has_folder = (self.base / AGNOSTIC_FOLDER).is_dir()
        if agnostic == "folder" and not has_folder:
            raise DatasetError(
                f"agnostic='folder' but {self.base / AGNOSTIC_FOLDER} does not exist. "
                "Use agnostic='parse' to build it from the parse maps."
            )
        self.agnostic_source = "folder" if agnostic == "folder" or (
            agnostic == "auto" and has_folder
        ) else "parse"

    def __len__(self) -> int:
        return len(self.pairs)

    def _agnostic_from_parse(self, person: Image.Image, person_name: str) -> torch.Tensor:
        parse_path = self.base / "image-parse-v3" / f"{Path(person_name).stem}.png"
        parse = _open(parse_path, "")
        labels = np.asarray(parse)
        if labels.ndim != 2:
            raise DatasetError(f"parse map needs one label per pixel (mode P or L): {parse_path}")
        height, width = self.resolution
        labels = np.asarray(parse.resize((width, height), Image.NEAREST))
        hidden = np.isin(labels, HIDDEN_PARSE_LABELS)
        array = np.asarray(person.resize((width, height), Image.BICUBIC), dtype=np.float32)
        array = array / 127.5 - 1.0
        array[hidden] = GRAY / 127.5 - 1.0
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> dict:
        person_name, listed_garment = self.pairs[index]
        garment_name = person_name if self.pairing == "paired" else listed_garment
        try:
            person = _open(self.base / "image" / person_name, "RGB")
            garment = _open(self.base / "cloth" / garment_name, "RGB")
            mask = _open(self.base / "cloth-mask" / garment_name, "L")
            if self.agnostic_source == "folder":
                agnostic = _to_tensor(
                    _open(self.base / AGNOSTIC_FOLDER / person_name, "RGB"), self.resolution
                )
            else:
                agnostic = self._agnostic_from_parse(person, person_name)
        except DatasetError as error:
            raise DatasetError(
                f"{self.split} sample {index} (pair {person_name} {garment_name}): {error}"
            ) from error

        height, width = self.resolution
        mask_array = np.asarray(mask.resize((width, height), Image.BILINEAR), dtype=np.float32)
        garment_mask = torch.from_numpy((mask_array > 127.5).astype(np.float32))[None]
        return {
            "person": _to_tensor(person, self.resolution),
            "garment": _to_tensor(garment, self.resolution),
            "garment_mask": garment_mask,
            "agnostic": agnostic,
            "person_id": person_name,
            "garment_id": garment_name,
        }


def verify_dataset(
    root: Path,
    expected_size: tuple[int, int] = NATIVE_SIZE,
    phases: tuple[str, ...] = ("train", "test"),
) -> list[str]:
    """Scan every official pair and return a list of problems (empty list means all fine).

    Checks that each file exists, opens, and has ``expected_size`` (height, width).
    Only image headers are read, so a full scan is quick.
    """
    root = Path(root)
    problems: list[str] = []
    for phase in phases:
        try:
            pairs = split_lists.read_pairs(split_lists.find_pairs_file(root, phase))
        except (FileNotFoundError, ValueError) as error:
            problems.append(f"{phase}: {error}")
            continue
        base = root / phase
        has_agnostic = (base / AGNOSTIC_FOLDER).is_dir()
        for person, garment in pairs:
            files = [
                (base / "image" / person, expected_size),
                (base / "cloth" / garment, expected_size),
                (base / "cloth-mask" / garment, expected_size),
                # The person's own garment, used in paired mode.
                (base / "cloth" / person, expected_size),
                (base / "cloth-mask" / person, expected_size),
                (base / "image-parse-v3" / f"{Path(person).stem}.png", expected_size),
            ]
            if has_agnostic:
                files.append((base / AGNOSTIC_FOLDER / person, expected_size))
            for path, size in files:
                folder = path.parent.name
                if not path.is_file():
                    problems.append(
                        f"{phase}: pair ({person}, {garment}): missing {folder}/{path.name}"
                    )
                    continue
                try:
                    with Image.open(path) as image:
                        found = (image.height, image.width)
                except OSError as error:
                    problems.append(f"{phase}: {folder}/{path.name}: cannot read ({error})")
                    continue
                if found != size:
                    problems.append(
                        f"{phase}: {folder}/{path.name}: size {found[0]}x{found[1]}, "
                        f"expected {size[0]}x{size[1]}"
                    )
    return problems

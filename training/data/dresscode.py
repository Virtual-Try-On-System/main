"""DressCode adapter (stub).

DressCode access needs a request form, an institutional email and a signed agreement, so the data
is not available yet and the real file layout cannot be checked. Layout as documented by the
authors and by community loaders (to be confirmed against the real download), upper body only::

    <root>/upper_body/images/<id>_0.jpg        person image
    <root>/upper_body/images/<id>_1.jpg        garment image
    <root>/upper_body/label_maps/<id>_4.png    SCHP label map
    <root>/upper_body/keypoints/<id>_2.json    OpenPose keypoints
    <root>/upper_body/train_pairs.txt          pair lists (test list names to be confirmed)

When the data arrives, implement this class with the same interface as ``VitonHDDataset``
(dict with ``person``, ``garment``, ``garment_mask``, ``agnostic``, ``person_id``, ``garment_id``).
Garment masks and agnostic images must then be derived from the label maps, because DressCode
has no ready cloth mask or agnostic folder.
"""

from pathlib import Path

from torch.utils.data import Dataset


class DressCodeDataset(Dataset):
    def __init__(self, root: Path, *args, **kwargs):
        raise NotImplementedError(
            "DressCode is not implemented yet: the dataset needs an access request "
            "(institutional email, signed agreement) and has not been downloaded. "
            "See training/data/dresscode.py and docs/data-card.md."
        )

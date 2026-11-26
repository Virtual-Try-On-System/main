"""Shared filesystem layout for the data preparation scripts."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VTON_DATA_DIR", ROOT / "data"))
ARTIFACTS = ROOT / "garment" / "artifacts"
ANALYSIS_OUT = ARTIFACTS / "analysis"
OUTPUTS = ROOT / "outputs"  # git-ignored; contact sheets and other images go here

FP_DIR = DATA_DIR / "fashionpedia"
FP_ANN_DIR = FP_DIR / "annotations"
FP_TRAIN_IMG = FP_DIR / "train"
FP_VAL_IMG = FP_DIR / "val_test2020" / "test"

S100K_DIR = DATA_DIR / "Shopping100k"
S100K_ATTR = S100K_DIR / "Attributes"
S100K_IMAGES = S100K_DIR / "Images"

SEED = 42

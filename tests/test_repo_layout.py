import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["garment", "body", "fit", "training", "service", "evaluation"]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package):
    importlib.import_module(package)


def test_docs_folder_exists():
    assert (ROOT / "docs" / "README.md").is_file()

"""Run a reference model over a manifest and record time and memory."""

import csv
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from evaluation.reference.base import ReferenceModel
from evaluation.reference.manifest import ManifestRow

TABLE_COLUMNS = ["output", "person", "garment", "cut", "category", "seconds", "peak_vram_mb"]


def _cuda():
    """Return the torch.cuda module when a GPU is usable, else None (torch is optional)."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda if torch.cuda.is_available() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    return info


def run_reference(
    model: ReferenceModel, rows: list[ManifestRow], out_dir: str | Path
) -> list[dict]:
    """Generate one output per row. Writes timings.csv and run.json into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    t0 = time.perf_counter()
    model.load()
    load_seconds = time.perf_counter() - t0

    cuda = _cuda()
    records = []
    for row in rows:
        with Image.open(row.person) as p, Image.open(row.garment) as g:
            person, garment = p.convert("RGB"), g.convert("RGB")
        if cuda:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        result = model.generate(person, garment, row.category)
        if cuda:
            cuda.synchronize()
        seconds = time.perf_counter() - t
        peak = round(cuda.max_memory_allocated() / 2**20, 1) if cuda else ""
        result.save(out_dir / row.output_name)
        records.append(
            {
                "output": row.output_name,
                "person": str(row.person),
                "garment": str(row.garment),
                "cut": row.cut,
                "category": row.category,
                "seconds": round(seconds, 3),
                "peak_vram_mb": peak,
            }
        )

    with (out_dir / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    run = {
        "model": model.name,
        "dtype": model.dtype,
        "started_utc": started,
        "load_seconds": round(load_seconds, 3),
        "environment": {**_environment(), **model.device_info()},
        "inputs": [
            {
                **rec,
                "person_sha256": _sha256(row.person),
                "garment_sha256": _sha256(row.garment),
            }
            for rec, row in zip(records, rows, strict=True)
        ],
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return records

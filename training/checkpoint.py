import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_PATTERN = re.compile(r"^step_(\d{9})\.pt$")


@dataclass
class TrainingState:
    step: int
    model: dict[str, Any]
    optimizer: dict[str, Any]
    rng: dict[str, Any]


def capture_rng() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def list_checkpoints(directory: Path) -> list[Path]:
    """Return checkpoints from oldest to newest."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if CHECKPOINT_PATTERN.match(path.name))


def save_checkpoint(directory: Path, state: TrainingState, keep: int) -> Path:
    """Write atomically, so an interruption never leaves a half-written checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step_{state.step:09d}.pt"
    temporary = target.with_suffix(".tmp")

    with temporary.open("wb") as file:
        torch.save(vars(state), file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)

    for old in list_checkpoints(directory)[:-keep]:
        old.unlink()
    return target


def load_latest_checkpoint(directory: Path, device: torch.device) -> TrainingState | None:
    """Load the newest readable checkpoint, skipping damaged files."""
    for path in reversed(list_checkpoints(directory)):
        try:
            data = torch.load(path, map_location=device, weights_only=True)
        except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as error:
            print(f"Skipping unreadable checkpoint {path.name}: {error}")
            continue
        return _move_rng_to_cpu(TrainingState(**data))
    return None


def _move_rng_to_cpu(state: TrainingState) -> TrainingState:
    """RNG states must stay on the CPU, whatever device the checkpoint was loaded to."""
    state.rng["torch"] = state.rng["torch"].cpu()
    if "cuda" in state.rng:
        state.rng["cuda"] = [item.cpu() for item in state.rng["cuda"]]
    return state

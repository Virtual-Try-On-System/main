"""Small denoising run on synthetic images. It proves the training environment, not a model."""

import argparse
import random
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from training.checkpoint import (
    TrainingState,
    capture_rng,
    load_latest_checkpoint,
    restore_rng,
    save_checkpoint,
)
from training.sync import pull_checkpoints, push_checkpoints


@dataclass
class TrainConfig:
    checkpoint_dir: Path
    steps: int = 1000
    checkpoint_every: int = 100
    keep: int = 3
    batch_size: int = 16
    image_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    log_every: int = 50
    step_delay: float = 0.0
    device: str = "auto"
    remote: str | None = None


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 3, 3, padding=1),
    )


def make_batch(config: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (config.batch_size, 3, config.image_size, config.image_size)
    clean = torch.rand(shape).to(device)
    noise = torch.randn(shape).to(device)
    return clean + noise, noise


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(
    config: TrainConfig,
    should_stop: Callable[[int], bool] = lambda step: False,
) -> nn.Module:
    """Train until config.steps, or save and return as soon as should_stop is true."""
    device = resolve_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if config.remote:
        pull_checkpoints(config.remote, config.checkpoint_dir)

    step = 0
    resumed = load_latest_checkpoint(config.checkpoint_dir, device)
    if resumed:
        step = resumed.step
        model.load_state_dict(resumed.model)
        optimizer.load_state_dict(resumed.optimizer)
        restore_rng(resumed.rng)
        print(f"Resumed from step {step}")
    else:
        print("Starting from step 0")

    def checkpoint() -> None:
        state = TrainingState(step, model.state_dict(), optimizer.state_dict(), capture_rng())
        path = save_checkpoint(config.checkpoint_dir, state, config.keep)
        if config.remote:
            push_checkpoints(config.checkpoint_dir, config.remote)
        print(f"Saved {path.name}")

    while step < config.steps:
        if should_stop(step):
            checkpoint()
            print(f"Stopped at step {step}")
            return model

        noisy, noise = make_batch(config, device)
        loss = nn.functional.mse_loss(model(noisy), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % config.log_every == 0:
            print(f"step={step} loss={loss.item():.5f}")
        if step % config.checkpoint_every == 0:
            checkpoint()
        if config.step_delay:
            time.sleep(config.step_delay)

    if step % config.checkpoint_every:
        checkpoint()
    print(f"Finished at step {step}")
    return model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/smoke"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--step-delay", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--remote", help="rclone destination for checkpoints")
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    train(parse_args(), should_stop=lambda step: stop_requested)


if __name__ == "__main__":
    main()

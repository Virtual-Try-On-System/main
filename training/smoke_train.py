"""Small denoising run on synthetic images. It proves the training environment, not a model."""

import argparse
import random
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from training.checkpoint import (
    TrainingState,
    capture_rng,
    load_latest_checkpoint,
    restore_rng,
    save_checkpoint,
)
from training.sync import pull_checkpoints, push_checkpoints
from training.tracking import MlflowTracker, NullTracker, Tracker, default_tracking_uri


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
    tracking_uri: str | None = None
    """MLflow tracking URI. None turns tracking off."""
    experiment: str = "smoke"
    location: str = "local"
    """Where the run happens, for example local or colab. Stored as a tag."""


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


def sample_grid(model: nn.Module, config: TrainConfig, device: torch.device) -> Image.Image:
    """Rows: noisy input, denoised output and clean target, for four synthetic images."""
    generator = torch.Generator().manual_seed(config.seed)
    shape = (4, 3, config.image_size, config.image_size)
    clean = torch.rand(shape, generator=generator)
    noisy = (clean + torch.randn(shape, generator=generator)).to(device)
    with torch.no_grad():
        denoised = (noisy - model(noisy)).cpu()
    rows = [torch.cat(list(images), dim=2) for images in (noisy.cpu(), denoised, clean)]
    grid = (torch.cat(rows, dim=1).clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous()
    height, width, _ = grid.shape
    return Image.frombytes("RGB", (width, height), bytes(grid.flatten().tolist()))


def start_tracker(config: TrainConfig, run_id: str | None) -> Tracker:
    if config.tracking_uri is None:
        return NullTracker()
    try:
        return MlflowTracker(
            config.tracking_uri, config.experiment, config.checkpoint_dir.name, run_id
        )
    except Exception as error:  # training goes on without tracking
        print(f"Tracking is off, MLflow could not start: {error}")
        return NullTracker()


def environment_tags(config: TrainConfig, device: torch.device) -> dict[str, str]:
    tags = {
        "location": config.location,
        "device": str(device),
        "torch_version": torch.__version__,
        "checkpoint_dir": str(config.checkpoint_dir),
    }
    if device.type == "cuda":
        tags["gpu"] = torch.cuda.get_device_name(device)
        tags["cuda_version"] = str(torch.version.cuda)
    return tags


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
    run_id = None
    resumed = load_latest_checkpoint(config.checkpoint_dir, device)
    if resumed:
        step = resumed.step
        run_id = resumed.tracking_run_id
        model.load_state_dict(resumed.model)
        optimizer.load_state_dict(resumed.optimizer)
        restore_rng(resumed.rng)
        print(f"Resumed from step {step}")
    else:
        print("Starting from step 0")

    tracker = start_tracker(config, run_id)
    if run_id is None:
        tracker.log_params(asdict(config))
    tracker.set_tags(environment_tags(config, device))
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def usage_metrics() -> dict[str, float]:
        metrics = {"session_duration_s": time.monotonic() - started}
        if device.type == "cuda":
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        return metrics

    def checkpoint() -> None:
        state = TrainingState(
            step, model.state_dict(), optimizer.state_dict(), capture_rng(), tracker.run_id
        )
        path = save_checkpoint(config.checkpoint_dir, state, config.keep)
        if config.remote:
            push_checkpoints(config.checkpoint_dir, config.remote)
        tracker.set_tags({"last_checkpoint": str(path)})
        print(f"Saved {path.name}")

    try:
        while step < config.steps:
            if should_stop(step):
                checkpoint()
                tracker.log_metrics(usage_metrics(), step)
                tracker.close("KILLED")
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
                tracker.log_metrics({"loss": loss.item(), **usage_metrics()}, step)
            if step % config.checkpoint_every == 0:
                checkpoint()
            if config.step_delay:
                time.sleep(config.step_delay)

        if step % config.checkpoint_every:
            checkpoint()
        tracker.log_metrics(usage_metrics(), step)
        tracker.log_image(sample_grid(model, config, device), f"step_{step:09d}.png")
    except BaseException:
        tracker.close("FAILED")
        raise
    tracker.close("FINISHED")
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
    parser.add_argument(
        "--tracking-uri",
        default=default_tracking_uri(),
        help="MLflow tracking URI, by default MLFLOW_TRACKING_URI or a local SQLite file",
    )
    parser.add_argument("--no-tracking", action="store_true", help="turn MLflow tracking off")
    parser.add_argument("--experiment", default="smoke", help="MLflow experiment name")
    parser.add_argument("--location", default="local", help="where the run happens, e.g. colab")
    args = vars(parser.parse_args())
    if args.pop("no_tracking"):
        args["tracking_uri"] = None
    return TrainConfig(**args)


def main() -> None:
    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    train(parse_args(), should_stop=lambda step: stop_requested)


if __name__ == "__main__":
    main()

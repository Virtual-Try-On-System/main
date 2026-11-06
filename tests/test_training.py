from dataclasses import replace
from pathlib import Path

import pytest
import torch

from training import sync
from training.checkpoint import (
    TrainingState,
    capture_rng,
    list_checkpoints,
    load_latest_checkpoint,
    save_checkpoint,
)
from training.smoke_train import TrainConfig, train


class SimulatedCrash(Exception):
    pass


def config_for(directory: Path) -> TrainConfig:
    return TrainConfig(
        checkpoint_dir=directory,
        steps=30,
        checkpoint_every=10,
        keep=2,
        batch_size=2,
        image_size=8,
        log_every=1000,
        device="cpu",
    )


def crash_at(crash_step: int):
    def should_stop(step: int) -> bool:
        if step == crash_step:
            raise SimulatedCrash
        return False

    return should_stop


def assert_same_weights(first: torch.nn.Module, second: torch.nn.Module) -> None:
    first_state, second_state = first.state_dict(), second.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name]), name


def dummy_state(step: int) -> TrainingState:
    return TrainingState(step, {"weight": torch.ones(1)}, {}, capture_rng())


def test_resume_after_crash_matches_uninterrupted_run(tmp_path):
    uninterrupted = train(config_for(tmp_path / "uninterrupted"))

    config = config_for(tmp_path / "crashed")
    with pytest.raises(SimulatedCrash):
        train(config, should_stop=crash_at(25))
    assert list_checkpoints(config.checkpoint_dir)[-1].name == "step_000000020.pt"

    assert_same_weights(uninterrupted, train(config))


def test_resume_after_graceful_stop_matches_uninterrupted_run(tmp_path):
    uninterrupted = train(config_for(tmp_path / "uninterrupted"))

    config = config_for(tmp_path / "stopped")
    train(config, should_stop=lambda step: step == 17)
    assert list_checkpoints(config.checkpoint_dir)[-1].name == "step_000000017.pt"

    assert_same_weights(uninterrupted, train(config))


def test_finished_run_does_not_train_again(tmp_path, capsys):
    config = config_for(tmp_path)
    train(config)
    capsys.readouterr()

    train(config)

    assert "Resumed from step 30" in capsys.readouterr().out


def test_steps_not_divisible_by_interval_still_end_with_a_checkpoint(tmp_path):
    config = replace(config_for(tmp_path), steps=25)

    train(config)

    assert list_checkpoints(tmp_path)[-1].name == "step_000000025.pt"


def test_only_the_newest_checkpoints_are_kept(tmp_path):
    for step in (1, 2, 3, 4):
        save_checkpoint(tmp_path, dummy_state(step), keep=2)

    assert [path.name for path in list_checkpoints(tmp_path)] == [
        "step_000000003.pt",
        "step_000000004.pt",
    ]


def test_damaged_newest_checkpoint_falls_back_to_previous(tmp_path):
    save_checkpoint(tmp_path, dummy_state(1), keep=3)
    newest = save_checkpoint(tmp_path, dummy_state(2), keep=3)
    newest.write_bytes(b"damaged")

    assert load_latest_checkpoint(tmp_path, torch.device("cpu")).step == 1


def test_leftover_temporary_file_is_ignored(tmp_path):
    save_checkpoint(tmp_path, dummy_state(1), keep=3)
    (tmp_path / "step_000000002.tmp").write_bytes(b"partial")

    assert load_latest_checkpoint(tmp_path, torch.device("cpu")).step == 1


def test_missing_folder_means_a_fresh_start(tmp_path):
    assert load_latest_checkpoint(tmp_path / "missing", torch.device("cpu")) is None


@pytest.mark.parametrize("source", ["checkpoints", "checkpoints/smoke", "data/public/viton"])
def test_allowed_folders_can_be_uploaded(source):
    sync.check_upload_allowed(Path(source))


@pytest.mark.parametrize(
    "source",
    ["data", "data/person_photos", "data/public/../person_photos", "outputs", ".", "/etc", ".env"],
)
def test_other_folders_cannot_be_uploaded(source):
    with pytest.raises(ValueError):
        sync.check_upload_allowed(Path(source))


def test_blocked_upload_never_calls_rclone(monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "run_rclone", lambda source, destination: calls.append(source))

    with pytest.raises(ValueError):
        sync.push_checkpoints(Path("data/person_photos"), "s3:bucket")

    assert calls == []

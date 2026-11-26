import shutil
from pathlib import Path

import mlflow
import pytest
import torch

from training.checkpoint import list_checkpoints, load_latest_checkpoint
from training.smoke_train import TrainConfig, train
from training.tracking import relocate_artifacts

EXPERIMENT = "test"


class SimulatedCrash(Exception):
    pass


@pytest.fixture
def tracking_uri(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    yield uri
    if mlflow.active_run():
        mlflow.end_run()


def config_for(directory: Path, tracking_uri: str | None) -> TrainConfig:
    return TrainConfig(
        checkpoint_dir=directory,
        steps=30,
        checkpoint_every=10,
        keep=2,
        batch_size=2,
        image_size=8,
        log_every=5,
        device="cpu",
        tracking_uri=tracking_uri,
        experiment=EXPERIMENT,
    )


def runs(tracking_uri: str) -> list:
    client = mlflow.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    return client.search_runs([experiment.experiment_id])


def loss_steps(tracking_uri: str, run_id: str) -> list[int]:
    history = mlflow.MlflowClient(tracking_uri).get_metric_history(run_id, "loss")
    return sorted(metric.step for metric in history)


def test_run_records_settings_curve_checkpoint_sample_and_duration(tmp_path, tracking_uri):
    train(config_for(tmp_path / "run", tracking_uri))

    (run,) = runs(tracking_uri)
    assert run.info.status == "FINISHED"
    assert run.data.params["steps"] == "30"
    assert run.data.params["learning_rate"] == "0.001"
    assert run.data.tags["location"] == "local"
    assert run.data.tags["device"] == "cpu"
    assert run.data.tags["last_checkpoint"].endswith("step_000000030.pt")
    assert "session_duration_s" in run.data.metrics
    assert loss_steps(tracking_uri, run.info.run_id) == [5, 10, 15, 20, 25, 30]

    artifacts = mlflow.MlflowClient(tracking_uri).list_artifacts(run.info.run_id, "samples")
    assert [artifact.path for artifact in artifacts] == ["samples/step_000000030.png"]
    # Artifacts sit next to the database, not in the folder the run was started from.
    assert (tmp_path / "mlruns" / "artifacts").is_dir()
    assert not (Path.cwd() / "mlruns" / run.info.experiment_id).exists()


def test_checkpoint_stores_run_id(tmp_path, tracking_uri):
    train(config_for(tmp_path / "run", tracking_uri))
    (run,) = runs(tracking_uri)
    state = load_latest_checkpoint(tmp_path / "run", torch.device("cpu"))
    assert state.tracking_run_id == run.info.run_id


def test_stopped_run_resumes_into_the_same_record(tmp_path, tracking_uri):
    config = config_for(tmp_path / "run", tracking_uri)

    train(config, should_stop=lambda step: step == 12)
    (stopped,) = runs(tracking_uri)
    assert stopped.info.status == "KILLED"

    train(config)
    (finished,) = runs(tracking_uri)
    assert finished.info.run_id == stopped.info.run_id
    assert finished.info.status == "FINISHED"
    assert loss_steps(tracking_uri, finished.info.run_id) == [5, 10, 15, 20, 25, 30]


def test_crashed_run_is_marked_failed_and_resumes(tmp_path, tracking_uri):
    config = config_for(tmp_path / "run", tracking_uri)

    def crash(step: int) -> bool:
        if step == 25:
            raise SimulatedCrash
        return False

    with pytest.raises(SimulatedCrash):
        train(config, should_stop=crash)
    (failed,) = runs(tracking_uri)
    assert failed.info.status == "FAILED"

    train(config)
    (finished,) = runs(tracking_uri)
    assert finished.info.run_id == failed.info.run_id
    assert finished.info.status == "FINISHED"


def test_record_copied_to_another_machine_still_shows_its_images(tmp_path, tracking_uri):
    train(config_for(tmp_path / "run", tracking_uri))
    if mlflow.active_run():
        mlflow.end_run()

    # Copy the record somewhere else, as when a Colab or cloud folder is downloaded.
    copied = tmp_path / "downloaded" / "mlruns"
    shutil.copytree(tmp_path / "mlruns", copied)
    copied_uri = f"sqlite:///{(copied / 'mlflow.db').as_posix()}"

    assert relocate_artifacts(copied_uri) == 1
    assert relocate_artifacts(copied_uri) == 0  # already pointing here

    (run,) = runs(copied_uri)
    assert run.info.artifact_uri.startswith(copied.resolve().as_uri())
    image = mlflow.MlflowClient(copied_uri).download_artifacts(
        run.info.run_id, "samples/step_000000030.png", str(tmp_path / "out")
    )
    assert Path(image).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_tracking_error_does_not_stop_training(tmp_path):
    config = config_for(tmp_path / "run", "unknown-scheme://nowhere")
    train(config)
    assert list_checkpoints(tmp_path / "run")[-1].name == "step_000000030.pt"


def test_training_without_tracking_writes_no_records(tmp_path):
    train(config_for(tmp_path / "run", None))
    assert not (tmp_path / "mlruns").exists()
    assert load_latest_checkpoint(tmp_path / "run", torch.device("cpu")).tracking_run_id is None

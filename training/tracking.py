"""Experiment tracking with MLflow.

Every training run records its settings, loss curve, checkpoint folder, sample outputs, duration
and GPU memory. A resumed run keeps writing to the same MLflow run, because the run id is stored
in the checkpoint.

A tracking error never stops training: it is printed and training goes on.
"""

import argparse
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"

# MLflow prints hints for coding agents on import. They are noise in training logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def default_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


class Tracker(Protocol):
    run_id: str | None

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def log_image(self, image: Any, name: str) -> None: ...

    def close(self, status: str) -> None: ...


class NullTracker:
    """Used when tracking is turned off, for example in tests."""

    run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    def set_tags(self, tags: dict[str, str]) -> None:
        pass

    def log_image(self, image: Any, name: str) -> None:
        pass

    def close(self, status: str) -> None:
        pass


class MlflowTracker:
    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        run_id: str | None = None,
    ) -> None:
        import mlflow

        folder = sqlite_folder(tracking_uri)
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)  # SQLite creates the file, not the folder
            relocate_artifacts(tracking_uri)
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=_artifact_location(tracking_uri))
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)
        self.run_id: str | None = run.info.run_id

    def _safely(self, action: str, function, *args, **kwargs) -> None:
        try:
            function(*args, **kwargs)
        except Exception as error:  # tracking must never stop training
            print(f"Tracking could not {action}: {error}")

    def log_params(self, params: dict[str, Any]) -> None:
        values = {key: str(value) for key, value in params.items()}
        self._safely("log params", self._mlflow.log_params, values)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._safely("log metrics", self._mlflow.log_metrics, metrics, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._safely("set tags", self._mlflow.set_tags, tags)

    def log_image(self, image: Any, name: str) -> None:
        def save_and_log() -> None:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / name
                image.save(path)
                self._mlflow.log_artifact(str(path), artifact_path="samples")

        self._safely("log an image", save_and_log)

    def close(self, status: str) -> None:
        self._safely("end the run", self._mlflow.end_run, status=status)


ARTIFACTS_FOLDER = "artifacts"


def relocate_artifacts(tracking_uri: str) -> int:
    """Point artifact paths in a SQLite record at the artifacts folder next to it.

    MLflow stores artifact locations as absolute paths, such as C:/Users/... or /content/drive/...
    A record copied from another machine (Colab, a cloud GPU or a container) would then show its
    metrics but not its images. This rewrites those paths for the machine that opens the record.
    Only experiments created by this module (stored under .../artifacts) are changed.
    Returns the number of experiments that were changed.
    """
    folder = sqlite_folder(tracking_uri)
    if folder is None or not (database := folder / _database_name(tracking_uri)).is_file():
        return 0
    new_root = _artifact_location(tracking_uri)
    changed = 0
    with closing(sqlite3.connect(database)) as connection, connection:
        experiments = connection.execute(
            "SELECT experiment_id, artifact_location FROM experiments"
        ).fetchall()
        for experiment_id, old_root in experiments:
            if not old_root or old_root == new_root:
                continue
            if not old_root.rstrip("/").endswith("/" + ARTIFACTS_FOLDER):
                continue
            connection.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (new_root, experiment_id),
            )
            connection.execute(
                "UPDATE runs SET artifact_uri = ? || substr(artifact_uri, ?) "
                "WHERE experiment_id = ? AND substr(artifact_uri, 1, ?) = ?",
                (new_root, len(old_root) + 1, experiment_id, len(old_root), old_root),
            )
            changed += 1
    return changed


def _database_name(tracking_uri: str) -> str:
    return Path(tracking_uri[len("sqlite:///"):]).name


def sqlite_folder(tracking_uri: str) -> Path | None:
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    return Path(tracking_uri[len(prefix):]).resolve().parent


def _artifact_location(tracking_uri: str) -> str | None:
    """Keep artifacts next to a SQLite database, so one folder holds the whole record.

    Otherwise MLflow puts them in ./mlruns of whatever folder the run was started from.
    """
    folder = sqlite_folder(tracking_uri)
    if folder is None:
        return None
    return (folder / ARTIFACTS_FOLDER).as_uri()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point the artifact paths of a copied MLflow record at this machine."
    )
    parser.add_argument("command", choices=["relocate"])
    parser.add_argument("--tracking-uri", default=default_tracking_uri())
    args = parser.parse_args()
    changed = relocate_artifacts(args.tracking_uri)
    print(f"Updated {changed} experiment(s) in {args.tracking_uri}")


if __name__ == "__main__":
    main()

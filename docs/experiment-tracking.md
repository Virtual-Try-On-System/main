# Experiment tracking

Every training run is recorded with [MLflow](https://mlflow.org/). Whoever runs a training does
not need to do anything extra: the training script records the run by itself.

## What is recorded

| What | Where in MLflow |
|---|---|
| Settings: steps, batch size, learning rate, seed and the rest | Parameters |
| Loss curve | Metric `loss`, every `log_every` steps |
| Duration of the session | Metric `session_duration_s` |
| GPU memory | Metric `peak_vram_mb`, only on a GPU |
| Checkpoint folder and the last checkpoint | Tags `checkpoint_dir` and `last_checkpoint` |
| Sample outputs | Artifact `samples/step_<N>.png`: noisy input, output and target |
| Where it ran, device, GPU, torch and CUDA versions | Tags `location`, `device`, `gpu`, `torch_version`, `cuda_version` |
| How it ended | Run status: `FINISHED`, `KILLED` (stopped and saved) or `FAILED` (crashed) |

The MLflow run id is stored in every checkpoint. A resumed training keeps writing to the same
run, so one training is one record, even after many interruptions.

Tracking never stops a training. If MLflow fails, the error is printed and training goes on.

## Where the record is kept

By default everything goes to `mlruns/` in the repository root: a SQLite file `mlflow.db` and an
`artifacts/` folder next to it. Git ignores `mlruns/`.

To use another place, set `MLFLOW_TRACKING_URI` or pass `--tracking-uri`, for example
`sqlite:///checkpoints/smoke/mlruns/mlflow.db`. To turn tracking off, pass `--no-tracking`.

## Run a training

```bash
uv run python -m training.smoke_train --steps 300 --checkpoint-dir checkpoints/smoke
```

In Docker on a GPU, the `train` service mounts `mlruns/`, so the record lands in the same place:

```bash
docker compose --profile training run --rm train python -m training.smoke_train --steps 300
```

## See the runs

The web interface needs the full MLflow package, which is not a project dependency. `uvx` runs it
without installing it into the project:

```bash
uvx --python 3.11 --from mlflow==3.18.0 mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Then open `http://127.0.0.1:5000`. Under the `smoke` experiment, each run shows its parameters,
the loss chart under **Model metrics** and the sample image under **Artifacts**.

If `uvx` fails on Windows with "file is being used by another process" (an antivirus scanning
the download), install it once with pip into its own environment instead:

```bash
uv venv --seed --python 3.11 ../mlflow-ui
```

```bash
../mlflow-ui/Scripts/python -m pip install mlflow==3.18.0
```

```bash
../mlflow-ui/Scripts/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Runs on a cloud machine or Colab

Keep the record inside the checkpoint folder, so it travels with the checkpoints:

```bash
bash training/run_on_gpu_host.sh --steps 1000 --remote s3:<bucket>/checkpoints/smoke --tracking-uri sqlite:///checkpoints/smoke/mlruns/mlflow.db --location aws
```

With `--remote`, the record is uploaded with every checkpoint and downloaded again on a new
machine, so the run continues in the same record. On Colab, point `--tracking-uri` at a folder
in Google Drive instead.

To look at it on your own computer, download that `mlruns` folder, then point its image paths at
your machine and start the web interface on it:

```bash
uv run python -m training.tracking relocate --tracking-uri sqlite:///path/to/mlruns/mlflow.db
```

MLflow stores the image folder as an absolute path of the machine that created the record. The
`relocate` command rewrites it for your machine. Training does the same automatically when it
opens a record, so a copied record can also be continued.

## For future training scripts

Use `training/tracking.py`: create an `MlflowTracker` (or `NullTracker` when tracking is off),
log parameters once, metrics with their step, and close the run with its status. Store
`tracker.run_id` in the checkpoint, as `training/smoke_train.py` does.

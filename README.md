# Virtual Try-On System

A fit-aware virtual try-on system. Most try-on tools stretch a garment to match the body, so a
crop top looks long and an oversized sweatshirt looks tight. This project combines garment
analysis, body analysis and a fit engine to show where a garment actually ends on the user's
body, in centimeters.

The project is at an early stage. The repository skeleton, tests, Docker setup and CI are ready.
The module folders are still empty and get filled as the tasks are completed.

## What is where

| Path | Purpose |
|---|---|
| `garment/` | Garment analysis: category, fit class, mask and ratio vector |
| `body/` | Body analysis: pose, cm/pixel scale and body measurements |
| `fit/` | Fit engine: compares garment and body, produces the target mask |
| `training/` | Training scripts and the cloud training environment, see [docs/cloud-training.md](docs/cloud-training.md) |
| `service/` | FastAPI service and the model interface |
| `evaluation/` | Metrics and comparison scripts |
| `docs/` | Project documentation |
| `tests/` | Automated tests |
| `pyproject.toml`, `uv.lock` | Dependencies and their pinned versions |
| `Dockerfile`, `docker-compose.yml` | Test environment |
| `.github/workflows/ci.yml` | Automated checks on every pull request |
| `CONTRIBUTING.md` | Branch, code review and data rules |

## Getting started

### 1. Clone the repository

You need [Git](https://git-scm.com/downloads) and access to the
[Virtual-Try-On-System](https://github.com/Virtual-Try-On-System) organization.

```bash
git clone https://github.com/Virtual-Try-On-System/main.git vton
```

```bash
cd vton
```

### 2. Choose how to run it

**Option A: Docker.** This is the quickest way to check that everything works. Install
[Docker Desktop](https://www.docker.com/products/docker-desktop/), make sure it is running, then:

```bash
docker compose run --rm --build test
```

You should see all tests pass. Nothing else needs to be installed.

**Option B: Local setup.** Use this while developing. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the dependencies. Use `--extra cu130` instead of `--extra cpu` to train on an NVIDIA
GPU. This creates a `.venv` folder and downloads Python 3.11 if you
do not have it, so your system Python version does not matter:

```bash
uv sync --extra cpu
```

### 3. Everyday commands

| Task | Command |
|---|---|
| Run the tests | `uv run pytest` |
| Check code style | `uv run ruff check .` |
| Fix import order and simple style issues | `uv run ruff check . --fix` |
| Run a Python file | `uv run python path/to/file.py` |
| Add a dependency | `uv add <package>` |
| Update after pulling changes | `uv sync --extra cpu` |

In your IDE, select `.venv` as the Python interpreter.

## Making a change

1. Update `main` and create a branch, for example `feature/garment-classifier`.
2. Make your change and run `uv run pytest` and `uv run ruff check .`.
3. Open a pull request. CI runs automatically.
4. A teammate reviews it, then you merge it.

The details are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Data rules

- Person and garment images stay on your machine. They are never committed.
- Person photos are never uploaded to the cloud.
- Keep datasets in `data/`, checkpoints in `checkpoints/` and run outputs in `outputs/`. Git
  ignores these folders, so create them locally when you need them.
- Image files and model weights are ignored everywhere in the repository, and CI fails if one is
  committed.
- Secrets go in `.env`, which git ignores. Copy `.env.example` to `.env` to create it.

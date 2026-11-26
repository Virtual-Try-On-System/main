"""Copy training files to and from cloud storage with rclone.

Only folders on the allow list can be uploaded, so person photos cannot reach the cloud through
this script. Nothing is ever deleted on the remote side.
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_UPLOAD_FOLDERS = ("checkpoints", "data/public")


def check_upload_allowed(source: Path) -> None:
    resolved = (ROOT / source).resolve()
    allowed = [(ROOT / folder).resolve() for folder in ALLOWED_UPLOAD_FOLDERS]
    if not any(resolved == folder or folder in resolved.parents for folder in allowed):
        raise ValueError(f"Upload is not allowed from {source}: {ALLOWED_UPLOAD_FOLDERS} only")


def run_rclone(source: str, destination: str) -> None:
    subprocess.run(["rclone", "copy", source, destination], check=True)


def push_checkpoints(source: Path, remote: str) -> None:
    check_upload_allowed(source)
    run_rclone(str(source), remote)


def pull_checkpoints(remote: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    run_rclone(remote, str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["push", "pull"])
    parser.add_argument("local", type=Path)
    parser.add_argument("remote")
    args = parser.parse_args()

    if args.direction == "push":
        push_checkpoints(args.local, args.remote)
    else:
        pull_checkpoints(args.remote, args.local)


if __name__ == "__main__":
    main()

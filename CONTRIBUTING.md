# Contributing

## Language

Code, folder names, comments, commit messages and documentation are written in English.

## Branches

- `main` always passes CI. Nobody pushes to it directly.
- Create a short-lived branch per task: `feature/<name>`, `fix/<name>` or `docs/<name>`.

## Commits

- Commit titles are written in English and summarize what the change does, for example
  `Add JSON schemas for garment and body`.
- Put the details, the reasons and the test results in the pull request description, not in
  the commit message.

## Code review

- Every change goes through a pull request.
- At least one teammate reviews and approves it before merge.
- CI must be green before merge.
- The author merges after approval.

## Dependencies

Dependencies are pinned in `uv.lock`. Add them with `uv add <package>` and commit the lock
file with the change. Never edit `uv.lock` by hand.

## Data rules

- Person photos and garment images are never committed.
- Person photos never leave local machines, including to cloud storage.
- Only non-personal data may be used for cloud model tests, and only if its license allows it.
- Keep datasets in `data/`, checkpoints in `checkpoints/` and run outputs in `outputs/`.
  These folders are ignored by git.
- Secrets go in `.env`, which is ignored by git. Never commit keys or tokens.

## Team rules

- A task closes only when its acceptance criterion is proven.
- If you are stuck on something, tell the team within 24 hours.

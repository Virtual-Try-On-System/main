# Cloud training

How to run training on a rented GPU machine and survive interruptions. The code does not depend on
a provider. It needs a Linux machine with an NVIDIA GPU, Docker and the NVIDIA Container Toolkit.

Provider choice and prices are in [cloud-costs.md](cloud-costs.md).

## What is built

| Piece | File |
|---|---|
| Training image with CUDA and rclone | `training/Dockerfile` |
| Trial training on synthetic images | `training/smoke_train.py` |
| Checkpoint save and resume | `training/checkpoint.py` |
| Cloud storage sync with an upload allow list | `training/sync.py` |
| One command for a GPU machine | `training/run_on_gpu_host.sh` |

## How interruptions are handled

- Checkpoints are written atomically, so an interruption cannot leave a half-written file.
- Each checkpoint holds the model, optimizer, RNG state and step.
- If `--remote` is set, every checkpoint is copied to cloud storage. The newest ones are also
  pulled back at startup, so a brand new machine continues from the last saved step.
- On `SIGTERM`, training saves a checkpoint and exits cleanly.
- On a hard kill, training restarts from the last checkpoint and repeats only the steps after it.
- An unreadable checkpoint is skipped and the previous one is used.
- A resumed run gives the same weights as an uninterrupted one. The tests check this on CPU.

## Run on a GPU machine

Clone the repository, then create `.env` from `.env.example` and fill in the storage values:

```bash
cp .env.example .env
```

Start training. Repeat the same command after any interruption:

```bash
bash training/run_on_gpu_host.sh --steps 1000 --checkpoint-every 100 --remote s3:<bucket>/checkpoints/smoke
```

Look for `Resumed from step N` in the output after a restart.

## Avoiding surprise costs

- Set a budget alarm before starting any machine.
- Stop or terminate the machine when training ends. A forgotten GPU machine is the biggest risk.
- Disks and storage can still cost money after the machine is gone.
- Keep storage and the machine in the same region.

## Data rules

- Person photos never leave local machines. `training/sync.py` uploads only from `checkpoints/`
  and `data/public/`.
- Keep only datasets whose license allows cloud upload in `data/public/`.
- Marketplace machines may run on third-party hardware, so check the host type before using one.
- The trial training uses synthetic images, so it needs no dataset.

## Still to do

- [ ] Choose the provider and open the account
- [ ] Get GPU quota
- [ ] Build the training image for the first time and confirm it sees the GPU
- [ ] Confirm a real rclone transfer to cloud storage
- [ ] Run the trial on a cloud machine and interrupt it
- [ ] Check the VITON-HD and DressCode licenses for cloud upload
- [ ] Replace estimated hours in the cost table with real ones
- [ ] Push the training image to a registry, so machines skip the build step

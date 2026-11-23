# SmolVLA-WAM

SmolVLA with touch on UniVTAC demonstrations, with or without a world action model: SmolVLA (SmolVLM2 frozen, the
action expert trained from scratch in fp32) plus the two GelSight marker streams through the UniVTAC ResNet18
tactile encoder (cross-attention fusion). With `wam_enabled` the action expert also denoises the future SigLIP
latents of the cameras and the future tactile latents.

This repository covers training (cluster scripts) and the policy code. Evaluation runs in the UniVTAC simulator,
through the deploy files in `univtac_eval/`.

## Code

`src/lerobot` is the LeRobot copy the UniVTAC repo vendored for SmolVLA (Apache-2.0, see `LICENSE`), with:

- `policies/smolvla`: the visuo-tactile extension (`vitac_encoder.py`, `use_tactile`, `tactile_*`, `vitac_*`), the
  world action model (`wam_*`), `camera_keys`.
- `datasets/factory.py`: per-key frame offsets (`observation_delta_indices_for`), so the tactile streams of the
  world action model load their future frames.

`scripts/`:

- `install_cluster.sbatch`, `train_cluster.sbatch`: the venv and the training runs on NHR alex.
- `process_data.py`: UniVTAC HDF5 episodes to a LeRobot dataset (reads them through the UniVTAC repo,
  `UNIVTAC_ROOT`). `exact_stats.py`: exact quantiles in the dataset's stats (LeRobot averages the per-episode
  ones).
- `precompute_wam_latents.py`: the frozen latents the world action model trains on.
- `infer.py`, `infer.sh`, `eval_traj.py`, `video_to_images.py`: offline inference and dataset tools.

## Setup (NHR alex)

```bash
git clone <this repo> ~/Projects/smolvla-wam
sbatch ~/Projects/smolvla-wam/scripts/install_cluster.sbatch   # venv + tactile encoder checkpoint
```

Everything heavy lives on the anvme workspace (`$WS/smolvla`: `venv-wam`, `ckpt`, `data`, `outputs`; SmolVLM2 in
`$WS/hf`). The datasets are `$WS/smolvla/data/local/<task>-clean-100_vitac` (shared with Cosmos3-WAM); a missing
one is built from the UniVTAC HDF5 episodes of `~/Projects/UniVTAC/data`.

## Train

```bash
cd ~/Projects/smolvla-wam
sbatch scripts/train_cluster.sbatch put_bottle_in_shelf --steps=15000 --batch_size=32 --save_freq=1500
sbatch --export=NONE,WAM=1 scripts/train_cluster.sbatch put_bottle_in_shelf            # world action model
sbatch --export=NONE,WAM=1,TACTILE_FROZEN=1 scripts/train_cluster.sbatch put_bottle_in_shelf
```

- Cameras follow the UniVTAC benchmark: head and wrist for insert_tube and lift_can, the head camera alone for
  the other tasks (`policy.camera_keys` in the run config the script writes).
- Run names are `smolvla_vitac_<task>-clean-100_<suffix>`, suffix `fp32_scratch`, `wam` or `wam_tacfrozen`
  (`RUN_SUFFIX` overrides), in `$WS/smolvla/outputs`; a fresh run into an existing output dir fails.
- `RESUME=1` continues a run from its last checkpoint (`--dependency=afterany:<job>`). `SMOKE=1` trains 20 steps
  into `$WS/smolvla/smoke`, logging to the wandb project `<project>_smoke`.
- Logs go to wandb online (entity `cong_fu`, project `smolvla_univtac`; `WANDB_MODE`, `WANDB_ENTITY`,
  `WANDB_PROJECT` override). The script runs with `--export=NONE`, so set variables in `--export`.

## Evaluate

See `univtac_eval/` (copied into `UniVTAC/policy/smolvla/`).

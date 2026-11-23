#!/bin/bash -l
# Train SmolVLA with touch (and its world action model, WAM=1) on one NHR alex A40, on the UniVTAC LeRobot datasets.
#   sbatch scripts/train_cluster.sbatch <task> [trainer args, e.g. --steps=15000 --batch_size=32]
#   sbatch --export=NONE,WAM=1 scripts/train_cluster.sbatch put_bottle_in_shelf
# Needs scripts/install_cluster.sbatch first. Everything heavy lives on the anvme workspace ($WS/smolvla): the venv,
# the tactile encoder checkpoint, the datasets and the outputs.
#
# Inputs: the cameras of the UniVTAC benchmark task (the camera_names of the released ACT checkpoints: head and
#   wrist for insert_tube and lift_can, the head camera alone otherwise), the 16-dim state, the two marker tactile
#   streams through the UniVTAC ResNet18 (cross-attention fusion). SmolVLM2 frozen, the action expert from scratch.
# WAM=1: the world action model, the future SigLIP latents of the cameras and the future tactile latents denoised
#   with the actions; TACTILE_FROZEN=1 freezes the tactile encoder. The frozen latents are precomputed once per
#   dataset (scripts/precompute_wam_latents.py) into <dataset>/wam_latents_{frozen,unfrozen}.
# Dataset: $WS/smolvla/data/local/<task>-<config>-<num>_vitac. When missing it is built from the UniVTAC HDF5
#   episodes (scripts/process_data.py, needs the UniVTAC repo at UNIVTAC_ROOT) and its quantiles made exact
#   (scripts/exact_stats.py).
# Run name (output dir, wandb job): smolvla_vitac_<task>-<config>-<num>_<RUN_SUFFIX>; RUN_SUFFIX defaults to
#   fp32_scratch, wam (WAM=1) and wam_tacfrozen (WAM=1 TACTILE_FROZEN=1). RESUME=1 continues the run from its last
#   checkpoint (submit with --dependency=afterany:<job>). SMOKE=1 trains SMOKE_STEPS steps into $WS/smolvla/smoke
#   with wandb project <WANDB_PROJECT>_smoke.
#
# wandb: online through the NHR proxy (needs ~/.netrc). WANDB_MODE=offline keeps the run under the output dir.
#SBATCH --job-name=smolvla-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=smolvla-train-%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

if [ $# -gt 0 ] && [[ $1 != --* ]]; then TASK_NAME=$1; shift; fi
TASK_NAME=${TASK_NAME:?usage: sbatch train_cluster.sbatch <task> [trainer args]}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
NUM_WORKERS=${NUM_WORKERS:-12}
REPO=${REPO:-$HOME/Projects/smolvla-wam}
export UNIVTAC_ROOT=${UNIVTAC_ROOT:-$HOME/Projects/UniVTAC}  # only for building a missing dataset
WS=${WS:-$(ws_find univtac)}
ROOT=${SMOLVLA_ROOT:-$WS/smolvla}
VENV=${VENV:-$ROOT/venv-wam}  # built by scripts/install_cluster.sbatch
TACTILE_CKPT=${TACTILE_CKPT:-$ROOT/ckpt/univtac_resnet18_20251128-125750.pth}
# SmolVLM2 and the datasets arrow cache on NVMe ($WORK is NFS: cold reads there starve the GPU).
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$ROOT/hf_datasets}

WAM=${WAM:-0}
TACTILE_FROZEN=${TACTILE_FROZEN:-0}
SMOKE=${SMOKE:-0}
SMOKE_STEPS=${SMOKE_STEPS:-20}
RESUME=${RESUME:-0}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_ENTITY=${WANDB_ENTITY:-cong_fu}
WANDB_PROJECT=${WANDB_PROJECT:-smolvla_univtac}

# Cameras of the UniVTAC benchmark task. Not part of the run name, since the task determines it.
case $TASK_NAME in
    insert_tube|lift_can) CAMERAS="[observation.images.head, observation.images.wrist]" ;;
    *) CAMERAS="[observation.images.head]" ;;
esac

if [ "$WAM" = 1 ]; then
    if [ "$TACTILE_FROZEN" = 1 ]; then DEFAULT_SUFFIX=wam_tacfrozen; else DEFAULT_SUFFIX=wam; fi
else
    DEFAULT_SUFFIX=fp32_scratch
fi
SETTINGS=${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
REPO_ID=local/${SETTINGS}_vitac
DATASET_ROOT=$ROOT/data/$REPO_ID
RUN_TAG=${SETTINGS}_${RUN_SUFFIX:-$DEFAULT_SUFFIX}
if [ "$SMOKE" = 1 ]; then
    OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/smoke/smolvla_vitac_${RUN_TAG}_${SLURM_JOB_ID}}
    WANDB_PROJECT=${WANDB_PROJECT}_smoke
else
    OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/smolvla_vitac_${RUN_TAG}}
fi

cd "$REPO"
export PYTHONPATH=$REPO/src
PY=$VENV/bin/python
# Single-GPU jobs can share a node; torchrun's default rendezvous port 29500 then collides.
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[smolvla] task=$TASK_NAME cameras=$CAMERAS wam=$WAM tactile_frozen=$TACTILE_FROZEN dataset=$DATASET_ROOT" \
     "output=$OUTPUT_DIR git=$(git rev-parse --short HEAD 2>/dev/null || echo none)"

if [ "$RESUME" = 1 ]; then
    CFG=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json
    [ -f "$CFG" ] || { echo "[smolvla] nothing to resume: $CFG"; exit 1; }
    exec "$VENV/bin/torchrun" --nproc_per_node=1 --master_port="$MASTER_PORT" \
        -m lerobot.scripts.lerobot_train --config_path="$CFG" --resume=true --num_workers="$NUM_WORKERS" "$@"
fi

[ -f "$TACTILE_CKPT" ] || { echo "[smolvla] tactile encoder checkpoint missing: $TACTILE_CKPT"; exit 1; }
if [ -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[smolvla] dataset exists, skipping process_data"
else
    "$PY" scripts/process_data.py "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" \
        --repo-id "$REPO_ID" --out "$DATASET_ROOT" --image-writer-processes 4 --image-writer-threads 4
    "$PY" scripts/exact_stats.py "$DATASET_ROOT"
fi

WAM_ARGS=()
if [ "$WAM" = 1 ]; then
    if [ "$TACTILE_FROZEN" = 1 ]; then
        LATENT_DIR=$DATASET_ROOT/wam_latents_frozen
        TACTILE_LATENTS=()  # script default: both marker streams
        WAM_ARGS=(--policy.vitac_freeze_backbone=true)
    else
        LATENT_DIR=$DATASET_ROOT/wam_latents_unfrozen
        TACTILE_LATENTS=(--tactile)  # none: the trained encoder computes them online
    fi
    # The latents of both cameras are computed; the policy reads the ones of its cameras. info.json marks a
    # complete run.
    if [ -f "$LATENT_DIR/info.json" ]; then
        echo "[smolvla] WAM latents exist: $LATENT_DIR"
    else
        "$PY" scripts/precompute_wam_latents.py "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" \
            --root "$DATASET_ROOT" --out "$LATENT_DIR" --num-workers "$NUM_WORKERS" --vitac-ckpt "$TACTILE_CKPT" \
            ${TACTILE_LATENTS[@]+"${TACTILE_LATENTS[@]}"}
    fi
    WAM_ARGS+=(--policy.wam_enabled=true --policy.wam_latent_dir="$LATENT_DIR")
fi

SMOKE_ARGS=()
[ "$SMOKE" = 1 ] && SMOKE_ARGS=(--steps="$SMOKE_STEPS" --save_freq="$SMOKE_STEPS" --log_freq=5)

# The run config: scripts/train_config.yml plus the cameras (a YAML list; draccus splits a command line string).
CFG=${TMPDIR:-/tmp}/smolvla_${SLURM_JOB_ID}.yml
sed "s|^policy:\$|policy:\n  camera_keys: $CAMERAS|" scripts/train_config.yml > "$CFG"
grep -qF "camera_keys: $CAMERAS" "$CFG" || { echo "[smolvla] could not set the cameras in $CFG"; exit 1; }

"$VENV/bin/torchrun" --nproc_per_node=1 --master_port="$MASTER_PORT" \
    -m lerobot.scripts.lerobot_train \
    --config_path="$CFG" \
    --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT" \
    --output_dir="$OUTPUT_DIR" --job_name="smolvla_vitac_${RUN_TAG}" \
    --policy.vitac_pretrained_path="$TACTILE_CKPT" \
    --save_freq=10000 --num_workers="$NUM_WORKERS" \
    --wandb.enable=true --wandb.mode="$WANDB_MODE" --wandb.entity="$WANDB_ENTITY" \
    --wandb.project="$WANDB_PROJECT" --wandb.disable_artifact=true \
    ${WAM_ARGS[@]+"${WAM_ARGS[@]}"} ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} "$@"

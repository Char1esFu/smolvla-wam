#!/bin/bash -l
# Build the training venv ($WS/smolvla/venv-wam) on an NHR alex A40 node and put the UniVTAC tactile encoder
# checkpoint next to it. Training checks: scripts/train_cluster.sbatch with SMOKE=1.
#   sbatch scripts/install_cluster.sbatch
#SBATCH --job-name=smolvla-install
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=smolvla-install-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1 cmake/3.31.9
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PATH=$HOME/.local/bin:$PATH

REPO=${REPO:-$HOME/Projects/smolvla-wam}
WS=${WS:-$(ws_find univtac)}
ROOT=${SMOLVLA_ROOT:-$WS/smolvla}
VENV=${VENV:-$ROOT/venv-wam}
UNIVTAC_ROOT=${UNIVTAC_ROOT:-$HOME/Projects/UniVTAC}
# uv cache on node-local disk: $WORK is close to its file-count quota.
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uv-cache UV_LINK_MODE=copy

cd "$REPO"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
VIRTUAL_ENV=$VENV uv pip install -e ".[training]" --torch-backend=cu128
"$VENV/bin/python" -c "
import torch, numpy, transformers, peft, lerobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('numpy', numpy.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)
print('lerobot', lerobot.__file__)
assert torch.cuda.is_available() and lerobot.__file__.startswith('$REPO/src/')"

# The UniVTAC ResNet18 tactile encoder (pretrained on simulated contact data).
mkdir -p "$ROOT/ckpt"
CKPT=$ROOT/ckpt/univtac_resnet18_20251128-125750.pth
[ -f "$CKPT" ] || cp "$UNIVTAC_ROOT/encoder/checkpoints/resnet18/20251128-125750/best.pth" "$CKPT"
ls -l "$CKPT"
echo "INSTALL DONE"

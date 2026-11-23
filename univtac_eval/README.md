# Evaluating in UniVTAC

The UniVTAC eval harness (`scripts/eval_policy.py` of the UniVTAC repo) runs in the IsaacLab environment; the
policy runs in a venv of this repo. `deploy_policy.Policy` starts `smolvla_server.py` with that venv and talks to it
over HTTP.

1. Clone this repo and install it into a local venv (here `~/proj/smolvla-wam/.venv`):

   ```bash
   git clone <this repo> ~/proj/smolvla-wam && cd ~/proj/smolvla-wam
   uv venv .venv --python 3.12
   VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[training]" --torch-backend=cu128
   ```

2. Put the files of this directory into `policy/smolvla/` of UniVTAC (`smolvla_python` in `deploy.yml` points at
   the venv).

3. Copy a run dir from the cluster (`$WS/smolvla/outputs/smolvla_vitac_<task>-clean-100_<suffix>`, at least
   `checkpoints/last/pretrained_model`) to `~/proj/smolvla-wam/outputs/` and run, from the UniVTAC root:

   ```bash
   TRAIN_CONFIG=fp32_scratch EP_NUM=100 \
       python scripts/eval_policy.py put_bottle_in_shelf clean smolvla/deploy --headless
   ```

`TRAIN_CONFIG` is the run suffix (`fp32_scratch`, `wam`, `wam_tacfrozen`). The cameras and tactile streams come
from the checkpoint config. Execution: the first `smolvla_replan_every` (10) actions of each chunk, the next chunk
guided towards the rest of the previous one (`smolvla_rtc`); results go to
`eval_result/smolvla/<task>/deploy/<TRAIN_CONFIG>-rtc10/`. Runs not found under `smolvla_outputs_root` are looked up
in `smolvla_archive_root` (the earlier runs with head and wrist cameras for every task).

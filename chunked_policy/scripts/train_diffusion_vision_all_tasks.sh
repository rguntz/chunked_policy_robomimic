#!/usr/bin/env bash
# Train the vision diffusion policy (UNet, ResNet encoder, paper setup) on the 5 robomimic
# ph tasks, one after the other, on GPU 0 only. All runs log to the same wandb project,
# one run per task. Missing ph datasets are downloaded before each task.
#
# Usage (from the repo root or anywhere):
#   bash chunked_policy/scripts/train_diffusion_vision_all_tasks.sh              # can, square, tool_hang, transport (lift is done)
#   bash chunked_policy/scripts/train_diffusion_vision_all_tasks.sh lift can     # a subset
#   EXTRA="training.debug=True logging.mode=disabled" bash chunked_policy/scripts/train_diffusion_vision_all_tasks.sh lift   # quick test
#
# Training length: 100k gradient steps for lift and can, 250k for square, tool_hang and
# transport. The configs count epochs, so epochs = steps / (steps per epoch at batch size 64),
# rounded up to a multiple of 50 plus 1: rollouts and checkpoints run every 50 epochs
# (epochs counted from 0), so the last epoch is evaluated and checkpointed.
#   task        steps/epoch   epochs   steps
#   lift            127         801    101.7k
#   can             335         301    100.8k
#   square          440         601    264.4k
#   tool_hang      1446         201    290.6k
#   transport      1415         201    284.4k
#
# Disk: each checkpoint is ~4.4 GB; with topk 2 + latest, a task keeps ~13 GB of checkpoints.
# The vision datasets are 0.8 / 2.0 / 2.6 / 66.5 / 16.2 GB (lift / can / square / tool_hang / transport).
set -euo pipefail
cd "$(dirname "$0")/../.."    # repo root

export CUDA_VISIBLE_DEVICES=0   # GPU 0 only (training; mujoco-py renders on the CPU)
PYTHON=.venv/bin/python
SEED=42
PROJECT=chunked_policy_robomimic_diffusion_vision
declare -A EPOCHS=([lift]=801 [can]=301 [square]=601 [tool_hang]=201 [transport]=201)

# lift is already trained, so it is left out of the default list (pass it explicitly to rerun it)
if [ $# -gt 0 ]; then TASKS=("$@"); else TASKS=(can square tool_hang transport); fi
for task in "${TASKS[@]}"; do
    [ -n "${EPOCHS[$task]:-}" ] || { echo "unknown task: $task (lift can square tool_hang transport)"; exit 1; }
done

failed=()
for task in "${TASKS[@]}"; do
    echo "=== $(date '+%F %T') $task: ${EPOCHS[$task]} epochs, $(df -h --output=avail . | tail -1 | tr -d ' ') free on disk"
    if ! $PYTHON chunked_policy/scripts/download_robomimic.py --tasks "$task" --obs vision; then
        echo "=== $task: dataset download FAILED, skipping"; failed+=("$task"); continue
    fi
    if ! $PYTHON train.py --config-name=train_diffusion_unet_vision_workspace \
            task="${task}_vision_abs" \
            training.seed=$SEED \
            training.device=cuda:0 \
            training.num_epochs="${EPOCHS[$task]}" \
            checkpoint.topk.k=2 \
            logging.project=$PROJECT \
            logging.name="diffusion_unet_vision_${task}_ph_seed${SEED}" \
            ${EXTRA:-}; then
        echo "=== $task: training FAILED, continuing with the next task"; failed+=("$task")
    fi
done

echo "=== $(date '+%F %T') done. failed: ${failed[*]:-none}"

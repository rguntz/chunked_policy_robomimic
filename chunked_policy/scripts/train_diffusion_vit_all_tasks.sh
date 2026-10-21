#!/usr/bin/env bash
# Train the vision diffusion policy (UNet) with the ViT encoder (MinVit from IBRL, IBRL robomimic
# settings) on the 5 robomimic ph tasks, one after the other, on GPU 0 only. Same setup as
# train_diffusion_vision_all_tasks.sh except the encoder (policy.vision_encoder=vit) and the
# number of epochs. All runs log to the same wandb project as the ResNet runs, one run per task.
# Missing ph datasets are downloaded before each task.
#
# Usage (from the repo root or anywhere):
#   bash chunked_policy/scripts/train_diffusion_vit_all_tasks.sh              # all 5 tasks
#   bash chunked_policy/scripts/train_diffusion_vit_all_tasks.sh lift can     # a subset
#   EXTRA="training.debug=True logging.mode=disabled" bash chunked_policy/scripts/train_diffusion_vit_all_tasks.sh lift   # quick test
#
# Training length: 350 epochs for lift and can, 600 for square, 150 for tool_hang and transport,
# plus 1: rollouts and checkpoints run every 50 epochs (epochs counted from 0), so the last
# epoch is evaluated and checkpointed. At batch size 64:
#   task        steps/epoch   epochs   steps
#   lift            127         351     44.6k
#   can             335         351    117.6k
#   square          440         601    264.4k
#   tool_hang      1446         151    218.3k
#   transport      1415         151    213.7k
#
# Disk: each checkpoint is ~4.5 GB (the UNet dominates); with topk 2 + latest, a task keeps
# ~13 GB of checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."    # repo root

export CUDA_VISIBLE_DEVICES=0   # GPU 0 only (training; mujoco-py renders on the CPU)
PYTHON=.venv/bin/python
SEED=42
PROJECT=chunked_policy_robomimic_diffusion_vision
declare -A EPOCHS=([lift]=351 [can]=351 [square]=601 [tool_hang]=151 [transport]=151)

if [ $# -gt 0 ]; then TASKS=("$@"); else TASKS=(lift can square tool_hang transport); fi
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
            policy.vision_encoder=vit \
            training.seed=$SEED \
            training.device=cuda:0 \
            training.num_epochs="${EPOCHS[$task]}" \
            checkpoint.topk.k=2 \
            logging.project=$PROJECT \
            logging.name="diffusion_unet_vision_vit_${task}_ph_seed${SEED}" \
            ${EXTRA:-}; then
        echo "=== $task: training FAILED, continuing with the next task"; failed+=("$task")
    fi
done

echo "=== $(date '+%F %T') done. failed: ${failed[*]:-none}"

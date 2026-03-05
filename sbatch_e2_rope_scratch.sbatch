#!/usr/bin/env bash
#SBATCH --job-name=e2_rope_scratch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=112          # adjust to node's total CPUs (10 per GPU * 8)
#SBATCH --mem=1000G                      # request all memory on the node
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out      # TODO: adjust log path
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=gpu03, gpu05 # will work eirther on 03 or 05 not on both

# to submit a job in terminal run 'sbatch sbatch_e2_rope_scratch.sbatch' command
set -euo pipefail

# ========== TODO: fill in for your node ==========
WORK_DIR="/home/xoren/synth_gen"      # TODO: set working directory on the node
# ==================================================

mkdir -p "${WORK_DIR}/logs"
cd "$WORK_DIR"

# --- Legacy manifest env vars needed by train_synthetic.yaml ---
# (These are commented out in .env; set them here so oc.env resolves them)

# --- WandB: use API key to override any other logged-in account ---
# Get your key from: https://wandb.ai/authorize
export WANDB_API_KEY="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"   # TODO: set your wandb API key
# export WANDB_ENTITY="your-entity"          # optional: set if needed
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
# --- Init NVIDIA driver (fixes error 802 on bare-metal nodes) ---
nvidia-smi -pm 1 2>/dev/null || true

echo "[$(date)] Starting e2 RoPE from-scratch training (8-GPU DDP)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Use torchrun to launch DDP (avoids PL subprocess CUDA init issues)
torchrun --standalone --nproc_per_node=8 run.py --config-name=train_synthetic \
  strategy=ddp \
  'trainer.devices=8' \
  'network.rope_enabled=True' \
  'network.rope_base=10000.0' \
  'loggers.wandb.name=e2_rope_8gpu_scratch'

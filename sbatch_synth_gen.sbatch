#!/usr/bin/env bash
#SBATCH --job-name=synth_100M
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=500G
#SBATCH --time=30-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/home/xoren/synth_gen"

mkdir -p "${WORK_DIR}/logs"
cd "$WORK_DIR"

echo "[$(date)] Starting synthetic data generation"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

# ========== Generation parameters ==========
NUM_SAMPLES=100000080
NUM_WORKERS=180
ANT_ISO_PROB=0.2          # isotropic probability
# ============================================

python unified_runner.py start \
  --data-dir /mnt/weka/xoren/synth_data \
  --workers "$NUM_WORKERS" \
  --num-samples "$NUM_SAMPLES" \
  --ant-iso-prob "$ANT_ISO_PROB"

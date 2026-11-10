#!/bin/bash -l
#SBATCH --job-name=vtab-export
#SBATCH --account=jiminxiao
#SBATCH --partition=gpu4090,gpu40901t --qos=4gpus
#SBATCH --cpus-per-task=16 --mem=64G --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#
# CPU-only: export the 19 VTAB tasks to uint8 arrays for platonic-rep's vtab_probe.py.
# Output ~/datasets/vtab-224 (~100 GB). Resumes per task (meta.json marks a task done).
#   sbatch scripts/export_vtab_arrays.slurm        (from ~/repos/plato/task_adaptation on hpc-2)
set -euo pipefail
cd ~/repos/plato/task_adaptation
export CUDA_VISIBLE_DEVICES= TFDS_DATA_DIR=~/datasets/tensorflow_datasets
~/.local/bin/uv run python scripts/export_vtab_arrays.py \
  --data_dir ~/datasets/tensorflow_datasets --out_dir ~/datasets/vtab-224 ${TASKS:+--tasks $TASKS}
du -sh ~/datasets/vtab-224

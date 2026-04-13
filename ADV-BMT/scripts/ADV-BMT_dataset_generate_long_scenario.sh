#!/bin/bash
# Generate ADV-BMT dataset for long scenarios (>91 frames).
# Supports sequential, range-based, batch (SLURM), and parallel processing modes.

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Configuration — modify these paths before running
# ============================================================
dir="/path/to/scenario/pkl/folder"        # Input folder containing raw scenario .pkl files
save_dir="/path/to/output/folder"         # Output folder for generated ADV scenarios
TF_mode="all_TF_except_adv"
num_mode=1
window_size=91
seed=42

mkdir -p "$save_dir"

# ============================================================
# Option 1: Process all scenarios sequentially (single GPU)
# ============================================================
python scripts/ADV-BMT_dataset_generate_long_scenario.py \
    --dir "$dir" \
    --save_dir "$save_dir" \
    --TF_mode "$TF_mode" \
    --num_mode "$num_mode" \
    --window_size "$window_size" \
    --seed "$seed"

# ============================================================
# Option 2: Process a specific number of scenarios
# ============================================================
# python scripts/ADV-BMT_dataset_generate_long_scenario.py \
#     --dir "$dir" \
#     --save_dir "$save_dir" \
#     --TF_mode "$TF_mode" \
#     --num_mode "$num_mode" \
#     --window_size "$window_size" \
#     --seed "$seed" \
#     --num_scenario 3

# ============================================================
# Option 3: Process a range of scenarios (e.g., scenarios 50-100)
# ============================================================
# python scripts/ADV-BMT_dataset_generate_long_scenario.py \
#     --dir "$dir" \
#     --save_dir "$save_dir" \
#     --TF_mode "$TF_mode" \
#     --num_mode "$num_mode" \
#     --window_size "$window_size" \
#     --seed "$seed" \
#     --start_idx 50 \
#     --end_idx 100

# ============================================================
# Option 4: Batch processing for SLURM/multi-node jobs
# Use --batch_id and --num_batches to split work across jobs
# Example: 4 jobs processing different portions of the dataset
# ============================================================
# BATCH_ID=${SLURM_ARRAY_TASK_ID:-0}  # Use SLURM array ID or default to 0
# NUM_BATCHES=4
#
# python scripts/ADV-BMT_dataset_generate_long_scenario.py \
#     --dir "$dir" \
#     --save_dir "$save_dir" \
#     --TF_mode "$TF_mode" \
#     --num_mode "$num_mode" \
#     --window_size "$window_size" \
#     --seed "$seed" \
#     --batch_id "$BATCH_ID" \
#     --num_batches "$NUM_BATCHES"

# ============================================================
# Option 5: Parallel processing within a single job (multi-CPU)
# Note: Each worker loads its own GPU model, so use with caution
# ============================================================
# python scripts/ADV-BMT_dataset_generate_long_scenario.py \
#     --dir "$dir" \
#     --save_dir "$save_dir" \
#     --TF_mode "$TF_mode" \
#     --num_mode "$num_mode" \
#     --window_size "$window_size" \
#     --seed "$seed" \
#     --parallel \
#     --num_workers 4

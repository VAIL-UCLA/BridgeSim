#!/bin/bash
# Generate ADV-BMT dataset for a single scenario (quick test / small-scale run).
# For large-scale or long-scenario generation, use ADV-BMT_dataset_generate_long_scenario.sh.

# ============================================================
# Configuration — modify these paths before running
# ============================================================
dir="/path/to/scenario/pkl/folder"        # Input folder containing raw scenario .pkl files
save_dir="/path/to/output/folder"         # Output folder for generated ADV scenarios
TF_mode="all_TF_except_adv"
num_scenario=1
num_mode=1

mkdir -p "$save_dir"

python bmt/ADV-BMT_dataset_generate.py \
    --dir "$dir" \
    --save_dir "$save_dir" \
    --TF_mode "$TF_mode" \
    --num_scenario "$num_scenario" \
    --num_mode "$num_mode"

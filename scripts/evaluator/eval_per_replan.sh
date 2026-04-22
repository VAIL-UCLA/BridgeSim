#!/bin/bash

# Evaluation script that runs different replan rates
# Usage: bash eval_per_replan.sh [model_type] [scenario_path] [gpu_id]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT/bridgesim/evaluation"

# Configuration
GPU_ID="${3:-${GPU_ID:-0}}"
export CUDA_VISIBLE_DEVICES=$GPU_ID
MODEL_TYPE="${1:-diffusiondrivev2}"
SCENARIO_PATH="${2:-}"
OUTPUT_BASE="${REPO_ROOT}/outputs/replan_per_scenario"
TRAFFIC_MODE="log_replay"

# Check if scenario path is provided
if [ -z "$SCENARIO_PATH" ]; then
    echo "Usage: bash eval_per_replan.sh [model_type] [scenario_path] [gpu_id]"
    echo "Error: scenario_path is required"
    exit 1
fi

# Replan rates to evaluate
REPLAN_RATES=(1 5 10 15 20 40)

# Load checkpoint and config paths from model_configs.py
MODEL_INFO=$(python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT/bridgesim/evaluation/models')
from model_configs import get_model_info, list_available_models
try:
    info = get_model_info('$MODEL_TYPE')
    print(info.get('checkpoint', ''))
    print(info.get('config', '') or '')
    print(info.get('plan_anchor_path', '') or '')
except ValueError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(f'Available models: {list_available_models()}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo "Failed to load model configuration. Exiting."
    exit 1
fi

CHECKPOINT_PATH=$(echo "$MODEL_INFO" | sed -n '1p')
CONFIG_PATH=$(echo "$MODEL_INFO" | sed -n '2p')
PLAN_ANCHOR_PATH=$(echo "$MODEL_INFO" | sed -n '3p')

# Extract scenario name from path
SCENARIO_NAME=$(basename "$SCENARIO_PATH")

echo "============================================================"
echo "Replan Rate Evaluation"
echo "Model: $MODEL_TYPE | GPU: $GPU_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
[ -n "$CONFIG_PATH" ] && echo "Config: $CONFIG_PATH"
[ -n "$PLAN_ANCHOR_PATH" ] && echo "Plan Anchor: $PLAN_ANCHOR_PATH"
echo "Scenario: $SCENARIO_PATH"
echo "Replan rates: ${REPLAN_RATES[*]}"
echo "============================================================"

# Loop through each replan rate
for REPLAN_RATE in "${REPLAN_RATES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running evaluation with replan_rate=$REPLAN_RATE"
    echo "============================================================"

    # Output directory includes model type, scenario name, and replan rate
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_TYPE}_${TRAFFIC_MODE}_replan_study/${SCENARIO_NAME}/replan_rate_${REPLAN_RATE}"

    echo "Output: $OUTPUT_DIR"

    # Build command
    CMD="python unified_evaluator.py \
        --model-type \"$MODEL_TYPE\" \
        --checkpoint \"$CHECKPOINT_PATH\" \
        --scenario-path \"$SCENARIO_PATH\" \
        --traffic-mode \"$TRAFFIC_MODE\" \
        --enable-vis \
        --output-dir \"$OUTPUT_DIR\" \
        --eval-mode \"closed_loop\" \
        --controller \"pure_pursuit\" \
        --replan-rate \"$REPLAN_RATE\" \
        --ego-replay-frames \"20\" \
        --scorer-type \"navsim\" \
        --enable-temporal-consistency"

    [ -n "$CONFIG_PATH" ] && CMD="$CMD --config \"$CONFIG_PATH\""
    [ -n "$PLAN_ANCHOR_PATH" ] && CMD="$CMD --plan-anchor-path \"$PLAN_ANCHOR_PATH\""

    eval $CMD

    if [ $? -ne 0 ]; then
        echo "WARNING: Evaluation failed for replan_rate=$REPLAN_RATE"
    else
        echo "Completed replan_rate=$REPLAN_RATE"
    fi
done

echo ""
echo "============================================================"
echo "All replan rate evaluations complete!"
echo "Results saved to: ${OUTPUT_BASE}/${MODEL_TYPE}_${TRAFFIC_MODE}_replan_study/${SCENARIO_NAME}/"
echo "============================================================"

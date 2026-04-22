#!/bin/bash

# Single scenario evaluation script
# Usage: bash run_eval.sh [model_type] [scenario_path] [gpu_id]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT/bridgesim/evaluation"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/metadrive:$PYTHONPATH"

# Configuration
GPU_ID="${3:-${GPU_ID:-0}}"
MODEL_TYPE="${1:-transfuser}"
SCENARIO_PATH="${2:-}"
OUTPUT_BASE="${REPO_ROOT}/outputs/single_eval"
TRAFFIC_MODE="${TRAFFIC_MODE:-IDM}"   # Options: no_traffic, log_replay, IDM
EVAL_MODE="closed_loop"  # Options: closed_loop, open_loop

# Extra args (passed after gpu_id, forwarded to unified_evaluator.py)
shift 3 2>/dev/null
EXTRA_ARGS="$@"

# Check if scenario path is provided
if [ -z "$SCENARIO_PATH" ]; then
    echo "Usage: bash run_eval.sh [model_type] [scenario_path] [gpu_id]"
    echo "Error: scenario_path is required"
    exit 1
fi

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
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_TYPE}_${TRAFFIC_MODE}"

echo "============================================================"
echo "Model: $MODEL_TYPE | GPU: $GPU_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
[ -n "$CONFIG_PATH" ] && echo "Config: $CONFIG_PATH"
[ -n "$PLAN_ANCHOR_PATH" ] && echo "Plan Anchor: $PLAN_ANCHOR_PATH"
echo "Scenario: $SCENARIO_PATH"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

# Build command
CMD="python unified_evaluator.py \
    --model-type \"$MODEL_TYPE\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --scenario-path \"$SCENARIO_PATH\" \
    --traffic-mode \"$TRAFFIC_MODE\" \
    --enable-vis \
    --output-dir \"$OUTPUT_DIR\" \
    --eval-mode \"$EVAL_MODE\" \
    --controller \"pure_pursuit\" \
    --replan-rate \"20\" \
    --ego-replay-frames \"20\" "

[ -n "$CONFIG_PATH" ] && CMD="$CMD --config \"$CONFIG_PATH\""
[ -n "$PLAN_ANCHOR_PATH" ] && CMD="$CMD --plan-anchor-path \"$PLAN_ANCHOR_PATH\""
[ -n "$EXTRA_ARGS" ] && CMD="$CMD $EXTRA_ARGS"

eval $CMD

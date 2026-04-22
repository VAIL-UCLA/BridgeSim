#!/bin/bash

# Batch replan rate study script
# Runs batch evaluation for multiple replan rates across all scenarios in a directory
# Usage: bash run_batch_replan_study.sh [model_type] [scenario_root] [options]
# Options:
#   --resume              Resume from previous run
#   --ego-replay-frames N Number of ego replay frames (default: 20)
#   --eval-frames N       Number of eval frames (default: 150)
#   --enable-vis          Enable visualization outputs
#   --enable-temporal-consistency  Enable temporal consistency

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT/bridgesim/evaluation"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/metadrive:$PYTHONPATH"

# Configuration
MODEL_TYPE="${1:-diffusiondrivev2}"
SCENARIO_ROOT="${2:-}"
GPU_ID="${GPU_ID:-0}"  # default; overridden by --gpu-id flag
OUTPUT_BASE="${REPO_ROOT}/outputs/replan_study"
TRAFFIC_MODE="log_replay"

# Check if scenario root is provided
if [ -z "$SCENARIO_ROOT" ]; then
    echo "Usage: bash run_batch_replan_study.sh [model_type] [scenario_root] [options]"
    echo "Error: scenario_root is required"
    exit 1
fi

# Replan rates to evaluate
REPLAN_RATES=(5)

# Default values for options
RESUME_FLAG=""
EGO_REPLAY_FRAMES="20"
EVAL_FRAMES="200"
ENABLE_VIS=""
ENABLE_TEMPORAL_CONSISTENCY=""
EXTRA_ARGS=""

# Parse flags (can be in any position)
shift 2 2>/dev/null  # Skip first two positional args if they exist
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --ego-replay-frames)
            EGO_REPLAY_FRAMES="$2"
            shift 2
            ;;
        --eval-frames)
            EVAL_FRAMES="$2"
            shift 2
            ;;
        --enable-vis)
            ENABLE_VIS="--enable-vis"
            shift
            ;;
        --enable-temporal-consistency)
            ENABLE_TEMPORAL_CONSISTENCY="--enable-temporal-consistency"
            shift
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

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

echo "============================================================"
echo "Batch Replan Rate Study"
echo "Model: $MODEL_TYPE | GPU: $GPU_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
[ -n "$CONFIG_PATH" ] && echo "Config: $CONFIG_PATH"
[ -n "$PLAN_ANCHOR_PATH" ] && echo "Plan Anchor: $PLAN_ANCHOR_PATH"
echo "Scenario Root: $SCENARIO_ROOT"
echo "Replan rates: ${REPLAN_RATES[*]}"
echo "Ego Replay Frames: $EGO_REPLAY_FRAMES"
echo "Eval Frames: $EVAL_FRAMES"
[ -n "$EXTRA_ARGS" ] && echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# Loop through each replan rate
for REPLAN_RATE in "${REPLAN_RATES[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running batch evaluation with replan_rate=$REPLAN_RATE"
    echo "============================================================"

    # Output directory includes model type and replan rate
    OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_TYPE}_${TRAFFIC_MODE}/replan_rate_${REPLAN_RATE}"

    echo "Output: $OUTPUT_DIR"

    # Build command
    CMD="python batch_evaluator.py \
        --model-type \"$MODEL_TYPE\" \
        --checkpoint \"$CHECKPOINT_PATH\" \
        --scenario-root \"$SCENARIO_ROOT\" \
        --output-dir \"$OUTPUT_DIR\" \
        --traffic-mode \"$TRAFFIC_MODE\" \
        --no-save-perframe \
        --max-workers 1 \
        --replan-rate \"$REPLAN_RATE\" \
        --ego-replay-frames \"$EGO_REPLAY_FRAMES\" \
        --eval-frames \"$EVAL_FRAMES\" \
        --eval-mode closed_loop \
        --controller pure_pursuit \
        --enable-temporal-consistency \
        $ENABLE_VIS"
        

    [ -n "$CONFIG_PATH" ] && CMD="$CMD --config \"$CONFIG_PATH\""
    [ -n "$PLAN_ANCHOR_PATH" ] && CMD="$CMD --plan-anchor-path \"$PLAN_ANCHOR_PATH\""
    [ -n "$EXTRA_ARGS" ] && CMD="$CMD $EXTRA_ARGS"

    eval $CMD

    if [ $? -ne 0 ]; then
        echo "WARNING: Batch evaluation failed for replan_rate=$REPLAN_RATE"
    else
        echo "Completed batch evaluation for replan_rate=$REPLAN_RATE"
    fi
done

echo ""
echo "============================================================"
echo "All replan rate batch evaluations complete!"
echo "Results saved to: ${OUTPUT_BASE}/${MODEL_TYPE}_${TRAFFIC_MODE}/"
echo "============================================================"

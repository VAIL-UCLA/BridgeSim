#!/bin/bash

# Batch evaluation script
# Usage: bash run_batch_eval.sh [model_type] [scenario_root] [options]
# Options:
#   --resume              Resume from previous run
#   --replan-rate N       Replan rate (default: 1)
#   --sim-dt F            Simulation timestep (default: 0.1)
#   --ego-replay-frames N Number of ego replay frames (default: 0)
#   --eval-frames N       Number of eval frames (default: all)
#   --scorer-type TYPE    Scorer type: legacy or navsim (default: legacy)
#   --score-start-frame N Frame to start scoring (default: ego-replay-frames)
#   --eval-mode MODE      Eval mode: closed_loop or open_loop (default: closed_loop)
#   --enable-vis          Enable visualization outputs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT/bridgesim/evaluation"

# Configuration
MODEL_TYPE="${1:-uniad}"
SCENARIO_ROOT="${2:-}"
OUTPUT_BASE="${REPO_ROOT}/outputs/batch_eval"
TRAFFIC_MODE="log_replay"

# Check if scenario root is provided
if [ -z "$SCENARIO_ROOT" ]; then
    echo "Usage: bash run_batch_eval.sh [model_type] [scenario_root] [options]"
    echo "Error: scenario_root is required"
    exit 1
fi

# Default values for new options (include flag prefix)
REPLAN_RATE="--replan-rate 10"
SIM_DT=""
EGO_REPLAY_FRAMES="--ego-replay-frames 20"
EVAL_FRAMES="--eval-frames 200"
SCORER_TYPE="--scorer-type navsim"
SCORE_START_FRAME=""
EVAL_MODE=""
ENABLE_VIS=""

# Parse flags (can be in any position)
RESUME_FLAG=""
shift 2 2>/dev/null  # Skip first two positional args if they exist
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --replan-rate)
            REPLAN_RATE="--replan-rate $2"
            shift 2
            ;;
        --sim-dt)
            SIM_DT="--sim-dt $2"
            shift 2
            ;;
        --ego-replay-frames)
            EGO_REPLAY_FRAMES="--ego-replay-frames $2"
            shift 2
            ;;
        --eval-frames)
            EVAL_FRAMES="--eval-frames $2"
            shift 2
            ;;
        --scorer-type)
            SCORER_TYPE="--scorer-type $2"
            shift 2
            ;;
        --score-start-frame)
            SCORE_START_FRAME="--score-start-frame $2"
            shift 2
            ;;
        --eval-mode)
            EVAL_MODE="--eval-mode $2"
            shift 2
            ;;
        --enable-vis)
            ENABLE_VIS="--enable-vis"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

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
echo "Batch Evaluation | Model: $MODEL_TYPE"
echo "Checkpoint: $CHECKPOINT_PATH"
[ -n "$CONFIG_PATH" ] && echo "Config: $CONFIG_PATH"
[ -n "$PLAN_ANCHOR_PATH" ] && echo "Plan Anchor: $PLAN_ANCHOR_PATH"
echo "Scenario Root: $SCENARIO_ROOT"
echo "Output: $OUTPUT_DIR"
[ -n "$RESUME_FLAG" ] && echo "Resume: enabled"
[ -n "$REPLAN_RATE" ] && echo "Replan Rate: $REPLAN_RATE"
[ -n "$SIM_DT" ] && echo "Sim DT: $SIM_DT"
[ -n "$EGO_REPLAY_FRAMES" ] && echo "Ego Replay Frames: $EGO_REPLAY_FRAMES"
[ -n "$EVAL_FRAMES" ] && echo "Eval Frames: $EVAL_FRAMES"
[ -n "$SCORER_TYPE" ] && echo "Scorer Type: $SCORER_TYPE"
[ -n "$SCORE_START_FRAME" ] && echo "Score Start Frame: $SCORE_START_FRAME"
[ -n "$EVAL_MODE" ] && echo "Eval Mode: $EVAL_MODE"
[ -n "$ENABLE_VIS" ] && echo "Visualization: enabled"
echo "============================================================"

# Build command
CMD="python batch_evaluator.py \
    --model-type \"$MODEL_TYPE\" \
    --checkpoint \"$CHECKPOINT_PATH\" \
    --scenario-root \"$SCENARIO_ROOT\" \
    --output-dir \"$OUTPUT_DIR\" \
    --traffic-mode \"$TRAFFIC_MODE\" \
    --no-save-perframe \
    --max-workers 1 \
    $RESUME_FLAG $REPLAN_RATE $SIM_DT $EGO_REPLAY_FRAMES $EVAL_FRAMES $SCORER_TYPE $SCORE_START_FRAME $EVAL_MODE $ENABLE_VIS"

[ -n "$CONFIG_PATH" ] && CMD="$CMD --config \"$CONFIG_PATH\""
[ -n "$PLAN_ANCHOR_PATH" ] && CMD="$CMD --plan-anchor-path \"$PLAN_ANCHOR_PATH\""

eval $CMD

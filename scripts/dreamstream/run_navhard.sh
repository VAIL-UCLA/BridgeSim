#!/usr/bin/env bash
#
# Closed-loop evaluation of a front-camera policy on Navhard with DreamStream
# observations, using the DreamStream paper protocol:
#   IDM traffic, 2 s ego replay (20 frames), replan every 5 frames (0.5 s),
#   8 s scored horizon (80 frames).
#
# Usage:
#   bash scripts/dreamstream/run_navhard.sh <model_type> <scenario_root> [extra batch_evaluator args]
#
#   <model_type>     drivor_front | diffusiondrive_front | diffusiondrivev2_front |
#                    ltf_front | rap_front | sparsedrivev2_front
#   <scenario_root>  directory of converted scenarios (e.g. the 421 Navhard scenes)
#
# Checkpoints are resolved from bridgesim/evaluation/models/model_configs.py
# ($DREAMSTREAM_POLICY_ROOT, default ckpts/DreamStream/policy).
#
# Common extra args:
#   --server-urls URL[,URL...]  DreamStream server(s) (default http://127.0.0.1:8790);
#                               pass one URL per worker together with --max-workers N
#   --max-workers N             run N scenarios in parallel
#   --no-world-model            evaluate on MetaDrive RGB instead of DreamStream
#   --output-dir DIR            default: outputs/dreamstream_navhard
#   --resume                    skip scenarios that already have results
#   --enable-vis                save per-frame images and videos
#
# Results: batch_driving_score_summary.csv in the run's subdirectory of
# <output-dir> (one row per scenario plus an AVERAGE row with DS, EPDMS_no_ep and RC).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '2,30p' "$0"
  exit 1
fi

MODEL_TYPE=$1
SCENARIO_ROOT=$(realpath "$2")
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="$REPO_ROOT/outputs/dreamstream_navhard"
WM_ARGS=(--use-world-model)
EXTRA=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-world-model) WM_ARGS=(); shift ;;
    --output-dir) OUTPUT_DIR=$(realpath -m "$2"); shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# Checkpoint / plan anchor for this model from model_configs.py
mapfile -t INFO < <(python - "$MODEL_TYPE" <<'EOF'
import sys
from bridgesim.evaluation.models.model_configs import get_model_info
info = get_model_info(sys.argv[1])
print(info.get("checkpoint") or "")
print(info.get("plan_anchor_path") or "")
EOF
)
CHECKPOINT=${INFO[0]}
PLAN_ANCHOR=${INFO[1]:-}
[[ -e "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT (see docs/dreamstream.md)"; exit 1; }

ANCHOR_ARGS=()
[[ -n "$PLAN_ANCHOR" ]] && ANCHOR_ARGS=(--plan-anchor-path "$PLAN_ANCHOR")

set -x
python bridgesim/evaluation/batch_evaluator.py \
  --model-type "$MODEL_TYPE" \
  --checkpoint "$CHECKPOINT" \
  "${ANCHOR_ARGS[@]}" \
  --scenario-root "$SCENARIO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --traffic-mode IDM \
  --ego-replay-frames 20 \
  --replan-rate 5 \
  --eval-frames 80 \
  "${WM_ARGS[@]}" \
  "${EXTRA[@]}"

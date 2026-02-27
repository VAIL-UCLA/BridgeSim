set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
EVAL_PY="$REPO_ROOT/bridgesim/evaluation/unified_evaluator.py"

MODEL_TYPE="${MODEL_TYPE:-alpamayo_r1}"
CHECKPOINT="${CHECKPOINT:-nvidia/Alpamayo-R1-10B}"
SCENARIO_ROOT="${SCENARIO_ROOT:-/data/aidan/scenarios/navtest_converted}"
OUT_ROOT="${OUT_ROOT:-/data/aidan/BridgeSim/outputs/alpamayo_sweep_navtest}"
TRAFFIC_MODE="${TRAFFIC_MODE:-log_replay}"
EVAL_MODE="${EVAL_MODE:-closed_loop}"
CONTROLLER="${CONTROLLER:-pure_pursuit}"
REPLAN_RATE="${REPLAN_RATE:-10}"
EGO_REPLAY_FRAMES="${EGO_REPLAY_FRAMES:-20}"

EVAL_METRICS="${EVAL_METRICS:-pdms}"

# 8/12/16 seconds => 80/120/160 eval frames
SECONDS_LIST=(8 12 16)

SCENES=(
  "sd_01052f67ec24538c"
  "sd_012dc5d8043555ef"
  "sd_0179d579d30e588c"
  "sd_018690bcb255590d"
  "sd_01d061d9a66451ea"
  "sd_020dee65dab453bb"
  "sd_025b657634505df3"
  "sd_026e7b1e0e335625"
  "sd_02902d180b405100"
  "sd_0fde069313a35062"
)

mkdir -p "$OUT_ROOT"

SUMMARY="$OUT_ROOT/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
  echo "scene,seconds,eval_frames,exit_code,final_score,mean_metric,route_completion,valid_frames,results_dir,log_path" > "$SUMMARY"
fi

run_one () {
  local scene="$1"
  local seconds="$2"
  local eval_frames="$3"

  local scenario_path="$SCENARIO_ROOT/$scene"
  local run_out="$OUT_ROOT/$scene/${seconds}s"
  mkdir -p "$run_out"

  local log_path="$run_out/run.log"

  echo "============================================================"
  echo "Running scene=$scene seconds=$seconds eval_frames=$eval_frames"
  echo "  scenario_path=$scenario_path"
  echo "  output_dir=$run_out"
  echo "  log=$log_path"
  echo "------------------------------------------------------------"

  set +e
  "$PY" "$EVAL_PY" \
    --model-type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --scenario-path "$scenario_path" \
    --traffic-mode "$TRAFFIC_MODE" \
    --output-dir "$run_out" \
    --eval-mode "$EVAL_MODE" \
    --controller "$CONTROLLER" \
    --replan-rate "$REPLAN_RATE" \
    --ego-replay-frames "$EGO_REPLAY_FRAMES" \
    --eval-frames "$eval_frames" \
    --eval-metrics "$EVAL_METRICS" \
    --enable-vis \
    2>&1 | tee "$log_path"
  local exit_code="${PIPESTATUS[0]}"
  set -e

  # Parse metrics from log
  # Expected lines include:
  #   Mean PDMS (no EP):            0.7211
  #   Mean EPDMS (no EP):           0.7211
  #   Route Completion:              0.4782
  #   FINAL SCORE: 0.7211 × 0.4782 = 0.3449
  #   Per-frame metrics (averaged over 74 valid frames):

  local final_score
  final_score="$(awk '
    /FINAL SCORE:/ {
      for (i=NF; i>=1; --i) {
        if ($i ~ /^[0-9]+(\.[0-9]+)?$/) { print $i; exit }
      }
    }' "$log_path")"
  final_score="${final_score:-NA}"

  local mean_metric
  mean_metric="$(awk '
    /Mean (P|EP)DMS \(no EP\):/ {print $NF}
  ' "$log_path" | tail -n 1)"
  mean_metric="${mean_metric:-NA}"

  local route_completion
  route_completion="$(awk '/Route Completion:/ {print $NF}' "$log_path" | tail -n 1)"
  route_completion="${route_completion:-NA}"

  local valid_frames
  valid_frames="$(awk '
    /Per-frame metrics \(averaged over/ {
      for (i=1; i<=NF; ++i) if ($i ~ /^[0-9]+$/) { print $i; exit }
    }' "$log_path" | tail -n 1)"
  valid_frames="${valid_frames:-NA}"

  local results_dir
  results_dir="$(awk -F': ' '/Results saved to:/ {print $2}' "$log_path" | tail -n 1)"
  results_dir="${results_dir:-$run_out}"

  echo "${scene},${seconds},${eval_frames},${exit_code},${final_score},${mean_metric},${route_completion},${valid_frames},${results_dir},${log_path}" >> "$SUMMARY"

  echo ">> Recorded: final_score=$final_score mean_metric=$mean_metric route_completion=$route_completion valid_frames=$valid_frames exit_code=$exit_code"
  echo ">> Results dir: $results_dir"
}

main () {
  for scene in "${SCENES[@]}"; do
    for seconds in "${SECONDS_LIST[@]}"; do
      eval_frames=$((seconds * 10))  # 10Hz
      run_one "$scene" "$seconds" "$eval_frames"
    done
  done

  echo "============================================================"
  echo "Done. Summary saved to: $SUMMARY"
  echo "Outputs under: $OUT_ROOT"
}

main "$@"
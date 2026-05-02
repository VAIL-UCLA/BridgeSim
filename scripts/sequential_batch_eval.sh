#!/usr/bin/env bash
#
# Batch-evaluate a dataset of scenarios with the unified evaluator and aggregate
# the per-scenario driving_score_summary.csv AVERAGE rows into a single report.
#
# All configuration lives in this file — edit the CONFIG block below, then run:
#   bash scripts/batch_eval.sh
#
# Dataset layout expected (each subdirectory is one scenario):
#   <DATASET_PATH>/
#     <scenario_A>/<scenario_A>_0/*.pkl
#     <scenario_B>/<scenario_B>_0/*.pkl
#     ...
#
# The unified evaluator writes:
#   <OUTPUT_PATH>/<model_subdir>/<scenario_name>/driving_score_summary.csv
# where <model_subdir> is derived from --model-type, --replan-rate, etc.
# The final row of each CSV is the per-scenario AVERAGE. This script finds all
# such CSVs under <OUTPUT_PATH> and averages their AVERAGE rows into
# <OUTPUT_PATH>/aggregate_summary.csv.

set -euo pipefail

# ============================ CONFIG ==========================================
# Required
DATASET_PATH="$(pwd)/local_scenarios/navhard"
OUTPUT_PATH="$(pwd)/outputs/pdm_closed_batch"

# Arguments passed verbatim to bridgesim/evaluation/unified_evaluator.py
# (everything except --scenario-path and --output-dir, which this script sets).
# Edit these to match the model you want to evaluate.
EVAL_ARGS=(
  --model-type pdm_closed
  --checkpoint none
  --traffic-mode log_replay
  --eval-mode closed_loop
  --replan-rate 5
  --ego-replay-frames 20
  --eval-frames 80
  --enable-vis
)
# ==============================================================================

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset path does not exist: $DATASET_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/bridgesim/evaluation/unified_evaluator.py"

shopt -s nullglob
SCENARIOS=()
for entry in "$DATASET_PATH"/*/; do
  scenario_name="$(basename "$entry")"
  # A valid scenario has a <name>/<name>_0/*.pkl subfolder.
  if compgen -G "${entry}${scenario_name}_0/*.pkl" > /dev/null; then
    SCENARIOS+=("${entry%/}")
  fi
done
shopt -u nullglob

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  echo "No scenarios found under $DATASET_PATH" >&2
  exit 1
fi

echo "Found ${#SCENARIOS[@]} scenarios under $DATASET_PATH"

for scenario in "${SCENARIOS[@]}"; do
  scenario_name="$(basename "$scenario")"
  echo "===== Evaluating $scenario_name ====="
  python "$EVAL_SCRIPT" \
    --scenario-path "$scenario" \
    --output-dir "$OUTPUT_PATH" \
    "${EVAL_ARGS[@]}"
done

echo "===== Aggregating results ====="
python - "$OUTPUT_PATH" <<'PYEOF'
import sys
from pathlib import Path
import pandas as pd

output_path = Path(sys.argv[1])
csv_paths = sorted(output_path.rglob("driving_score_summary.csv"))
if not csv_paths:
    print(f"No driving_score_summary.csv files found under {output_path}", file=sys.stderr)
    sys.exit(1)

rows = []
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    avg_row = df[df["frame_id"].astype(str) == "AVERAGE"]
    if avg_row.empty:
        print(f"[skip] no AVERAGE row: {csv_path}", file=sys.stderr)
        continue
    row = avg_row.iloc[0].to_dict()
    # scenario_name is the parent dir; model_subdir is the grandparent.
    row["scenario"] = csv_path.parent.name
    row["model_subdir"] = csv_path.parent.parent.name
    rows.append(row)

if not rows:
    print("No AVERAGE rows collected — nothing to aggregate.", file=sys.stderr)
    sys.exit(1)

per_scenario = pd.DataFrame(rows)

metric_cols = [c for c in ["NC", "DAC", "DDC", "TL", "TTC", "LK", "HC", "EC",
                           "DS", "EPDMS_no_ep", "RC"] if c in per_scenario.columns]
for c in metric_cols:
    per_scenario[c] = pd.to_numeric(per_scenario[c], errors="coerce")

lead_cols = ["model_subdir", "scenario"] + metric_cols
per_scenario = per_scenario[lead_cols]

overall = per_scenario[metric_cols].mean(numeric_only=True).round(6).to_dict()
overall_row = {"model_subdir": "ALL", "scenario": "AVERAGE", **overall}
final = pd.concat([per_scenario, pd.DataFrame([overall_row])], ignore_index=True)

out_csv = output_path / "aggregate_summary.csv"
final.to_csv(out_csv, index=False)

print(f"Wrote aggregate summary to {out_csv}")
print(f"Scenarios aggregated: {len(per_scenario)}")
print("Overall averages:")
for k, v in overall.items():
    print(f"  {k:>12}: {v:.6f}" if v == v else f"  {k:>12}: nan")
PYEOF

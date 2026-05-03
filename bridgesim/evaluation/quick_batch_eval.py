"""
Quick batch evaluator: scans all scenarios, picks a diverse N-scenario subset,
and runs them through the standard evaluation pipeline.

Selection uses greedy max-min diversity sampling over normalised feature vectors:
  [n_vehicles, n_pedestrians, n_cyclists, has_traffic_lights, ego_dist_m]

This guarantees representative coverage (dense traffic, many pedestrians, TL
scenarios, cyclists, short/long routes) without needing manual curation.

Usage:
    python bridgesim/evaluation/quick_batch_eval.py \
        --model-type pdm_closed --checkpoint none \
        --scenario-root local_scenarios/navhard \
        --output-dir outputs/quick \
        --num-scenarios 15
"""

import os
import sys
import pickle  # local scenario files (MetaDrive ScenarioDescription) require pickle
import argparse
import subprocess
import time
import csv as csv_mod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(scenario_dir: Path) -> Optional[Dict]:
    """Return a feature dict for one scenario, or None if unreadable."""
    inner_dirs = [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.endswith("_0")]
    if not inner_dirs:
        return None
    pkl_path = inner_dirs[0] / f"{scenario_dir.name}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    sdc_id = data["metadata"]["sdc_id"]
    tracks = data["tracks"]

    type_counts: Dict[str, int] = {}
    for t in tracks.values():
        tp = str(t.get("type", "UNKNOWN"))
        type_counts[tp] = type_counts.get(tp, 0) + 1

    ego = tracks.get(sdc_id, {})
    pos = ego.get("state", {}).get("position")
    valid = ego.get("state", {}).get("valid")
    ego_dist = 0.0
    if pos is not None and valid is not None:
        pos_arr = np.array(pos)
        valid_arr = np.array(valid, dtype=bool)
        valid_pos = pos_arr[valid_arr]
        if len(valid_pos) >= 2:
            ego_dist = float(np.sum(np.linalg.norm(np.diff(valid_pos[:, :2], axis=0), axis=1)))

    has_tl = len(data.get("dynamic_map_states", {})) > 0

    return {
        "path": scenario_dir,
        "n_vehicles": type_counts.get("VEHICLE", 0),
        "n_pedestrians": type_counts.get("PEDESTRIAN", 0),
        "n_cyclists": type_counts.get("CYCLIST", 0),
        "has_tl": int(has_tl),
        "ego_dist": ego_dist,
    }


def scan_scenarios(scenario_root: Path) -> List[Dict]:
    """Scan all scenario directories and extract features."""
    dirs = sorted([d for d in scenario_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"[quick_eval] Scanning {len(dirs)} scenarios for features...", flush=True)
    features = []
    for d in dirs:
        try:
            f = _extract_features(d)
            if f is not None:
                features.append(f)
        except Exception:
            pass
    print(f"[quick_eval] Successfully scanned {len(features)} scenarios.", flush=True)
    return features


# ---------------------------------------------------------------------------
# Diversity sampling
# ---------------------------------------------------------------------------

def _feature_matrix(scenarios: List[Dict]) -> np.ndarray:
    """Build normalised (N, 5) feature matrix."""
    mat = np.array([
        [s["n_vehicles"], s["n_pedestrians"], s["n_cyclists"], s["has_tl"], s["ego_dist"]]
        for s in scenarios
    ], dtype=float)
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
    return (mat - col_min) / span


def greedy_diverse_sample(scenarios: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    Greedy max-min diversity: start with a random seed scenario, then
    repeatedly add the scenario whose minimum distance to the current
    selection is largest.
    """
    if n >= len(scenarios):
        return list(scenarios)

    mat = _feature_matrix(scenarios)
    rng = np.random.default_rng(seed)

    selected_indices = [int(rng.integers(len(scenarios)))]
    remaining = set(range(len(scenarios))) - set(selected_indices)

    while len(selected_indices) < n and remaining:
        sel_mat = mat[selected_indices]
        remaining_list = list(remaining)
        rem_mat = mat[remaining_list]
        dists = np.min(
            np.sqrt(((rem_mat[:, None, :] - sel_mat[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        best_idx = remaining_list[int(np.argmax(dists))]
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [scenarios[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Evaluation runner (mirrors batch_evaluator.py subprocess logic)
# ---------------------------------------------------------------------------

def _build_cmd(scenario_path: Path, args) -> List[str]:
    cmd = [
        sys.executable,
        "unified_evaluator.py",
        "--model-type", args.model_type,
        "--checkpoint", args.checkpoint,
        "--scenario-path", str(scenario_path),
        "--traffic-mode", args.traffic_mode,
        "--output-dir", args.output_dir,
        "--controller", args.controller,
        "--replan-rate", str(args.replan_rate),
        "--sim-dt", str(args.sim_dt),
        "--ego-replay-frames", str(args.ego_replay_frames),
        "--eval-mode", args.eval_mode,
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.eval_frames is not None:
        cmd.extend(["--eval-frames", str(args.eval_frames)])
    if args.score_start_frame is not None:
        cmd.extend(["--score-start-frame", str(args.score_start_frame)])
    if not args.save_perframe:
        cmd.append("--no-save-perframe")
    if args.enable_vis:
        cmd.append("--enable-vis")
    return cmd


def _output_subdir(args) -> str:
    ef = str(args.eval_frames) if args.eval_frames is not None else "None"
    return f"{args.model_type}_rr{args.replan_rate}_erf{args.ego_replay_frames}_ef{ef}"


def run_scenario(scenario_path: Path, args) -> Dict:
    cmd = _build_cmd(scenario_path, args)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent,
            timeout=600,
        )
        duration = time.time() - start
        if result.returncode == 0:
            summary_csv = (
                Path(args.output_dir)
                / _output_subdir(args)
                / scenario_path.name
                / "driving_score_summary.csv"
            )
            if summary_csv.exists():
                with open(summary_csv) as f:
                    for row in csv_mod.DictReader(f):
                        if row.get("frame_id") == "AVERAGE":
                            return {"status": "success", "duration": duration, "scores": dict(row)}
            return {"status": "error", "duration": duration, "error": "no summary CSV"}
        return {"status": "failed", "duration": duration, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "duration": time.time() - start}
    except Exception as e:
        return {"status": "error", "duration": time.time() - start, "error": str(e)}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

COLUMNS = ["DS", "EPDMS_no_ep", "RC", "NC", "DAC", "DDC", "TL", "TTC", "LK", "HC", "EC"]


def print_summary(results: List[Tuple[str, Dict]]):
    totals = {c: [] for c in COLUMNS}
    for name, r in results:
        if r["status"] != "success":
            print(f"  SKIP {name}: {r.get('error', r['status'])}")
            continue
        for c in COLUMNS:
            try:
                totals[c].append(float(r["scores"][c]))
            except (KeyError, ValueError, TypeError):
                pass

    print(f"\n{'='*60}")
    print("Quick Eval Complete! Summary (Averages)")
    print(f"{'='*60}")
    for c in COLUMNS:
        vals = totals[c]
        print(f"  {c:<12} {np.mean(vals):.6f}" if vals else f"  {c:<12} N/A")
    n_ok = sum(1 for _, r in results if r["status"] == "success")
    print(f"{'='*60}")
    print(f"  Scenarios evaluated: {n_ok} / {len(results)}")


def export_csv(results: List[Tuple[str, Dict]], output_dir: str):
    out_path = Path(output_dir) / "quick_batch_driving_score_summary.csv"
    cols = ["scenario"] + COLUMNS
    totals = {c: [] for c in COLUMNS}
    rows = []
    for name, r in results:
        if r["status"] != "success":
            continue
        scores = r.get("scores", {})
        row = {"scenario": name}
        for c in COLUMNS:
            val = scores.get(c, "")
            row[c] = val
            try:
                totals[c].append(float(val))
            except (ValueError, TypeError):
                pass
        rows.append(row)

    avg_row = {"scenario": "AVERAGE"}
    for c in COLUMNS:
        vals = totals[c]
        avg_row[c] = f"{np.mean(vals):.6f}" if vals else ""

    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(avg_row)
    print(f"[quick_eval] Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick diverse-subset batch evaluator")

    parser.add_argument("--model-type", required=True,
                        choices=["uniad", "vad", "tcp", "rap", "lead", "lead_navsim", "drivor",
                                 "transfuser", "ltf", "egomlp", "ego_mlp", "diffusiondrive",
                                 "diffusiondrivev2", "openpilot", "alpamayo_r1",
                                 "pdm_closed", "pdm_closed2"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--output-dir", required=True)

    # Subset selection
    parser.add_argument("--num-scenarios", type=int, default=15,
                        help="Number of diverse scenarios to run (default: 15, ~4-5 min)")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Random seed for diversity sampling (default: 42)")

    # Standard eval options (mirrors batch_evaluator.py)
    parser.add_argument("--config", default=None)
    parser.add_argument("--traffic-mode", default="log_replay",
                        choices=["no_traffic", "log_replay", "IDM"])
    parser.add_argument("--controller", default="pure_pursuit", choices=["pid", "pure_pursuit"])
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None)
    parser.add_argument("--eval-mode", default="closed_loop",
                        choices=["closed_loop", "open_loop"])
    parser.add_argument("--save-perframe", action="store_true", default=True)
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    parser.add_argument("--enable-vis", action="store_true",
                        help="Enable visualization outputs (images, topdown views)")

    args = parser.parse_args()

    scenario_root = Path(args.scenario_root)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Scan and select
    all_scenarios = scan_scenarios(scenario_root)
    if not all_scenarios:
        print("[quick_eval] No scenarios found.")
        sys.exit(1)

    selected = greedy_diverse_sample(all_scenarios, args.num_scenarios, seed=args.selection_seed)

    print(f"\n[quick_eval] Selected {len(selected)} scenarios:")
    for s in selected:
        print(f"  {s['path'].name}  veh={s['n_vehicles']:3d} ped={s['n_pedestrians']:3d} "
              f"cyc={s['n_cyclists']} tl={bool(s['has_tl'])} dist={s['ego_dist']:.0f}m")
    print()

    # 2. Run
    results: List[Tuple[str, Dict]] = []
    for i, scenario in enumerate(selected):
        name = scenario["path"].name
        print(f"[{i+1:2d}/{len(selected)}] {name} ...", end=" ", flush=True)
        r = run_scenario(scenario["path"], args)
        duration = r.get("duration", 0.0)
        if r["status"] == "success":
            ds = r.get("scores", {}).get("DS", "?")
            ec = r.get("scores", {}).get("EC", "?")
            print(f"OK  DS={ds}  EC={ec}  ({duration:.0f}s)")
        else:
            print(f"FAIL ({r['status']})  ({duration:.0f}s)")
        results.append((name, r))

    # 3. Summary
    print_summary(results)
    export_csv(results, args.output_dir)


if __name__ == "__main__":
    main()

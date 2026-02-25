#!/usr/bin/env python3
"""
Standalone script to scan for evaluation_results.json files under a directory
and aggregate them, replicating the logic from batch_evaluator.py.

Produces:
  - aggregated_results.json  (legacy scores, infractions, planning metrics)
  - evaluation_summary.csv   (per-scenario + average legacy & EPDMS scores)

Usage:
  python aggregate_results.py --input-dir /path/to/eval/outputs --output-dir /path/to/save
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


def find_evaluation_results(input_dir: Path) -> List[Path]:
    """Recursively find all evaluation_results.json files under input_dir."""
    results = sorted(input_dir.rglob("evaluation_results.json"))
    # Exclude any aggregated_results.json that might also match
    results = [r for r in results if r.name == "evaluation_results.json"]
    return results


def aggregate_results(result_files: List[Path], output_path: str) -> None:
    """
    Aggregate individual scenario results into a final JSON.

    Computes global statistics from multiple scenario evaluation results,
    including driving scores, infractions per km, and planning metrics (L2 errors).
    """
    if not result_files:
        print("No results to aggregate")
        return

    # Load all individual results
    records = []
    for idx, result_file in enumerate(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if '_checkpoint' in data and 'records' in data['_checkpoint']:
                    record = data['_checkpoint']['records'][0]
                    record['index'] = idx
                    records.append(record)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    if not records:
        print("No valid records found")
        return

    total_scenarios = len(records)

    # Aggregate infractions
    infractions_sum = {}
    for record in records:
        for key, value in record['infractions'].items():
            if key not in infractions_sum:
                infractions_sum[key] = 0
            infractions_sum[key] += len(value) if isinstance(value, list) else value

    # Aggregate scores
    score_route_values = [r['scores']['score_route'] for r in records]
    score_penalty_values = [r['scores']['score_penalty'] for r in records]
    score_composed_values = [r['scores']['score_composed'] for r in records]

    score_route_mean = np.mean(score_route_values)
    score_penalty_mean = np.mean(score_penalty_values)
    score_composed_mean = np.mean(score_composed_values)

    score_route_std = np.std(score_route_values) if len(score_route_values) > 1 else 0
    score_penalty_std = np.std(score_penalty_values) if len(score_penalty_values) > 1 else 0
    score_composed_std = np.std(score_composed_values) if len(score_composed_values) > 1 else 0

    # Aggregate planning metrics (L2)
    planning_metrics_keys = ['avg_l2_1s', 'avg_l2_2s', 'avg_l2_3s',
                             'l2_0.5s', 'l2_1.0s', 'l2_1.5s', 'l2_2.0s', 'l2_2.5s', 'l2_3.0s']
    planning_metrics_mean = {}
    planning_metrics_std = {}

    for key in planning_metrics_keys:
        values = []
        for r in records:
            if 'planning_metrics' in r and key in r['planning_metrics']:
                values.append(r['planning_metrics'][key])

        if values:
            planning_metrics_mean[key] = round(np.mean(values), 3)
            planning_metrics_std[key] = round(np.std(values), 3) if len(values) > 1 else 0.0
        else:
            planning_metrics_mean[key] = 0.0
            planning_metrics_std[key] = 0.0

    # Calculate total route length and duration
    total_route_length = sum(r['meta']['route_length'] for r in records)
    total_duration_game = sum(r['meta']['duration_game'] for r in records)

    # Calculate km driven (based on route completion)
    km_driven_list = []
    for record in records:
        route_length_km = record['meta']['route_length'] / 1000.0
        completion_pct = record['scores']['score_route'] / 100.0
        km_driven_list.append(route_length_km * completion_pct)
    total_km_driven = max(sum(km_driven_list), 0.001)

    # Collect exceptions
    exceptions = []
    for record in records:
        if 'Failed' in record['status'] or record['entry_status'] != 'Finished':
            exceptions.append((record['route_id'], 0, record['status']))

    # Create global record
    global_record = {
        'index': 0,
        'route_id': -1,
        'status': f"Evaluated {total_scenarios} scenarios",
        'infractions': infractions_sum,
        'scores_mean': {
            'score_route': round(float(score_route_mean), 6),
            'score_penalty': round(float(score_penalty_mean), 6),
            'score_composed': round(float(score_composed_mean), 6)
        },
        'scores_std_dev': {
            'score_route': round(float(score_route_std), 6),
            'score_penalty': round(float(score_penalty_std), 6),
            'score_composed': round(float(score_composed_std), 6)
        },
        'planning_metrics_mean': planning_metrics_mean,
        'planning_metrics_std_dev': planning_metrics_std,
        'meta': {
            'route_length': total_route_length,
            'duration_game': round(total_duration_game, 3),
            'duration_system': 0,
            'exceptions': exceptions
        }
    }

    # Calculate values per km
    ROUND_DIGITS = 3
    values = [
        str(global_record['scores_mean']['score_composed']),
        str(global_record['scores_mean']['score_route']),
        str(global_record['scores_mean']['score_penalty']),
        str(round(global_record['infractions'].get('collisions_pedestrian', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('collisions_vehicle', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('collisions_layout', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('red_light', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('stop_infraction', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('outside_route_lanes', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('route_dev', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('route_timeout', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('vehicle_blocked', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('yield_emergency_vehicle_infractions', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('scenario_timeouts', 0) / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions'].get('min_speed_infractions', 0) / total_km_driven, ROUND_DIGITS)),
    ]

    labels = [
        "Avg. driving score", "Avg. route completion", "Avg. infraction penalty",
        "Collisions with pedestrians", "Collisions with vehicles", "Collisions with layout",
        "Red lights infractions", "Stop sign infractions", "Off-road infractions",
        "Route deviations", "Route timeouts", "Agent blocked",
        "Yield emergency vehicles infractions", "Scenario timeouts", "Min speed infractions",
        "L2 (m) 1s", "L2 (m) 2s", "L2 (m) 3s"
    ]

    values.extend([
        str(planning_metrics_mean.get('avg_l2_1s', 0.0)),
        str(planning_metrics_mean.get('avg_l2_2s', 0.0)),
        str(planning_metrics_mean.get('avg_l2_3s', 0.0))
    ])

    final_results = {
        'entry_status': 'Finished',
        'eligible': True,
        'sensors': [],
        'values': values,
        'labels': labels,
        '_checkpoint': {
            'global_record': global_record,
            'progress': [total_scenarios, total_scenarios],
            'records': records
        }
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\nAggregated results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total scenarios: {total_scenarios}")
    print(f"  Avg. driving score: {global_record['scores_mean']['score_composed']:.6f}")
    print(f"  Avg. route completion: {global_record['scores_mean']['score_route']:.6f}%")
    print(f"  Avg. penalty: {global_record['scores_mean']['score_penalty']:.6f}")
    print(f"  Total km driven: {total_km_driven:.3f} km")
    print(f"  Total infractions: {sum(global_record['infractions'].values())}")
    print(f"\nPlanning Metrics (L2 error in meters):")
    print(f"  L2 @ 1s: {planning_metrics_mean.get('avg_l2_1s', 0.0):.3f} +/- {planning_metrics_std.get('avg_l2_1s', 0.0):.3f}")
    print(f"  L2 @ 2s: {planning_metrics_mean.get('avg_l2_2s', 0.0):.3f} +/- {planning_metrics_std.get('avg_l2_2s', 0.0):.3f}")
    print(f"  L2 @ 3s: {planning_metrics_mean.get('avg_l2_3s', 0.0):.3f} +/- {planning_metrics_std.get('avg_l2_3s', 0.0):.3f}")


def export_results_csv(result_files: List[Path], csv_path: Path) -> None:
    """Export per-scenario CSV with legacy driving scores and EPDMS scores."""
    columns = [
        'scenario_name',
        # Legacy scores
        'driving_score', 'route_completion', 'infraction_penalty',
        # EPDMS scores
        'epdms_score', 'no_at_fault_collisions', 'drivable_area_compliance',
        'driving_direction_compliance', 'traffic_light_compliance',
        'time_to_collision', 'lane_keeping',
        'history_comfort', 'extended_comfort'
    ]

    rows = []
    legacy_totals = {'driving_score': [], 'route_completion': [], 'infraction_penalty': []}
    epdms_totals = {
        'epdms_score': [], 'no_at_fault_collisions': [], 'drivable_area_compliance': [],
        'driving_direction_compliance': [], 'traffic_light_compliance': [],
        'time_to_collision': [], 'lane_keeping': [],
        'history_comfort': [], 'extended_comfort': []
    }

    for results_json in result_files:
        scenario_dir = results_json.parent
        scenario_name = scenario_dir.name
        row = {'scenario_name': scenario_name}

        # Load legacy scores
        try:
            with open(results_json, 'r') as f:
                data = json.load(f)
                if '_checkpoint' in data and 'records' in data['_checkpoint']:
                    record = data['_checkpoint']['records'][0]
                    scores = record.get('scores', {})
                    row['driving_score'] = scores.get('score_composed', 0.0)
                    row['route_completion'] = scores.get('score_route', 0.0)
                    row['infraction_penalty'] = scores.get('score_penalty', 0.0)

                    legacy_totals['driving_score'].append(row['driving_score'])
                    legacy_totals['route_completion'].append(row['route_completion'])
                    legacy_totals['infraction_penalty'].append(row['infraction_penalty'])
        except Exception as e:
            print(f"Error loading legacy scores from {results_json}: {e}")

        # Load EPDMS scores from *_closedloop_epdms_summary.csv
        epdms_files = list(scenario_dir.glob("*_closedloop_epdms_summary.csv"))
        if epdms_files:
            epdms_file = epdms_files[0]
            try:
                with open(epdms_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for epdms_row in reader:
                        row['epdms_score'] = float(epdms_row.get('final_score', 0.0))
                        row['no_at_fault_collisions'] = float(epdms_row.get('mean_no_at_fault_collisions', 0.0))
                        row['drivable_area_compliance'] = float(epdms_row.get('mean_drivable_area_compliance', 0.0))
                        row['driving_direction_compliance'] = float(epdms_row.get('mean_driving_direction_compliance', 0.0))
                        row['traffic_light_compliance'] = float(epdms_row.get('mean_traffic_light_compliance', 0.0))
                        row['time_to_collision'] = float(epdms_row.get('mean_time_to_collision', 0.0))
                        row['lane_keeping'] = float(epdms_row.get('mean_lane_keeping', 0.0))
                        row['history_comfort'] = float(epdms_row.get('mean_history_comfort', 0.0))
                        row['extended_comfort'] = float(epdms_row.get('mean_extended_comfort', 0.0))

                        for key in epdms_totals:
                            if key in row:
                                epdms_totals[key].append(row[key])
                        break
            except Exception as e:
                print(f"Error loading EPDMS scores from {epdms_file}: {e}")

        rows.append(row)

    # Calculate averages
    avg_row = {'scenario_name': 'AVERAGE'}
    for key, values in legacy_totals.items():
        avg_row[key] = np.mean(values) if values else 0.0
    for key, values in epdms_totals.items():
        avg_row[key] = np.mean(values) if values else 0.0

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in columns})
        writer.writerow({k: '' for k in columns})
        writer.writerow({k: f"{avg_row.get(k, ''):.6f}" if isinstance(avg_row.get(k), float) else avg_row.get(k, '') for k in columns})

    print(f"\nExported evaluation summary CSV to: {csv_path}")
    print(f"\n{'='*70}")
    print("Summary (Averages)")
    print(f"{'='*70}")
    print(f"\n--- Legacy Scores ---")
    print(f"  Driving Score:        {avg_row.get('driving_score', 0):.4f}")
    print(f"  Route Completion:     {avg_row.get('route_completion', 0):.4f}%")
    print(f"  Infraction Penalty:   {avg_row.get('infraction_penalty', 0):.4f}")
    print(f"\n--- EPDMS Scores (Closed-Loop) ---")
    print(f"  EPDMS Score (final):  {avg_row.get('epdms_score', 0):.4f}")
    print(f"\n  Binary Metrics:")
    print(f"    No At-Fault Collisions:       {avg_row.get('no_at_fault_collisions', 0):.4f}")
    print(f"    Drivable Area Compliance:     {avg_row.get('drivable_area_compliance', 0):.4f}")
    print(f"    Driving Direction Compliance: {avg_row.get('driving_direction_compliance', 0):.4f}")
    print(f"    Traffic Light Compliance:     {avg_row.get('traffic_light_compliance', 0):.4f}")
    print(f"\n  Weighted Metrics:")
    print(f"    Time to Collision:            {avg_row.get('time_to_collision', 0):.4f}")
    print(f"    Lane Keeping:                 {avg_row.get('lane_keeping', 0):.4f}")
    print(f"    History Comfort:              {avg_row.get('history_comfort', 0):.4f}")
    print(f"    Extended Comfort:             {avg_row.get('extended_comfort', 0):.4f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan for evaluation_results.json files and aggregate them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Aggregate all results under an output directory
  python aggregate_results.py --input-dir outputs/drivor_eval/ --output-dir outputs/drivor_eval/aggregated/

  # Save to a custom location
  python aggregate_results.py --input-dir /data/eval_run_01 --output-dir /data/summaries/run_01
"""
    )

    parser.add_argument('--input-dir', type=str, required=True,
                        help="Root directory to scan for evaluation_results.json files (recursive)")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save aggregated_results.json and evaluation_summary.csv")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all evaluation_results.json files
    result_files = find_evaluation_results(input_dir)
    print(f"Found {len(result_files)} evaluation_results.json files under {input_dir}")

    if not result_files:
        print("Nothing to aggregate. Exiting.")
        sys.exit(0)

    for rf in result_files:
        print(f"  - {rf.relative_to(input_dir)}")

    # Aggregate into JSON
    aggregated_json_path = output_dir / "aggregated_results.json"
    print(f"\n{'='*60}")
    print(f"Aggregating Results from {len(result_files)} Scenarios")
    print(f"{'='*60}")
    aggregate_results(result_files, str(aggregated_json_path))

    # Export CSV
    csv_path = output_dir / "evaluation_summary.csv"
    export_results_csv(result_files, csv_path)


if __name__ == "__main__":
    main()

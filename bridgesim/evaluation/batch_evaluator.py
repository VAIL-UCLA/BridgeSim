"""
Batch evaluator for running evaluation on multiple scenarios.

Features:
- Process all scenarios in a directory
- Progress tracking with ETA
- Error handling and recovery
- Result aggregation
- Optional parallel processing
- Resume capability
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
import time
import numpy as np


class BatchEvaluator:
    """Batch evaluator for multiple scenarios."""

    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 scenario_root: str,
                 output_root: str,
                 config_path: str = None,
                 plan_anchor_path: str = None,
                 traffic_mode: str = "log_replay",
                 max_workers: int = 1,
                 resume: bool = False,
                 save_perframe: bool = True,
                 enable_bev_calibrator: bool = False,
                 bev_calibrator_checkpoint: str = None,
                 bev_sample_steps: int = 50,
                 bev_use_ema: bool = True,
                 controller_type: str = "pure_pursuit",
                 replan_rate: int = 1,
                 sim_dt: float = 0.1,
                 ego_replay_frames: int = 0,
                 eval_frames: int = None,
                 scorer_type: str = "legacy",
                 score_start_frame: int = None,
                 eval_mode: str = "closed_loop",
                 enable_vis: bool = False,
                 enable_temporal_consistency: bool = False,
                 temporal_alpha: float = 1.5,
                 temporal_lambda: float = 0.3,
                 temporal_max_history: int = 8,
                 temporal_sigma: float = 5.0,
                 consensus_temperature: float = 1.0):
        """
        Initialize batch evaluator.

        Args:
            model_type: Model type (uniad, vad, tcp, rap, lead)
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            scenario_root: Root directory containing scenarios
            output_root: Root directory for outputs
            traffic_mode: Traffic mode (no_traffic, log_replay, IDM)
            max_workers: Number of parallel workers (1 = sequential)
            resume: Resume from previous run (skip completed scenarios)
            save_perframe: Save per-frame outputs (planning_traj.npy, etc.)
            enable_bev_calibrator: Enable BEV calibrator for domain adaptation
            bev_calibrator_checkpoint: Path to BEV calibrator checkpoint
            bev_sample_steps: Number of Euler sampling steps for BEV flow matching
            bev_use_ema: Use EMA weights for BEV calibrator
            controller_type: Controller type ('pid' or 'pure_pursuit')
            replan_rate: How often to run model inference (1=every frame)
            sim_dt: Simulation timestep in seconds (default: 0.1s for 10Hz)
            ego_replay_frames: Number of initial frames to replay ego log actions
            eval_frames: Number of frames to evaluate after ego replay ends
            scorer_type: Scorer type ('legacy' or 'navsim')
            score_start_frame: Frame to start calculating scores
            eval_mode: Evaluation mode ('closed_loop' or 'open_loop')
            enable_vis: Enable visualization outputs
        """
        self.model_type = model_type
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.plan_anchor_path = plan_anchor_path
        self.scenario_root = Path(scenario_root)
        self.output_root = Path(output_root)
        self.traffic_mode = traffic_mode
        self.max_workers = max_workers
        self.resume = resume
        self.save_perframe = save_perframe
        self.enable_bev_calibrator = enable_bev_calibrator
        self.bev_calibrator_checkpoint = bev_calibrator_checkpoint
        self.bev_sample_steps = bev_sample_steps
        self.bev_use_ema = bev_use_ema
        self.controller_type = controller_type
        self.replan_rate = replan_rate
        self.sim_dt = sim_dt
        self.ego_replay_frames = ego_replay_frames
        self.eval_frames = eval_frames
        self.scorer_type = scorer_type
        self.score_start_frame = score_start_frame
        self.eval_mode = eval_mode
        self.enable_vis = enable_vis
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_alpha = temporal_alpha
        self.temporal_lambda = temporal_lambda
        self.temporal_max_history = temporal_max_history
        self.temporal_sigma = temporal_sigma
        self.consensus_temperature = consensus_temperature

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_root / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Result tracking
        self.results = {
            'start_time': datetime.now().isoformat(),
            'config': {
                'model_type': model_type,
                'config_path': config_path,
                'checkpoint_path': checkpoint_path,
                'scenario_root': str(scenario_root),
                'output_root': str(output_root),
                'traffic_mode': traffic_mode,
                'max_workers': max_workers,
                'save_perframe': save_perframe,
                'replan_rate': replan_rate,
                'sim_dt': sim_dt,
                'ego_replay_frames': ego_replay_frames,
                'eval_frames': eval_frames,
                'scorer_type': scorer_type,
                'score_start_frame': score_start_frame,
                'eval_mode': eval_mode,
                'enable_vis': enable_vis,
                'enable_temporal_consistency': enable_temporal_consistency,
                'temporal_alpha': temporal_alpha,
                'temporal_lambda': temporal_lambda,
                'temporal_max_history': temporal_max_history,
                'temporal_sigma': temporal_sigma,
                'consensus_temperature': consensus_temperature,
            },
            'scenarios': {}
        }

        # Load previous results if resuming
        self.results_file = self.output_root / "batch_results.json"
        if self.resume and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                prev_results = json.load(f)
                self.results['scenarios'] = prev_results.get('scenarios', {})
            print(f"Resuming from previous run with {len(self.results['scenarios'])} completed scenarios")

    def get_scenarios(self) -> List[Path]:
        """Get list of scenario directories."""
        scenarios = [
            d for d in sorted(self.scenario_root.iterdir())
            if d.is_dir() and not d.name.startswith('.')
        ]

        # Filter out completed scenarios if resuming
        if self.resume:
            scenarios = [
                s for s in scenarios
                if s.name not in self.results['scenarios'] or
                   self.results['scenarios'][s.name].get('status') != 'success'
            ]
            if scenarios:
                print(f"Found {len(scenarios)} scenarios to evaluate (after filtering completed)")

        return scenarios

    def evaluate_scenario(self, scenario_path: Path) -> Dict[str, Any]:
        """
        Evaluate a single scenario.

        Args:
            scenario_path: Path to scenario directory

        Returns:
            Dictionary with evaluation results
        """
        scenario_name = scenario_path.name
        log_file = self.log_dir / f"{scenario_name}.log"

        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            "unified_evaluator.py",
            "--model-type", self.model_type,
            "--checkpoint", self.checkpoint_path,
            "--scenario-path", str(scenario_path),
            "--traffic-mode", self.traffic_mode,
            "--output-dir", str(self.output_root),
        ]

        # Add config path if available (required for UniAD/VAD)
        if self.config_path:
            cmd.extend(["--config", self.config_path])

        # Add plan anchor path if available (for DiffusionDrive/V2)
        if self.plan_anchor_path:
            cmd.extend(["--plan-anchor-path", self.plan_anchor_path])

        # Add save-perframe flag
        if not self.save_perframe:
            cmd.append("--no-save-perframe")

        # Add BEV calibrator flags
        if self.enable_bev_calibrator:
            cmd.append("--enable-bev-calibrator")
            if self.bev_calibrator_checkpoint:
                cmd.extend(["--bev-calibrator-checkpoint", self.bev_calibrator_checkpoint])
            cmd.extend(["--bev-sample-steps", str(self.bev_sample_steps)])
            if self.bev_use_ema:
                cmd.append("--bev-use-ema")
            else:
                cmd.append("--bev-no-ema")

        # Add controller type
        cmd.extend(["--controller", self.controller_type])

        # Add replan rate
        cmd.extend(["--replan-rate", str(self.replan_rate)])

        # Add simulation timestep
        cmd.extend(["--sim-dt", str(self.sim_dt)])

        # Add ego replay frames
        cmd.extend(["--ego-replay-frames", str(self.ego_replay_frames)])

        # Add eval frames if specified
        if self.eval_frames is not None:
            cmd.extend(["--eval-frames", str(self.eval_frames)])

        # Add scorer type
        cmd.extend(["--scorer-type", self.scorer_type])

        # Add score start frame if specified
        if self.score_start_frame is not None:
            cmd.extend(["--score-start-frame", str(self.score_start_frame)])

        # Add eval mode
        cmd.extend(["--eval-mode", self.eval_mode])

        # Add visualization flag
        if self.enable_vis:
            cmd.append("--enable-vis")

        # Add temporal consistency flags
        if self.enable_temporal_consistency:
            cmd.append("--enable-temporal-consistency")
            cmd.extend(["--temporal-alpha", str(self.temporal_alpha)])
            cmd.extend(["--temporal-lambda", str(self.temporal_lambda)])
            cmd.extend(["--temporal-max-history", str(self.temporal_max_history)])
            cmd.extend(["--temporal-sigma", str(self.temporal_sigma)])
            cmd.extend(["--consensus-temperature", str(self.consensus_temperature)])

        # Run evaluation
        start_time = time.time()
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).parent,  # Run from evaluation directory
                    timeout=3600  # 1 hour timeout per scenario
                )

            duration = time.time() - start_time

            if result.returncode == 0:
                # Check if output exists
                output_dir = self.output_root / scenario_name
                results_file = output_dir / "evaluation_results.json"

                if results_file.exists():
                    with open(results_file, 'r') as f:
                        eval_results = json.load(f)

                    return {
                        'status': 'success',
                        'duration': duration,
                        'exit_code': 0,
                        'log_file': str(log_file),
                        'results': eval_results,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'error',
                        'duration': duration,
                        'exit_code': 0,
                        'error': 'Results file not found',
                        'log_file': str(log_file),
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'status': 'failed',
                    'duration': duration,
                    'exit_code': result.returncode,
                    'log_file': str(log_file),
                    'timestamp': datetime.now().isoformat()
                }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'status': 'timeout',
                'duration': duration,
                'error': 'Evaluation timeout (1 hour)',
                'log_file': str(log_file),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'log_file': str(log_file),
                'timestamp': datetime.now().isoformat()
            }

    def save_results(self):
        """Save results to JSON file."""
        self.results['end_time'] = datetime.now().isoformat()

        # Calculate summary statistics
        total = len(self.results['scenarios'])
        success = sum(1 for r in self.results['scenarios'].values() if r['status'] == 'success')
        failed = sum(1 for r in self.results['scenarios'].values() if r['status'] == 'failed')
        error = sum(1 for r in self.results['scenarios'].values() if r['status'] == 'error')
        timeout = sum(1 for r in self.results['scenarios'].values() if r['status'] == 'timeout')

        self.results['summary'] = {
            'total': total,
            'success': success,
            'failed': failed,
            'error': error,
            'timeout': timeout,
            'success_rate': success / total if total > 0 else 0.0
        }

        # Calculate total duration
        if self.results['scenarios']:
            total_duration = sum(r.get('duration', 0) for r in self.results['scenarios'].values())
            self.results['summary']['total_duration_seconds'] = total_duration
            self.results['summary']['average_duration_seconds'] = total_duration / total

        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def run(self):
        """Run batch evaluation."""
        scenarios = self.get_scenarios()

        if not scenarios:
            print("No scenarios to evaluate!")
            return

        print(f"\n{'='*60}")
        print(f"Starting Batch Evaluation")
        print(f"{'='*60}")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Model: {self.model_type}")
        print(f"Output: {self.output_root}")
        print(f"{'='*60}\n")

        # Sequential evaluation with progress bar
        for scenario_path in tqdm(scenarios, desc="Evaluating scenarios"):
            scenario_name = scenario_path.name

            # Evaluate scenario
            result = self.evaluate_scenario(scenario_path)
            self.results['scenarios'][scenario_name] = result

            # Save results after each scenario (for resume capability)
            self.save_results()

            # Show status
            status_symbol = {
                'success': '✓',
                'failed': '✗',
                'error': '⚠',
                'timeout': '⏱'
            }.get(result['status'], '?')

            tqdm.write(f"{status_symbol} {scenario_name}: {result['status']} ({result.get('duration', 0):.1f}s)")

        # Final summary
        self.print_summary()

        # Aggregate results from successful scenarios
        self.aggregate_all_results()

    def aggregate_all_results(self):
        """Collect and aggregate results from all successful scenarios."""
        # Collect result files from successful scenarios
        result_files = []
        for scenario_name, result in self.results['scenarios'].items():
            if result['status'] == 'success':
                # The evaluation_results.json is in output_root/scenario_name/
                results_json = self.output_root / scenario_name / "evaluation_results.json"
                if results_json.exists():
                    result_files.append(results_json)

        if not result_files:
            print("\nNo successful scenarios to aggregate.")
        else:
            # Define output path for aggregated results
            aggregated_output = self.output_root / "aggregated_results.json"

            print(f"\n{'='*60}")
            print(f"Aggregating Results from {len(result_files)} Scenarios")
            print(f"{'='*60}")

            # Call the aggregate_results function
            aggregate_results(result_files, str(aggregated_output))

        # Generate comprehensive CSV with legacy and EPDMS scores
        # (always generate, even if some scenarios failed)
        self.export_results_csv()

    def export_results_csv(self):
        """Export comprehensive CSV with legacy driving scores and EPDMS scores."""
        import csv

        csv_path = self.output_root / "evaluation_summary.csv"

        # Define CSV columns
        columns = [
            'scenario_name',
            # Legacy scores
            'driving_score', 'route_completion', 'infraction_penalty',
            # EPDMS scores (closed-loop: no ego_progress, uses route_completion instead)
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

        for scenario_name, result in self.results['scenarios'].items():
            if result['status'] != 'success':
                continue

            scenario_dir = self.output_root / scenario_name
            row = {'scenario_name': scenario_name}

            # Load legacy scores from evaluation_results.json
            results_json = scenario_dir / "evaluation_results.json"
            if results_json.exists():
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

            # Load EPDMS scores from *_closedloop_epdms_summary.csv (new format from base_evaluator)
            epdms_files = list(scenario_dir.glob("*_closedloop_epdms_summary.csv"))
            if epdms_files:
                epdms_file = epdms_files[0]  # Take the most recent one
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
                        break  # Only first row

            rows.append(row)

        # Calculate averages
        avg_row = {'scenario_name': 'AVERAGE'}
        for key, values in legacy_totals.items():
            avg_row[key] = np.mean(values) if values else 0.0
        for key, values in epdms_totals.items():
            avg_row[key] = np.mean(values) if values else 0.0

        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in columns})
            # Add empty row before average
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

    def print_summary(self):
        """Print evaluation summary."""
        summary = self.results.get('summary', {})

        print(f"\n{'='*60}")
        print(f"Batch Evaluation Complete!")
        print(f"{'='*60}")
        print(f"Total scenarios: {summary.get('total', 0)}")
        print(f"  ✓ Success: {summary.get('success', 0)}")
        print(f"  ✗ Failed: {summary.get('failed', 0)}")
        print(f"  ⚠ Error: {summary.get('error', 0)}")
        print(f"  ⏱ Timeout: {summary.get('timeout', 0)}")
        print(f"\nSuccess rate: {summary.get('success_rate', 0):.1%}")

        if 'total_duration_seconds' in summary:
            total_hours = summary['total_duration_seconds'] / 3600
            avg_seconds = summary.get('average_duration_seconds', 0)
            print(f"Total duration: {total_hours:.2f} hours")
            print(f"Average per scenario: {avg_seconds:.1f} seconds")

        print(f"\nResults saved to: {self.results_file}")
        print(f"Logs saved to: {self.log_dir}")
        print(f"{'='*60}\n")


def aggregate_results(result_files: List[Path], output_path: str) -> None:
    """
    Aggregate individual scenario results into a final JSON.

    This function computes global statistics from multiple scenario evaluation results,
    including driving scores, infractions per km, and planning metrics (L2 errors).

    Args:
        result_files: List of paths to individual evaluation_results.json files
        output_path: Path to save the aggregated results JSON
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
                # Extract the record from the checkpoint
                if '_checkpoint' in data and 'records' in data['_checkpoint']:
                    record = data['_checkpoint']['records'][0]
                    record['index'] = idx
                    records.append(record)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    if not records:
        print("No valid records found")
        return

    # Compute global statistics
    total_scenarios = len(records)

    # Aggregate infractions
    infractions_sum = {}
    for record in records:
        for key, value in record['infractions'].items():
            if key not in infractions_sum:
                infractions_sum[key] = 0
            infractions_sum[key] += len(value) if isinstance(value, list) else value

    # Aggregate scores
    score_route_sum = sum(r['scores']['score_route'] for r in records)
    score_penalty_sum = sum(r['scores']['score_penalty'] for r in records)
    score_composed_sum = sum(r['scores']['score_composed'] for r in records)

    score_route_mean = score_route_sum / total_scenarios
    score_penalty_mean = score_penalty_sum / total_scenarios
    score_composed_mean = score_composed_sum / total_scenarios

    # Calculate standard deviations
    score_route_values = [r['scores']['score_route'] for r in records]
    score_penalty_values = [r['scores']['score_penalty'] for r in records]
    score_composed_values = [r['scores']['score_composed'] for r in records]

    score_route_std = np.std(score_route_values) if len(score_route_values) > 1 else 0
    score_penalty_std = np.std(score_penalty_values) if len(score_penalty_values) > 1 else 0
    score_composed_std = np.std(score_composed_values) if len(score_composed_values) > 1 else 0

    # Aggregate planning metrics (L2)
    planning_metrics_keys = ['avg_l2_1s', 'avg_l2_2s', 'avg_l2_3s',
                             'l2_0.5s', 'l2_1.0s', 'l2_1.5s', 'l2_2.0s', 'l2_2.5s', 'l2_3.0s']
    planning_metrics_mean = {}
    planning_metrics_std = {}

    for key in planning_metrics_keys:
        # Collect values from all records that have planning_metrics
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
    total_km_driven = sum(km_driven_list)
    total_km_driven = max(total_km_driven, 0.001)  # Avoid division by zero

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
            'score_route': round(score_route_mean, 6),
            'score_penalty': round(score_penalty_mean, 6),
            'score_composed': round(score_composed_mean, 6)
        },
        'scores_std_dev': {
            'score_route': round(score_route_std, 6),
            'score_penalty': round(score_penalty_std, 6),
            'score_composed': round(score_composed_std, 6)
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

    # Calculate values per km (following CARLA format)
    ROUND_DIGITS = 3
    values = [
        str(global_record['scores_mean']['score_composed']),
        str(global_record['scores_mean']['score_route']),
        str(global_record['scores_mean']['score_penalty']),
        str(round(global_record['infractions']['collisions_pedestrian'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['collisions_vehicle'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['collisions_layout'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['red_light'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['stop_infraction'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['outside_route_lanes'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['route_dev'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['route_timeout'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['vehicle_blocked'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['yield_emergency_vehicle_infractions'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['scenario_timeouts'] / total_km_driven, ROUND_DIGITS)),
        str(round(global_record['infractions']['min_speed_infractions'] / total_km_driven, ROUND_DIGITS)),
    ]

    labels = [
        "Avg. driving score", "Avg. route completion", "Avg. infraction penalty",
        "Collisions with pedestrians", "Collisions with vehicles", "Collisions with layout",
        "Red lights infractions", "Stop sign infractions", "Off-road infractions",
        "Route deviations", "Route timeouts", "Agent blocked",
        "Yield emergency vehicles infractions", "Scenario timeouts", "Min speed infractions",
        "L2 (m) 1s", "L2 (m) 2s", "L2 (m) 3s"
    ]

    # Append L2 metrics to values
    values.extend([
        str(planning_metrics_mean.get('avg_l2_1s', 0.0)),
        str(planning_metrics_mean.get('avg_l2_2s', 0.0)),
        str(planning_metrics_mean.get('avg_l2_3s', 0.0))
    ])

    # Create final results structure
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

    # Save to file (ensure parent directory exists)
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
    print(f"  L2 @ 1s: {planning_metrics_mean.get('avg_l2_1s', 0.0):.3f} ± {planning_metrics_std.get('avg_l2_1s', 0.0):.3f}")
    print(f"  L2 @ 2s: {planning_metrics_mean.get('avg_l2_2s', 0.0):.3f} ± {planning_metrics_std.get('avg_l2_2s', 0.0):.3f}")
    print(f"  L2 @ 3s: {planning_metrics_mean.get('avg_l2_3s', 0.0):.3f} ± {planning_metrics_std.get('avg_l2_3s', 0.0):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for multiple scenarios")

    parser.add_argument('--model-type', type=str, required=True,
                        choices=["uniad", "vad", "tcp", "rap", "lead", "lead_navsim", "drivor",
                                 "transfuser", "ltf", "egomlp", "ego_mlp", "diffusiondrive", "diffusiondrivev2"],
                        help="Model type")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")

    parser.add_argument('--config', type=str, default=None,
                        help="Path to model config file (required for UniAD/VAD)")

    parser.add_argument('--plan-anchor-path', type=str, default=None,
                        help="Path to plan anchor file (for DiffusionDrive/V2 models)")

    parser.add_argument('--scenario-root', type=str, required=True,
                        help="Root directory containing scenarios")

    parser.add_argument('--output-dir', type=str, required=True,
                        help="Root directory for outputs")

    parser.add_argument('--traffic-mode', type=str, default='log_replay',
                        choices=['no_traffic', 'log_replay', 'IDM'],
                        help="Traffic mode")

    parser.add_argument('--max-workers', type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")

    parser.add_argument('--resume', action='store_true',
                        help="Resume from previous run (skip completed scenarios)")

    parser.add_argument('--save-perframe', action='store_true', default=True,
                        help="Save per-frame outputs (planning_traj.npy, etc.). Default: True")

    parser.add_argument('--no-save-perframe', dest='save_perframe', action='store_false',
                        help="Disable saving per-frame outputs to save disk space")

    # BEV Calibration
    parser.add_argument('--enable-bev-calibrator', action='store_true',
                        help="Enable BEV calibrator for domain adaptation (MetaDrive -> Bench2Drive)")

    parser.add_argument('--bev-calibrator-checkpoint', type=str,
                        default="/home/zhihao/workspace/BridgeSim/calibration/checkpoints/BridgeSim-BevFlow/uniad-b2d/lr1e-4/epoch=9-step=24450.ckpt",
                        help="Path to BEV calibrator checkpoint (.ckpt file)")

    parser.add_argument('--bev-sample-steps', type=int, default=50,
                        help="Number of Euler sampling steps for BEV flow matching (default: 50)")

    parser.add_argument('--bev-use-ema', action='store_true', default=True,
                        help="Use EMA weights for BEV calibrator (default: True)")

    parser.add_argument('--bev-no-ema', dest='bev_use_ema', action='store_false',
                        help="Disable EMA weights for BEV calibrator")

    # Controller selection
    parser.add_argument('--controller', type=str, default='pure_pursuit',
                        choices=['pid', 'pure_pursuit'],
                        help="Controller type for vehicle control (default: pure_pursuit)")

    # Replan rate
    parser.add_argument('--replan-rate', type=int, default=1,
                        help="How often to run model inference (1=every frame, 5=every 5 frames, etc.). "
                             "Between replans, cached waypoints are consumed sequentially. (default: 1)")

    # Simulation timestep
    parser.add_argument('--sim-dt', type=float, default=0.1,
                        help="Simulation timestep in seconds (default: 0.1s for 10Hz). "
                             "Model trajectory is interpolated to this interval for smooth control.")

    # Ego replay frames
    parser.add_argument('--ego-replay-frames', type=int, default=0,
                        help="Number of initial frames to replay ego log actions while still running model inference (default: 0)")

    # Eval frames
    parser.add_argument('--eval-frames', type=int, default=None,
                        help="Number of frames to evaluate after ego replay ends (default: None = run full scenario). "
                             "Total frames = min(ego_replay_frames + eval_frames, scenario_length)")

    # Scorer type
    parser.add_argument('--scorer-type', type=str, default='legacy',
                        choices=['legacy', 'navsim'],
                        help="Scorer type for closed_loop mode (default: legacy). "
                             "'navsim' uses NavSim-style EPDMS scoring with per-frame metrics.")

    # Score start frame
    parser.add_argument('--score-start-frame', type=int, default=None,
                        help="Frame to start calculating scores (default: None = uses ego_replay_frames). "
                             "Scoring metrics and route completion are computed only from this frame onwards.")

    # Eval mode
    parser.add_argument('--eval-mode', type=str, default='closed_loop',
                        choices=['closed_loop', 'open_loop'],
                        help="Evaluation mode: closed_loop (agent controlled by model) or open_loop (agent follows ground truth)")

    # Visualization
    parser.add_argument('--enable-vis', action='store_true',
                        help="Enable visualization outputs (images, topdown views)")

    # Temporal consistency parameters (for DiffusionDriveV2)
    parser.add_argument('--enable-temporal-consistency', action='store_true',
                        help="Enable temporal consistency scoring for DiffusionDriveV2")

    parser.add_argument('--temporal-alpha', type=float, default=1.5,
                        help="Temporal consistency decay base (default: 1.5)")

    parser.add_argument('--temporal-lambda', type=float, default=0.3,
                        help="Weight for temporal consistency in combined score (default: 0.3)")

    parser.add_argument('--temporal-max-history', type=int, default=8,
                        help="Maximum number of past trajectory predictions to store (default: 8)")

    parser.add_argument('--temporal-sigma', type=float, default=5.0,
                        help="Position normalization factor in meters (default: 5.0)")

    parser.add_argument('--consensus-temperature', type=float, default=1.0,
                        help="Softmax temperature for consensus trajectory weighting (default: 1.0)")

    args = parser.parse_args()

    # Create batch evaluator
    evaluator = BatchEvaluator(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        scenario_root=args.scenario_root,
        output_root=args.output_dir,
        config_path=args.config,
        plan_anchor_path=args.plan_anchor_path,
        traffic_mode=args.traffic_mode,
        max_workers=args.max_workers,
        resume=args.resume,
        save_perframe=args.save_perframe,
        enable_bev_calibrator=args.enable_bev_calibrator,
        bev_calibrator_checkpoint=args.bev_calibrator_checkpoint,
        bev_sample_steps=args.bev_sample_steps,
        bev_use_ema=args.bev_use_ema,
        controller_type=args.controller,
        replan_rate=args.replan_rate,
        sim_dt=args.sim_dt,
        ego_replay_frames=args.ego_replay_frames,
        eval_frames=args.eval_frames,
        scorer_type=args.scorer_type,
        score_start_frame=args.score_start_frame,
        eval_mode=args.eval_mode,
        enable_vis=args.enable_vis,
        enable_temporal_consistency=args.enable_temporal_consistency,
        temporal_alpha=args.temporal_alpha,
        temporal_lambda=args.temporal_lambda,
        temporal_max_history=args.temporal_max_history,
        temporal_sigma=args.temporal_sigma,
        consensus_temperature=args.consensus_temperature
    )

    # Run evaluation
    try:
        evaluator.run()
    except KeyboardInterrupt:
        print("\n\nBatch evaluation interrupted by user!")
        print("Progress has been saved. Use --resume to continue.")
        evaluator.save_results()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBatch evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        evaluator.save_results()
        sys.exit(1)


if __name__ == "__main__":
    main()

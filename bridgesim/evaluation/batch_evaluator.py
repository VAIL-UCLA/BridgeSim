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
                 # scorer_type: str = "legacy",
                 score_start_frame: int = None,
                 eval_mode: str = "closed_loop",
                 enable_vis: bool = False,
                 enable_temporal_consistency: bool = False,
                 temporal_alpha: float = 1.5,
                 temporal_lambda: float = 0.3,
                 temporal_max_history: int = 8,
                 temporal_sigma: float = 5.0,
                 consensus_temperature: float = 1.0,
                 alp_python: str = "",
                 alp_script: str = "",
                 alp_coord_mode: str = "x_forward_y_left",
                 alp_top_p: float = 0.98,
                 alp_temperature: float = 0.6,
                 alp_num_traj_samples: int = 20,
                 alp_max_generation_length: int = 256,
                 trajectory_scorer: str = None,
                 num_groups: int = None,
                 num_proposals: int = None,
                 v2_scorer_checkpoint: str = None):
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
            score_start_frame: Frame to start calculating scores
            eval_mode: Evaluation mode ('closed_loop' or 'open_loop')
            enable_vis: Enable visualization outputs
            alp_python: Path to Alpamayo venv python (alpamayo_r1 only)
            alp_script: Path to BridgeSim Alpamayo glue script (alpamayo_r1 only)
            alp_coord_mode: Coordinate mapping for Alpamayo trajectory conversion
            alp_top_p: Top-p sampling parameter for Alpamayo
            alp_temperature: Temperature for Alpamayo generation
            alp_num_traj_samples: Number of trajectory samples for Alpamayo
            alp_max_generation_length: Max generation length for Alpamayo
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
        # self.scorer_type = scorer_type
        self.score_start_frame = score_start_frame
        self.eval_mode = eval_mode
        self.enable_vis = enable_vis
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_alpha = temporal_alpha
        self.temporal_lambda = temporal_lambda
        self.temporal_max_history = temporal_max_history
        self.temporal_sigma = temporal_sigma
        self.consensus_temperature = consensus_temperature
        self.alp_python = alp_python
        self.alp_script = alp_script
        self.alp_coord_mode = alp_coord_mode
        self.alp_top_p = alp_top_p
        self.alp_temperature = alp_temperature
        self.alp_num_traj_samples = alp_num_traj_samples
        self.alp_max_generation_length = alp_max_generation_length
        self.trajectory_scorer = trajectory_scorer
        self.num_groups = num_groups
        self.num_proposals = num_proposals
        self.v2_scorer_checkpoint = v2_scorer_checkpoint

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Result tracking (in-memory only)
        self.results = {'scenarios': {}}

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
                if not (self.output_root / self._get_output_subdir() / s.name / "driving_score_summary.csv").exists()
            ]
            if scenarios:
                pass  # resume filter count print suppressed

        return scenarios

    def _get_output_subdir(self) -> str:
        """
        Compute the output subdirectory name that unified_evaluator.py creates inside
        --output-dir.  Must stay in sync with the logic in unified_evaluator.py.
        """
        eval_frames_str = str(self.eval_frames) if self.eval_frames is not None else "None"
        subdir = f"{self.model_type}_rr{self.replan_rate}_erf{self.ego_replay_frames}_ef{eval_frames_str}"
        if self.trajectory_scorer:
            ng = self.num_groups if self.num_groups is not None else (
                1 if self.model_type == "diffusiondrive" else 10)
            subdir += f"_scorer_{self.trajectory_scorer}_ng{ng}"
            if self.num_proposals is not None:
                subdir += f"_np{self.num_proposals}"
        if self.enable_temporal_consistency:
            subdir += f"_ta{self.temporal_alpha}_th{self.temporal_max_history}"
        return subdir

    def evaluate_scenario(self, scenario_path: Path) -> Dict[str, Any]:
        """
        Evaluate a single scenario.

        Args:
            scenario_path: Path to scenario directory

        Returns:
            Dictionary with evaluation results
        """
        scenario_name = scenario_path.name

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

        # Add trajectory scorer flags
        if self.trajectory_scorer:
            cmd.extend(["--trajectory-scorer", self.trajectory_scorer])
        if self.num_groups is not None:
            cmd.extend(["--num-groups", str(self.num_groups)])
        if self.num_proposals is not None:
            cmd.extend(["--num-proposals", str(self.num_proposals)])
        if self.v2_scorer_checkpoint:
            cmd.extend(["--v2-scorer-checkpoint", self.v2_scorer_checkpoint])

        # Add Alpamayo-specific flags
        if self.model_type == "alpamayo_r1":
            if self.alp_python:
                cmd.extend(["--alp-python", self.alp_python])
            if self.alp_script:
                cmd.extend(["--alp-script", self.alp_script])
            cmd.extend(["--alp-coord-mode", self.alp_coord_mode])
            cmd.extend(["--alp-top-p", str(self.alp_top_p)])
            cmd.extend(["--alp-temperature", str(self.alp_temperature)])
            cmd.extend(["--alp-num-traj-samples", str(self.alp_num_traj_samples)])
            cmd.extend(["--alp-max-generation-length", str(self.alp_max_generation_length)])

        # Run evaluation
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,  # Run from evaluation directory
                timeout=3600  # 1 hour timeout per scenario
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                # Check if output exists (unified_evaluator nests under output_subdir/scenario_name/)
                output_dir = self.output_root / self._get_output_subdir() / scenario_name
                summary_csv = output_dir / "driving_score_summary.csv"

                if summary_csv.exists():
                    # Read the AVERAGE row for in-memory tracking
                    import csv as csv_mod
                    avg_scores = {}
                    with open(summary_csv, 'r') as f:
                        reader = csv_mod.DictReader(f)
                        for row in reader:
                            if row.get('frame_id') == 'AVERAGE':
                                avg_scores = {k: v for k, v in row.items() if k != 'frame_id'}
                                break

                    return {
                        'status': 'success',
                        'duration': duration,
                        'exit_code': 0,
                        'scores': avg_scores,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'status': 'error',
                        'duration': duration,
                        'exit_code': 0,
                        'error': 'driving_score_summary.csv not found',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                return {
                    'status': 'failed',
                    'duration': duration,
                    'exit_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'status': 'timeout',
                'duration': duration,
                'error': 'Evaluation timeout (1 hour)',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'status': 'error',
                'duration': duration,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def run(self):
        """Run batch evaluation."""
        scenarios = self.get_scenarios()

        if not scenarios:
            return

        # Sequential evaluation
        for scenario_path in tqdm(scenarios, desc="Evaluating scenarios", disable=True):
            scenario_name = scenario_path.name

            # Evaluate scenario
            result = self.evaluate_scenario(scenario_path)
            self.results['scenarios'][scenario_name] = result

        # Final summary
        # self.print_summary()

        # Aggregate results from successful scenarios
        self.aggregate_all_results()

    def aggregate_all_results(self):
        """Export batch summary CSV from all successful scenarios."""
        self.export_batch_summary_csv()

    def export_batch_summary_csv(self):
        """Export batch_driving_score_summary.csv with one row per scenario + AVERAGE."""
        import csv as csv_mod

        csv_path = self.output_root / "batch_driving_score_summary.csv"
        columns = ['scenario', 'DS', 'EPDMS_no_ep', 'RC', 'NC', 'DAC', 'DDC', 'TL', 'TTC', 'LK', 'HC', 'EC']

        rows = []
        totals = {col: [] for col in columns if col != 'scenario'}

        for scenario_name, result in self.results['scenarios'].items():
            if result['status'] != 'success':
                continue

            # Read AVERAGE row from driving_score_summary.csv
            summary_csv = self.output_root / self._get_output_subdir() / scenario_name / "driving_score_summary.csv"
            if not summary_csv.exists():
                continue

            with open(summary_csv, 'r') as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    if row.get('frame_id') == 'AVERAGE':
                        out_row = {'scenario': scenario_name}
                        for col in columns[1:]:
                            val = row.get(col, '')
                            out_row[col] = val
                            try:
                                totals[col].append(float(val))
                            except (ValueError, TypeError):
                                pass
                        rows.append(out_row)
                        break

        # AVERAGE row across scenarios
        avg_row = {'scenario': 'AVERAGE'}
        for col in columns[1:]:
            vals = totals[col]
            avg_row[col] = f"{np.mean(vals):.6f}" if vals else ''

        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in columns})
            writer.writerow(avg_row)

        print(f"\n{'='*60}")
        print("Batch Evaluation Complete! Batch Summary (Averages)")
        print(f"{'='*60}")
        print(f"  DS:           {avg_row.get('DS', 'N/A')}")
        print(f"  EPDMS_no_ep:  {avg_row.get('EPDMS_no_ep', 'N/A')}")
        print(f"  RC:           {avg_row.get('RC', 'N/A')}")
        print(f"  NC:           {avg_row.get('NC', 'N/A')}")
        print(f"  DAC:          {avg_row.get('DAC', 'N/A')}")
        print(f"  DDC:          {avg_row.get('DDC', 'N/A')}")
        print(f"  TL:           {avg_row.get('TL', 'N/A')}")
        print(f"  TTC:          {avg_row.get('TTC', 'N/A')}")
        print(f"  LK:           {avg_row.get('LK', 'N/A')}")
        print(f"  HC:           {avg_row.get('HC', 'N/A')}")
        print(f"  EC:           {avg_row.get('EC', 'N/A')}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for multiple scenarios")

    parser.add_argument('--model-type', type=str, required=True,
                        choices=["uniad", "vad", "tcp", "rap", "lead", "lead_navsim", "drivor",
                                 "transfuser", "ltf", "egomlp", "ego_mlp", "diffusiondrive", "diffusiondrivev2", "openpilot",
                                 "alpamayo_r1", "pdm_lite"],
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

    # AlpamayoR1 parameters
    parser.add_argument('--alp-python', type=str, default="",
                        help="Path to Alpamayo venv python (alpamayo_r1 only). "
                             "Falls back to ALPAMAYO_PYTHON env var if empty.")
    parser.add_argument('--alp-script', type=str, default="",
                        help="Path to BridgeSim Alpamayo glue script (alpamayo_r1 only). "
                             "Falls back to ALPAMAYO_SCRIPT env var if empty.")
    parser.add_argument('--alp-coord-mode', type=str, default="x_forward_y_left",
                        help="Coordinate mapping for Alpamayo trajectory conversion (default: x_forward_y_left)")
    parser.add_argument('--alp-top-p', type=float, default=0.98,
                        help="Top-p sampling for Alpamayo (default: 0.98)")
    parser.add_argument('--alp-temperature', type=float, default=0.6,
                        help="Generation temperature for Alpamayo (default: 0.6)")
    parser.add_argument('--alp-num-traj-samples', type=int, default=20,
                        help="Number of trajectory samples for Alpamayo (default: 20)")
    parser.add_argument('--alp-max-generation-length', type=int, default=256,
                        help="Max generation length for Alpamayo (default: 256)")

    # Trajectory scorer (for DiffusionDrive/V2 inference scaling)
    parser.add_argument('--trajectory-scorer', type=str, default=None,
                        choices=["cls", "learned", "gt", "tta"],
                        help="Trajectory scorer for inference scaling (for DiffusionDrive/V2). "
                             "'cls' uses poses_cls (v1 only). "
                             "'learned' uses v2 learned coarse scorer (v1 needs --v2-scorer-checkpoint). "
                             "'gt' uses GT scorer. 'tta' uses TTA scorer.")
    parser.add_argument('--num-groups', type=int, default=None,
                        help="Number of trajectory candidate groups. Total candidates = num_groups * 20. "
                             "Only used with --trajectory-scorer. (default: 1 for v1, 10 for v2)")
    parser.add_argument('--num-proposals', type=int, default=None,
                        help="Truncate candidates to the first num_proposals before scoring. "
                             "Used to study the effect of different numbers of proposals.")
    parser.add_argument('--v2-scorer-checkpoint', type=str, default=None,
                        help="Path to DiffusionDrive v2 checkpoint for loading coarse scorer weights. "
                             "Required when using --trajectory-scorer learned with DiffusionDrive v1.")

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
        score_start_frame=args.score_start_frame,
        eval_mode=args.eval_mode,
        enable_vis=args.enable_vis,
        enable_temporal_consistency=args.enable_temporal_consistency,
        temporal_alpha=args.temporal_alpha,
        temporal_lambda=args.temporal_lambda,
        temporal_max_history=args.temporal_max_history,
        temporal_sigma=args.temporal_sigma,
        consensus_temperature=args.consensus_temperature,
        alp_python=args.alp_python,
        alp_script=args.alp_script,
        alp_coord_mode=args.alp_coord_mode,
        alp_top_p=args.alp_top_p,
        alp_temperature=args.alp_temperature,
        alp_num_traj_samples=args.alp_num_traj_samples,
        alp_max_generation_length=args.alp_max_generation_length,
        trajectory_scorer=args.trajectory_scorer,
        num_groups=args.num_groups,
        num_proposals=args.num_proposals,
        v2_scorer_checkpoint=args.v2_scorer_checkpoint,
    )

    # Run evaluation
    try:
        evaluator.run()
    except KeyboardInterrupt:
        print("\n\nBatch evaluation interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBatch evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified Evaluator - Single entry point for all models.
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Initialize CUDA device BEFORE any imports that might use CUDA
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    try:
        from cuda import cudart
        # cudaSetDevice returns a tuple (error_code, )
        err, = cudart.cudaSetDevice(0)
        if err == cudart.cudaError_t.cudaSuccess:
            pass  # early init print suppressed
        else:
            pass  # early init warning suppressed
    except Exception as e:
        # Pass silently if cuda-python is not installed or other errors occur
        # The main code will handle device placement via PyTorch
        pass

# Add evaluation to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridgesim.evaluation.core.base_evaluator import BaseEvaluator, _silence


def create_trajectory_scorer(args):
    """Create trajectory scorer based on args. Returns None if no scorer requested."""
    scorer_name = args.trajectory_scorer
    if scorer_name is None:
        return None

    model_type = args.model_type.lower()

    if scorer_name == "cls":
        from bridgesim.evaluation.scorers import ClsScorer
        return ClsScorer()

    elif scorer_name == "learned":
        from bridgesim.evaluation.scorers import LearnedScorer
        v2_ckpt = args.v2_scorer_checkpoint
        if model_type == "diffusiondrive" and v2_ckpt is None:
            raise ValueError(
                "--v2-scorer-checkpoint is required when using "
                "--trajectory-scorer learned with DiffusionDrive v1."
            )
        return LearnedScorer(
            v2_scorer_checkpoint_path=v2_ckpt,
            device="cuda",
        )

    elif scorer_name == "gt":
        from bridgesim.evaluation.scorers.GT_scorer import GTScorer
        return GTScorer()

    elif scorer_name == "tta":
        from bridgesim.evaluation.scorers.TTA_scorer import TTAScorer
        return TTAScorer()

    else:
        raise ValueError(f"Unknown trajectory scorer: {scorer_name}")


def create_model_adapter(args):
    """Create appropriate model adapter based on args."""
    model_type = args.model_type.lower()

    bev_calibrator = None
    if args.enable_bev_calibrator:
        if model_type in ["uniad", "vad"]:
            # UniAD/VAD use the original BEV calibrator (256 channels, 200x200)
            from bridgesim.evaluation.features.bev_calibrator import create_bev_calibrator
            bev_calibrator = create_bev_calibrator(
                checkpoint_path=args.bev_calibrator_checkpoint,
                sample_steps=args.bev_sample_steps,
                use_ema=args.bev_use_ema,
                device="cuda"
            )
        elif model_type in ["diffusiondrive", "diffusiondrivev2", "transfuser"]:
            # DiffusionDrive v1/v2 use TransFuser BEV calibrator (512 channels, 8x8)
            from bridgesim.evaluation.features.transfuser_bev_calibrator import create_transfuser_bev_calibrator
            bev_calibrator = create_transfuser_bev_calibrator(
                checkpoint_path=args.bev_calibrator_checkpoint,
                sample_steps=args.bev_sample_steps,
                use_ema=args.bev_use_ema,
                device="cuda"
            )
        else:
            pass  # BEV calibrator not supported warning suppressed

    if model_type == "uniad":
        from bridgesim.evaluation.models.uniad_vad_adapter import UniADVADAdapter
        if not args.config:
            raise ValueError("--config is required for UniAD model")
        return UniADVADAdapter(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_type="uniad",
            bev_calibrator=bev_calibrator
        )

    elif model_type == "vad":
        from bridgesim.evaluation.models.uniad_vad_adapter import UniADVADAdapter
        if not args.config:
            raise ValueError("--config is required for VAD model")
        return UniADVADAdapter(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_type="vad",
            bev_calibrator=bev_calibrator
        )

    elif model_type == "tcp":
        from bridgesim.evaluation.models.tcp_adapter import TCPAdapter
        return TCPAdapter(
            checkpoint_path=args.checkpoint,
            planner_type=args.planner_type
        )

    elif model_type == "rap":
        from bridgesim.evaluation.models.rap_adapter import RAPAdapter
        scorer = create_trajectory_scorer(args)
        return RAPAdapter(
            checkpoint_path=args.checkpoint,
            image_source=args.image_source,
            scorer=scorer,
            num_proposals=args.num_proposals,
        )

    elif model_type == "lead":
        from bridgesim.evaluation.models.lead_adapter import LEADAdapter
        return LEADAdapter(
            checkpoint_path=args.checkpoint
        )

    elif model_type == "drivor":
        from bridgesim.evaluation.models.drivor_adapter import DrivoRAdapter
        scorer = create_trajectory_scorer(args)
        return DrivoRAdapter(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            num_cameras=args.num_cameras,
            image_size=tuple(args.image_size) if args.image_size else (512, 288),
            num_poses=args.num_poses,
            use_lidar=args.use_lidar,
            scorer=scorer,
            num_proposals=args.num_proposals,
        )

    elif model_type == "transfuser":
        from bridgesim.evaluation.models.transfuser_adapter import TransfuserAdapter
        return TransfuserAdapter(
            checkpoint_path=args.checkpoint,
            bev_calibrator=bev_calibrator,
        )

    elif model_type == "ltf":
        from bridgesim.evaluation.models.ltf_adapter import LTFAdapter
        return LTFAdapter(checkpoint_path=args.checkpoint)

    elif model_type in ["egomlp", "ego_mlp"]:
        from bridgesim.evaluation.models.ego_mlp_adapter import EgoStatusMLPAdapter
        return EgoStatusMLPAdapter(checkpoint_path=args.checkpoint)

    elif model_type == "diffusiondrive":
        from bridgesim.evaluation.models.diffusiondrive_adapter import DiffusionDriveAdapter
        scorer = create_trajectory_scorer(args)
        num_groups = args.num_groups if args.num_groups is not None else (1 if scorer is None else 1)
        return DiffusionDriveAdapter(
            checkpoint_path=args.checkpoint,
            plan_anchor_path=args.plan_anchor_path,
            scorer=scorer,
            num_groups=num_groups,
            num_proposals=args.num_proposals,
            bev_calibrator=bev_calibrator,
        )

    elif model_type == "diffusiondrivev2":
        from bridgesim.evaluation.models.diffusiondrivev2_adapter import DiffusionDriveV2Adapter
        scorer = create_trajectory_scorer(args)
        num_groups = args.num_groups if args.num_groups is not None else (10 if scorer is not None else 10)
        return DiffusionDriveV2Adapter(
            checkpoint_path=args.checkpoint,
            plan_anchor_path=args.plan_anchor_path,
            scorer=scorer,
            num_groups=num_groups,
            num_proposals=args.num_proposals,
            bev_calibrator=bev_calibrator,
            enable_temporal_consistency=args.enable_temporal_consistency,
            temporal_alpha=args.temporal_alpha,
            temporal_lambda=args.temporal_lambda,
            temporal_max_history=args.temporal_max_history,
            temporal_sigma=args.temporal_sigma,
            consensus_temperature=args.consensus_temperature,
        )

    elif model_type == "lead_navsim":
        from bridgesim.evaluation.models.lead_navsim_adapter import LEADNavsimAdapter
        return LEADNavsimAdapter(checkpoint_path=args.checkpoint)

    elif model_type == "openpilot":
        from bridgesim.evaluation.models.openpilot_adapter import OpenPilotAdapter
        return OpenPilotAdapter(checkpoint_path=args.checkpoint)

    elif model_type == "alpamayo_r1":
        from bridgesim.evaluation.models.alpamayo_r1_adapter import AlpamayoR1Adapter
        scorer = create_trajectory_scorer(args)
        num_groups = args.num_groups if args.num_groups is not None else (1 if scorer is None else 1)
        return AlpamayoR1Adapter(
            checkpoint_path=args.checkpoint,
            alp_python=args.alp_python,
            alp_script=args.alp_script,
            coord_mode=args.alp_coord_mode,
            top_p=args.alp_top_p,
            temperature=args.alp_temperature,
            num_traj_samples=args.alp_num_traj_samples,
            max_generation_length=args.alp_max_generation_length,
            scorer=scorer,
            num_groups=num_groups,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluator for autonomous driving models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["uniad", "vad", "tcp", "rap", "lead", "lead_navsim", "drivor", "transfuser", "ltf", "egomlp", "ego_mlp", "diffusiondrive", "diffusiondrivev2", "openpilot", "alpamayo_r1"],
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config (required for UniAD/VAD)"
    )

    # Model-specific parameters
    parser.add_argument(
        "--planner-type",
        type=str,
        default="only_traj",
        choices=["only_ctrl", "only_traj", "merge_ctrl_traj"],
        help="TCP planner type (only used for TCP model)"
    )
    parser.add_argument(
        "--image-source",
        type=str,
        default="metadrive",
        choices=["rasterized", "metadrive", "rasterized_3d"],
        help="RAP image source (only used for RAP model)"
    )
    parser.add_argument(
        "--plan-anchor-path",
        type=str,
        default=None,
        help="Path to plan anchor file (for DiffusionDrive/V2 models)"
    )

    # Inference scaling parameters (for DiffusionDrive/V2)
    parser.add_argument(
        "--trajectory-scorer",
        type=str,
        default=None,
        choices=["cls", "learned", "gt", "tta"],
        help="Trajectory scorer for inference scaling (for DiffusionDrive/V2). "
             "'confidence' uses poses_cls (v1 only). "
             "'coarse_topk' uses v2 learned coarse scorer (v1 needs --v2-scorer-checkpoint). "
             "'epdms' uses evaluator EPDMS scorer (not yet implemented)."
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="Number of groups for trajectory candidate generation. "
             "Total candidates = num_groups * 20. Only used with --trajectory-scorer. "
             "(default: 1 for v1, 10 for v2)"
    )
    parser.add_argument(
        "--num-proposals",
        type=int,
        default=None,
        help="Truncate candidates to the first num_proposals before scoring. "
             "Used to study the effect of different numbers of proposals."
    )
    parser.add_argument(
        "--v2-scorer-checkpoint",
        type=str,
        default=None,
        help="Path to DiffusionDrive v2 checkpoint for loading coarse scorer weights. "
             "Required when using --trajectory-scorer coarse_topk with DiffusionDrive v1."
    )

    # AlpamayoR1 external-env parameters
    parser.add_argument(
        "--alp-python",
        type=str,
        default=os.environ.get("ALPAMAYO_PYTHON", ""),
        help="Path to Alpamayo venv python. If empty, adapter will try sibling layout or env var ALPAMAYO_PYTHON."
    )
    parser.add_argument(
        "--alp-script",
        type=str,
        default=os.environ.get("ALPAMAYO_SCRIPT", ""),
        help="Path to BridgeSim Alpamayo glue script (tools/bridgesim_infer_once.py). Optional."
    )
    parser.add_argument(
        "--alp-coord-mode",
        type=str,
        default="x_forward_y_left",
        choices=["x_forward_y_left", "x_forward_y_right", "x_right_y_forward", "x_left_y_forward"],
        help="Coordinate mapping for Alpamayo trajectory conversion."
    )
    parser.add_argument("--alp-top-p", type=float, default=0.98)
    parser.add_argument("--alp-temperature", type=float, default=0.6)
    parser.add_argument("--alp-num-traj-samples", type=int, default=20)
    parser.add_argument("--alp-max-generation-length", type=int, default=256)

    # Temporal consistency parameters (for DiffusionDriveV2)
    parser.add_argument(
        "--enable-temporal-consistency",
        action="store_true",
        help="Enable temporal consistency scoring for DiffusionDriveV2. "
             "This combines the learned PDM scorer with a temporal consistency term "
             "that favors trajectories consistent with previous pseudo-expert predictions."
    )
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=1.5,
        help="Temporal consistency decay base. Higher values give more weight to older "
             "trajectory predictions. Range: 1.0-3.0 (default: 1.5)"
    )
    parser.add_argument(
        "--temporal-lambda",
        type=float,
        default=0.3,
        help="Weight for temporal consistency in combined score. "
             "0.0 = pure PDM scorer, 1.0 = pure temporal consistency. (default: 0.3)"
    )
    parser.add_argument(
        "--temporal-max-history",
        type=int,
        default=8,
        help="Maximum number of past trajectory predictions to store. (default: 8)"
    )
    parser.add_argument(
        "--temporal-sigma",
        type=float,
        default=5.0,
        help="Position normalization factor in meters for temporal consistency. (default: 5.0)"
    )
    parser.add_argument(
        "--consensus-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for consensus trajectory weighting. "
             "Lower values make consensus dominated by highest-PDM trajectory. "
             "Higher values make consensus closer to uniform mean. (default: 1.0)"
    )

    # DrivoR-specific parameters
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of cameras for DrivoR (4 or 8)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 288],
        metavar=("WIDTH", "HEIGHT"),
        help="Image size for DrivoR (width height)"
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=8,
        help="Number of trajectory poses for DrivoR"
    )
    parser.add_argument(
        "--use-lidar",
        action="store_true",
        help="Use LiDAR input for DrivoR"
    )

    # Scenario and environment
    parser.add_argument(
        "--scenario-path",
        type=str,
        required=True,
        help="Path to converted scenario directory"
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default="log_replay",
        choices=["no_traffic", "log_replay", "IDM"],
        help="Traffic mode for evaluation"
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="closed_loop",
        choices=["closed_loop", "open_loop"],
        help="Evaluation mode: closed_loop (agent controlled by model) or open_loop (agent follows ground truth)"
    )

    # Output and visualization
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_outputs",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--enable-vis",
        action="store_true",
        help="Enable visualization outputs (images, topdown views)"
    )
    parser.add_argument(
        "--save-perframe",
        action="store_true",
        default=True,
        help="Save per-frame outputs (planning_traj.npy, etc.). Default: True"
    )
    parser.add_argument(
        "--no-save-perframe",
        dest="save_perframe",
        action="store_false",
        help="Disable saving per-frame outputs to save disk space"
    )

    # BEV Calibration
    parser.add_argument(
        "--enable-bev-calibrator",
        action="store_true",
        help="Enable BEV calibrator for domain adaptation (MetaDrive -> Bench2Drive)"
    )
    parser.add_argument(
        "--bev-calibrator-checkpoint",
        type=str,
        default="/home/zhihao/workspace/BridgeSim/calibration/checkpoints/BridgeSim-BevFlow/uniad-b2d/lr1e-4/epoch=9-step=24450.ckpt",
        help="Path to BEV calibrator checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--bev-sample-steps",
        type=int,
        default=50,
        help="Number of Euler sampling steps for BEV flow matching (default: 50)"
    )
    parser.add_argument(
        "--bev-use-ema",
        action="store_true",
        default=True,
        help="Use EMA weights for BEV calibrator (default: True)"
    )
    parser.add_argument(
        "--bev-no-ema",
        dest="bev_use_ema",
        action="store_false",
        help="Disable EMA weights for BEV calibrator"
    )

    # Controller selection
    parser.add_argument(
        "--controller",
        type=str,
        default="pure_pursuit",
        choices=["pid", "pure_pursuit"],
        help="Controller type for vehicle control (default: pure_pursuit)"
    )

    # Replan rate
    parser.add_argument(
        "--replan-rate",
        type=int,
        default=1,
        help="How often to run model inference (1=every frame, 5=every 5 frames, etc.). "
             "Between replans, cached waypoints are consumed sequentially. "
             "Model must predict enough waypoints for the replan interval. (default: 1)"
    )

    # Simulation timestep
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=0.1,
        help="Simulation timestep in seconds (default: 0.1s for 10Hz). "
             "Model trajectory is interpolated to this interval for smooth control."
    )
    parser.add_argument(
        "--ego-replay-frames",
        type=int,
        default=0,
        help="Number of initial frames to replay ego log actions while still running model inference (default: 0)"
    )
    parser.add_argument(
        "--eval-frames",
        type=int,
        default=None,
        help="Number of frames to evaluate after ego replay ends (default: None = run full scenario). "
             "Total frames = min(ego_replay_frames + eval_frames, scenario_length)"
    )
    parser.add_argument(
        "--scorer-type",
        type=str,
        default="legacy",
        choices=["legacy", "navsim"],
        help="Scorer type for closed_loop mode (default: legacy). "
             "'navsim' uses NavSim-style EPDMS scoring with per-frame metrics."
    )
    parser.add_argument(
        "--score-start-frame",
        type=int,
        default=None,
        help="Frame to start calculating scores (default: None = uses ego_replay_frames). "
             "Scoring metrics and route completion are computed only from this frame onwards."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model_type in ["uniad", "vad"] and not args.config:
        parser.error(f"--config is required for {args.model_type} model")

    # Initialize CUDA device
    if args.model_type in ["uniad", "vad"]:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            try:
                from cuda import cudart
                err, = cudart.cudaSetDevice(0)
            except Exception:
                pass
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        else:
            pass  # CUDA not available warning suppressed

    # Create model adapter
    with _silence():
        model_adapter = create_model_adapter(args)

    # Construct output directory with model parameters
    # Format: {output_dir}/{model_type}_rr{replan_rate}_erf{ego_replay_frames}_ef{eval_frames}[_ta{temporal_alpha}_th{temporal_max_history}]/
    eval_frames_str = str(args.eval_frames) if args.eval_frames is not None else "None"
    output_subdir = f"{args.model_type}_rr{args.replan_rate}_erf{args.ego_replay_frames}_ef{eval_frames_str}"
    if args.trajectory_scorer:
        ng = args.num_groups if args.num_groups is not None else (1 if args.model_type == "diffusiondrive" else 10)
        output_subdir += f"_scorer_{args.trajectory_scorer}_ng{ng}"
        if args.num_proposals is not None:
            output_subdir += f"_np{args.num_proposals}"
    if args.enable_temporal_consistency:
        output_subdir += f"_ta{args.temporal_alpha}_th{args.temporal_max_history}"
    full_output_dir = os.path.join(args.output_dir, output_subdir)

    # Create evaluator
    evaluator = BaseEvaluator(
        model_adapter=model_adapter,
        scenario_path=args.scenario_path,
        output_dir=full_output_dir,
        traffic_mode=args.traffic_mode,
        enable_vis=args.enable_vis,
        save_perframe=args.save_perframe,
        eval_mode=args.eval_mode,
        controller_type=args.controller,
        replan_rate=args.replan_rate,
        sim_dt=args.sim_dt,
        ego_replay_frames=args.ego_replay_frames,
        eval_frames=args.eval_frames,
        scorer_type=args.scorer_type,
        score_start_frame=args.score_start_frame,
    )

    # Run evaluation
    evaluator.run()


if __name__ == "__main__":
    main()

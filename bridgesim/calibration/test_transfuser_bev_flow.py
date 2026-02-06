#!/usr/bin/env python3
"""
Evaluate TransFuser BEV flow matching on validation data.

Workflow:
- Load validation scenes from NAVSIM log pickles.
- Extract BEV features for 4 conditions:
  1. navsim: NavSim camera + real LiDAR
  2. navsim_no_lidar: NavSim camera + dummy LiDAR
  3. metadrive: MetaDrive camera + dummy LiDAR
  4. generated: Flow-sampled from metadrive BEV
- Run model predictions on each BEV type.
- Compute L2 loss between predicted trajectories and ground truth.
- Save visualizations and JSON summary.

Supports: TransFuser, DiffusionDrive, DiffusionDriveV2 (same backbone, different trajectory heads).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add navsim to path
SCRIPT_DIR = Path(__file__).resolve().parent
NAVSIM_ROOT = SCRIPT_DIR.parent / "navsim"
sys.path.insert(0, str(NAVSIM_ROOT))

from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from bridgesim.modelzoo.navsim.common.dataclasses import SceneFilter, SensorConfig
from bridgesim.modelzoo.navsim.common.dataloader import SceneLoader

# Import from local modules
from precompute_transfuser_bev_cache import (
    MODEL_TYPES,
    create_bev_visualization,
    create_domain_comparison_visualization,
    create_dummy_lidar_bev,
    extract_fused_features,
    get_predictions_from_fused_features,
    load_model_and_config,
    load_scene_filter_from_yaml,
    save_visualization,
    set_seed,
)
from train_transfuser_bev_flow import FlowMatchingLit


class ValidationDataset(Dataset):
    """
    Dataset for evaluating BEV flow matching on validation scenes.

    Returns camera/lidar features for navsim and metadrive domains,
    plus ground truth trajectory and status features for model inference.
    """

    def __init__(
        self,
        navsim_scene_loader: SceneLoader,
        metadrive_scene_loader: SceneLoader,
        feature_builder: TransfuserFeatureBuilder,
        config: Any,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            navsim_scene_loader: SceneLoader with navsim sensor path
            metadrive_scene_loader: SceneLoader with metadrive sensor path
            feature_builder: TransFuser feature builder
            config: TransFuser config
            max_samples: Optional cap on number of samples
        """
        self.navsim_scene_loader = navsim_scene_loader
        self.metadrive_scene_loader = metadrive_scene_loader
        self.feature_builder = feature_builder
        self.config = config

        # Get all tokens
        self.tokens = navsim_scene_loader.tokens
        self.tokens_per_log = navsim_scene_loader.get_tokens_list_per_log()

        # Build token to log_name mapping
        self.token_to_log: Dict[str, str] = {}
        for log_name, tokens in self.tokens_per_log.items():
            for token in tokens:
                self.token_to_log[token] = log_name

        # Apply max_samples limit
        if max_samples is not None:
            self.tokens = self.tokens[:max_samples]

        print(f"ValidationDataset: {len(self.tokens)} tokens to process")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Get validation sample for a single token.

        Returns:
            Dictionary with:
                - token, log_name
                - navsim_camera, navsim_lidar (real LiDAR)
                - metadrive_camera
                - dummy_lidar (zeros)
                - status_feature (8D: command + velocity + acceleration)
                - gt_trajectory (future poses from scene log)
            or None if loading fails
        """
        token = self.tokens[idx]
        log_name = self.token_to_log.get(token, "unknown")

        try:
            # === NAVSIM domain ===
            navsim_agent_input = self.navsim_scene_loader.get_agent_input_from_token(token)
            navsim_camera = self.feature_builder._get_camera_feature(navsim_agent_input)
            navsim_lidar = self.feature_builder._get_lidar_feature(navsim_agent_input)

            # === MetaDrive domain ===
            metadrive_agent_input = self.metadrive_scene_loader.get_agent_input_from_token(token)
            metadrive_camera = self.feature_builder._get_camera_feature(metadrive_agent_input)

            # === Dummy lidar (zeros) ===
            dummy_lidar = create_dummy_lidar_bev(self.config)

            # === Status feature from ego state ===
            # Use navsim agent_input since it has the correct ego state
            ego_status = navsim_agent_input.ego_statuses[-1]
            status_feature = torch.cat([
                torch.tensor(ego_status.driving_command, dtype=torch.float32),
                torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
                torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
            ])

            # === Ground truth trajectory from scene ===
            # Model predicts 8 poses (4 seconds at 0.5s intervals)
            scene = self.navsim_scene_loader.get_scene_from_token(token)
            gt_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
            gt_poses = torch.tensor(gt_trajectory.poses, dtype=torch.float32)

            return {
                "token": token,
                "log_name": log_name,
                "navsim_camera": navsim_camera,
                "navsim_lidar": navsim_lidar,
                "metadrive_camera": metadrive_camera,
                "dummy_lidar": dummy_lidar,
                "status_feature": status_feature,
                "gt_trajectory": gt_poses,
            }

        except Exception as e:
            print(f"Error loading token {token}: {e}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Collate function that filters None samples and stacks tensors.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    tokens = [b["token"] for b in batch]
    log_names = [b["log_name"] for b in batch]

    navsim_cameras = torch.stack([b["navsim_camera"] for b in batch])
    navsim_lidars = torch.stack([b["navsim_lidar"] for b in batch])
    metadrive_cameras = torch.stack([b["metadrive_camera"] for b in batch])
    dummy_lidars = torch.stack([b["dummy_lidar"] for b in batch])
    status_features = torch.stack([b["status_feature"] for b in batch])
    gt_trajectories = torch.stack([b["gt_trajectory"] for b in batch])

    return {
        "tokens": tokens,
        "log_names": log_names,
        "navsim_cameras": navsim_cameras,
        "navsim_lidars": navsim_lidars,
        "metadrive_cameras": metadrive_cameras,
        "dummy_lidars": dummy_lidars,
        "status_features": status_features,
        "gt_trajectories": gt_trajectories,
    }


def build_scene_loaders(
    data_path: Path,
    navsim_sensor_path: Path,
    metadrive_sensor_path: Path,
    scene_filter: SceneFilter,
) -> Tuple[SceneLoader, SceneLoader]:
    """
    Build SceneLoaders for both NAVSIM and MetaDrive domains.
    """
    # Sensor config for NAVSIM (with lidar)
    navsim_sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=[3],
    )

    # Sensor config for MetaDrive (cameras only, no lidar)
    metadrive_sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )

    navsim_loader = SceneLoader(
        data_path=data_path,
        original_sensor_path=navsim_sensor_path,
        scene_filter=scene_filter,
        sensor_config=navsim_sensor_config,
    )

    metadrive_loader = SceneLoader(
        data_path=data_path,
        original_sensor_path=metadrive_sensor_path,
        scene_filter=scene_filter,
        sensor_config=metadrive_sensor_config,
    )

    return navsim_loader, metadrive_loader


def load_flow_model(
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[FlowMatchingLit, torch.nn.Module]:
    """
    Load flow matching model from checkpoint.

    Returns:
        Tuple of (lit_model, sampler_model)
    """
    lit_model = FlowMatchingLit.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit_model.eval()
    lit_model.to(device)

    sampler = (
        lit_model.ema_flow_model
        if use_ema and lit_model.ema_flow_model is not None
        else lit_model.flow_model
    )
    sampler = sampler.to(device)
    sampler.eval()

    return lit_model, sampler


def compute_trajectory_l2_loss(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute ADE (Average Displacement Error) between predicted and ground truth trajectories
    at multiple time horizons: 1s, 2s, and 4s.

    The model predicts 8 poses at 0.5s intervals:
    - 1s = first 2 timesteps (0.5s, 1.0s)
    - 2s = first 4 timesteps (0.5s, 1.0s, 1.5s, 2.0s)
    - 4s = all 8 timesteps (0.5s to 4.0s)

    Args:
        pred_trajectory: Predicted trajectory (B, T, 3) or (T, 3) with [x, y, heading]
        gt_trajectory: Ground truth trajectory (B, T, 3) or (T, 3) with [x, y, heading]

    Returns:
        Dictionary with ADE at different horizons: {"ade_1s", "ade_2s", "ade_4s"}
    """
    # Use only x, y for L2 loss (ignore heading)
    pred_xy = pred_trajectory[..., :2]
    gt_xy = gt_trajectory[..., :2]

    # Compute L2 distance per timestep
    l2_per_timestep = torch.sqrt(((pred_xy - gt_xy) ** 2).sum(dim=-1))

    # Compute ADE at different horizons
    # 1s = 2 timesteps, 2s = 4 timesteps, 4s = 8 timesteps
    ade_1s = l2_per_timestep[..., :2].mean(dim=-1)
    ade_2s = l2_per_timestep[..., :4].mean(dim=-1)
    ade_4s = l2_per_timestep.mean(dim=-1)

    return {
        "ade_1s": ade_1s,
        "ade_2s": ade_2s,
        "ade_4s": ade_4s,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TransFuser BEV flow matching on validation data."
    )

    # Data paths
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        required=True,
        help="Path to NAVSIM log pickles directory.",
    )
    parser.add_argument(
        "--navsim-sensor-path",
        type=Path,
        required=True,
        help="Path to NAVSIM sensor blobs directory.",
    )
    parser.add_argument(
        "--metadrive-sensor-path",
        type=Path,
        required=True,
        help="Path to MetaDrive sensor images directory.",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to TransFuser/DiffusionDrive model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="transfuser",
        choices=MODEL_TYPES,
        help="Model type: transfuser, diffusiondrive, or diffusiondrivev2.",
    )
    parser.add_argument(
        "--plan-anchor-path",
        type=Path,
        default=None,
        help="Path to plan anchor file (.npy) for DiffusionDrive/V2 models.",
    )

    # Flow model
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        required=True,
        help="Path to flow matching model checkpoint.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=50,
        help="Euler sampling steps for flow matching.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        default=True,
        help="Use EMA flow model weights (default: True).",
    )
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Force using non-EMA flow weights.",
    )

    # Data loading
    parser.add_argument(
        "--split-yaml",
        type=Path,
        default=None,
        help="Path to split YAML file defining SceneFilter.",
    )
    parser.add_argument(
        "--log-names",
        type=str,
        nargs="+",
        default=None,
        help="Specific log names to process.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process.",
    )

    # Output
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bev_flow_eval"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--save-gen-bev-dir",
        type=Path,
        default=None,
        help="Optional directory to save generated BEV tensors.",
    )

    # Processing
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available).",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of BEV features and predictions.",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: out_dir/visualizations).",
    )
    parser.add_argument(
        "--vis-num-samples",
        type=int,
        default=100,
        help="Number of samples to visualize.",
    )

    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create output directories
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir = args.vis_output_dir or (args.out_dir / "visualizations")

    # Load TransFuser/DiffusionDrive model
    print("Loading prediction model...")
    model, config = load_model_and_config(
        args.checkpoint, device, args.model_type, args.plan_anchor_path
    )
    feature_builder = TransfuserFeatureBuilder(config)

    # Load flow model
    print("Loading flow model...")
    _, flow_sampler = load_flow_model(args.flow_ckpt, device, use_ema=args.use_ema)

    # Setup scene filter
    if args.split_yaml is not None:
        print(f"Loading scene filter from YAML: {args.split_yaml}")
        scene_filter = load_scene_filter_from_yaml(
            args.split_yaml,
            log_names_override=args.log_names,
            max_scenes_override=args.max_scenes,
        )
    else:
        scene_filter = SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            has_route=True,
            log_names=args.log_names,
            max_scenes=args.max_scenes,
        )

    # Build scene loaders
    print("Building scene loaders...")
    navsim_loader, metadrive_loader = build_scene_loaders(
        data_path=args.navsim_log_path,
        navsim_sensor_path=args.navsim_sensor_path,
        metadrive_sensor_path=args.metadrive_sensor_path,
        scene_filter=scene_filter,
    )
    print(f"Total tokens: {len(navsim_loader.tokens)}")

    # Create dataset and dataloader
    dataset = ValidationDataset(
        navsim_scene_loader=navsim_loader,
        metadrive_scene_loader=metadrive_loader,
        feature_builder=feature_builder,
        config=config,
        max_samples=args.max_scenes,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Tracking metrics - now tracking ADE at 1s, 2s, 4s horizons
    records: List[Dict] = []
    # Each domain has metrics for 3 time horizons
    all_ade_navsim = {"1s": [], "2s": [], "4s": []}
    all_ade_navsim_no_lidar = {"1s": [], "2s": [], "4s": []}
    all_ade_metadrive = {"1s": [], "2s": [], "4s": []}
    all_ade_generated = {"1s": [], "2s": [], "4s": []}
    all_bev_mse = []
    all_bev_cosine = []

    vis_count = 0
    progress = tqdm(dataloader, desc="Evaluating")

    for batch in progress:
        if batch is None:
            continue

        tokens = batch["tokens"]
        log_names = batch["log_names"]
        batch_size = len(tokens)

        # Move tensors to device
        navsim_cameras = batch["navsim_cameras"].to(device)
        navsim_lidars = batch["navsim_lidars"].to(device)
        metadrive_cameras = batch["metadrive_cameras"].to(device)
        dummy_lidars = batch["dummy_lidars"].to(device)
        status_features = batch["status_features"].to(device)
        gt_trajectories = batch["gt_trajectories"].to(device)

        # === Extract BEV features for all 4 types ===

        # 1. navsim (with real lidar)
        navsim_bev = extract_fused_features(
            model, navsim_cameras, navsim_lidars, device
        )

        # 2. navsim_no_lidar (with dummy lidar)
        navsim_no_lidar_bev = extract_fused_features(
            model, navsim_cameras, dummy_lidars, device
        )

        # 3. metadrive (with dummy lidar)
        metadrive_bev = extract_fused_features(
            model, metadrive_cameras, dummy_lidars, device
        )

        # 4. generated (flow-sampled from metadrive)
        # Flow model expects (B, C, H, W) directly - no flattening needed
        generated_bev = flow_sampler.sample(metadrive_bev, sample_steps=args.sample_steps)

        # === Get predictions from each BEV ===
        navsim_preds = get_predictions_from_fused_features(
            model, navsim_bev, status_features, device, args.model_type
        )
        navsim_no_lidar_preds = get_predictions_from_fused_features(
            model, navsim_no_lidar_bev, status_features, device, args.model_type
        )
        metadrive_preds = get_predictions_from_fused_features(
            model, metadrive_bev, status_features, device, args.model_type
        )
        generated_preds = get_predictions_from_fused_features(
            model, generated_bev, status_features, device, args.model_type
        )

        # === Compute L2 losses ===
        for i in range(batch_size):
            token = tokens[i]
            log_name = log_names[i]
            gt_traj = gt_trajectories[i].cpu()

            # Helper to get trajectory safely
            def get_traj(preds, idx):
                if preds is None or preds["trajectory"] is None:
                    return None
                return preds["trajectory"][idx]

            navsim_traj = get_traj(navsim_preds, i)
            navsim_no_lidar_traj = get_traj(navsim_no_lidar_preds, i)
            metadrive_traj = get_traj(metadrive_preds, i)
            generated_traj = get_traj(generated_preds, i)

            # Compute ADE at multiple horizons (1s, 2s, 4s)
            ade_navsim = compute_trajectory_l2_loss(navsim_traj, gt_traj) if navsim_traj is not None else None
            ade_navsim_no_lidar = compute_trajectory_l2_loss(navsim_no_lidar_traj, gt_traj) if navsim_no_lidar_traj is not None else None
            ade_metadrive = compute_trajectory_l2_loss(metadrive_traj, gt_traj) if metadrive_traj is not None else None
            ade_generated = compute_trajectory_l2_loss(generated_traj, gt_traj) if generated_traj is not None else None

            # Track metrics for each horizon
            if ade_navsim is not None:
                all_ade_navsim["1s"].append(ade_navsim["ade_1s"].item())
                all_ade_navsim["2s"].append(ade_navsim["ade_2s"].item())
                all_ade_navsim["4s"].append(ade_navsim["ade_4s"].item())
            if ade_navsim_no_lidar is not None:
                all_ade_navsim_no_lidar["1s"].append(ade_navsim_no_lidar["ade_1s"].item())
                all_ade_navsim_no_lidar["2s"].append(ade_navsim_no_lidar["ade_2s"].item())
                all_ade_navsim_no_lidar["4s"].append(ade_navsim_no_lidar["ade_4s"].item())
            if ade_metadrive is not None:
                all_ade_metadrive["1s"].append(ade_metadrive["ade_1s"].item())
                all_ade_metadrive["2s"].append(ade_metadrive["ade_2s"].item())
                all_ade_metadrive["4s"].append(ade_metadrive["ade_4s"].item())
            if ade_generated is not None:
                all_ade_generated["1s"].append(ade_generated["ade_1s"].item())
                all_ade_generated["2s"].append(ade_generated["ade_2s"].item())
                all_ade_generated["4s"].append(ade_generated["ade_4s"].item())

            # BEV reconstruction metrics (generated vs navsim)
            gen_bev_i = generated_bev[i]
            navsim_bev_i = navsim_bev[i]
            bev_mse = ((gen_bev_i - navsim_bev_i) ** 2).mean().item()
            bev_cosine = torch.nn.functional.cosine_similarity(
                gen_bev_i.flatten().unsqueeze(0),
                navsim_bev_i.flatten().unsqueeze(0),
            ).item()

            all_bev_mse.append(bev_mse)
            all_bev_cosine.append(bev_cosine)

            # Record with ADE at all horizons
            record = {
                "token": token,
                "log_name": log_name,
                "ade_navsim_1s": ade_navsim["ade_1s"].item() if ade_navsim else None,
                "ade_navsim_2s": ade_navsim["ade_2s"].item() if ade_navsim else None,
                "ade_navsim_4s": ade_navsim["ade_4s"].item() if ade_navsim else None,
                "ade_navsim_no_lidar_1s": ade_navsim_no_lidar["ade_1s"].item() if ade_navsim_no_lidar else None,
                "ade_navsim_no_lidar_2s": ade_navsim_no_lidar["ade_2s"].item() if ade_navsim_no_lidar else None,
                "ade_navsim_no_lidar_4s": ade_navsim_no_lidar["ade_4s"].item() if ade_navsim_no_lidar else None,
                "ade_metadrive_1s": ade_metadrive["ade_1s"].item() if ade_metadrive else None,
                "ade_metadrive_2s": ade_metadrive["ade_2s"].item() if ade_metadrive else None,
                "ade_metadrive_4s": ade_metadrive["ade_4s"].item() if ade_metadrive else None,
                "ade_generated_1s": ade_generated["ade_1s"].item() if ade_generated else None,
                "ade_generated_2s": ade_generated["ade_2s"].item() if ade_generated else None,
                "ade_generated_4s": ade_generated["ade_4s"].item() if ade_generated else None,
                "bev_mse": bev_mse,
                "bev_cosine": bev_cosine,
            }
            records.append(record)

            # Save generated BEV if requested
            if args.save_gen_bev_dir is not None:
                gen_bev_path = args.save_gen_bev_dir / log_name / f"{token}.pt"
                gen_bev_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "bev": gen_bev_i.cpu(),
                    "token": token,
                    "log_name": log_name,
                }, gen_bev_path)

            # Visualization
            if args.visualize and vis_count < args.vis_num_samples:
                # Get GT trajectory for visualization
                gt_traj_np = gt_traj.numpy()

                # Create visualizations for each BEV type
                def make_vis(camera, lidar, preds, title):
                    traj = None
                    if preds is not None and preds["trajectory"] is not None:
                        traj = preds["trajectory"][i].numpy()
                    return create_bev_visualization(
                        camera_image=camera[i].cpu().numpy(),
                        lidar_feature=lidar[i].cpu().numpy(),
                        bev_semantic_map=preds["bev_semantic_map"][i].numpy() if preds else None,
                        agent_states=preds["agent_states"][i].numpy() if preds else None,
                        agent_labels=preds["agent_labels"][i].numpy() if preds else None,
                        trajectory=traj,
                        gt_trajectory=gt_traj_np,
                        config=config,
                        title=title,
                    )

                navsim_vis = make_vis(
                    navsim_cameras, navsim_lidars, navsim_preds, "NavSim (with LiDAR)"
                )
                navsim_no_lidar_vis = make_vis(
                    navsim_cameras, dummy_lidars, navsim_no_lidar_preds, "NavSim (no LiDAR)"
                )
                metadrive_vis = make_vis(
                    metadrive_cameras, dummy_lidars, metadrive_preds, "MetaDrive"
                )
                # For generated, use metadrive camera but generated predictions
                generated_vis = make_vis(
                    metadrive_cameras, dummy_lidars, generated_preds, "Generated (flow-sampled)"
                )

                # Save individual visualizations
                save_visualization(navsim_vis, vis_output_dir / "navsim", token)
                save_visualization(navsim_no_lidar_vis, vis_output_dir / "navsim_no_lidar", token)
                save_visualization(metadrive_vis, vis_output_dir / "metadrive", token)
                save_visualization(generated_vis, vis_output_dir / "generated", token)

                vis_count += 1

        # Update progress bar (show 4s ADE for compact display)
        if all_ade_generated["4s"]:
            progress.set_postfix({
                "ade4s_nav": f"{np.mean(all_ade_navsim['4s']):.3f}",
                "ade4s_gen": f"{np.mean(all_ade_generated['4s']):.3f}",
                "ade4s_md": f"{np.mean(all_ade_metadrive['4s']):.3f}",
            })

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === Compute summary statistics ===
    def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
        """Helper to compute mean and std."""
        if not values:
            return {"mean": None, "std": None}
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary = {
        "num_samples": len(records),
        # ADE at different horizons for each domain
        "ade_navsim": {
            "1s": compute_stats(all_ade_navsim["1s"]),
            "2s": compute_stats(all_ade_navsim["2s"]),
            "4s": compute_stats(all_ade_navsim["4s"]),
        },
        "ade_navsim_no_lidar": {
            "1s": compute_stats(all_ade_navsim_no_lidar["1s"]),
            "2s": compute_stats(all_ade_navsim_no_lidar["2s"]),
            "4s": compute_stats(all_ade_navsim_no_lidar["4s"]),
        },
        "ade_metadrive": {
            "1s": compute_stats(all_ade_metadrive["1s"]),
            "2s": compute_stats(all_ade_metadrive["2s"]),
            "4s": compute_stats(all_ade_metadrive["4s"]),
        },
        "ade_generated": {
            "1s": compute_stats(all_ade_generated["1s"]),
            "2s": compute_stats(all_ade_generated["2s"]),
            "4s": compute_stats(all_ade_generated["4s"]),
        },
        "bev_mse": compute_stats(all_bev_mse),
        "bev_cosine": compute_stats(all_bev_cosine),
        "records": records,
    }

    # Save summary
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(records)}")

    def fmt_ade(stats: Dict) -> str:
        """Format ADE stats as string."""
        if stats["mean"] is None:
            return "N/A"
        return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"

    print(f"\nADE (Average Displacement Error) in meters (lower is better):")
    print(f"{'Domain':<25} {'ADE@1s':<20} {'ADE@2s':<20} {'ADE@4s':<20}")
    print(f"{'-'*85}")
    print(f"{'NavSim (with LiDAR)':<25} {fmt_ade(summary['ade_navsim']['1s']):<20} {fmt_ade(summary['ade_navsim']['2s']):<20} {fmt_ade(summary['ade_navsim']['4s']):<20}")
    print(f"{'NavSim (no LiDAR)':<25} {fmt_ade(summary['ade_navsim_no_lidar']['1s']):<20} {fmt_ade(summary['ade_navsim_no_lidar']['2s']):<20} {fmt_ade(summary['ade_navsim_no_lidar']['4s']):<20}")
    print(f"{'Generated (flow)':<25} {fmt_ade(summary['ade_generated']['1s']):<20} {fmt_ade(summary['ade_generated']['2s']):<20} {fmt_ade(summary['ade_generated']['4s']):<20}")
    print(f"{'MetaDrive':<25} {fmt_ade(summary['ade_metadrive']['1s']):<20} {fmt_ade(summary['ade_metadrive']['2s']):<20} {fmt_ade(summary['ade_metadrive']['4s']):<20}")

    print(f"\nBEV Reconstruction (generated vs navsim):")
    print(f"  MSE:    {summary['bev_mse']['mean']:.6f} +/- {summary['bev_mse']['std']:.6f}")
    print(f"  Cosine: {summary['bev_cosine']['mean']:.4f} +/- {summary['bev_cosine']['std']:.4f}")
    print(f"\nResults saved to: {summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


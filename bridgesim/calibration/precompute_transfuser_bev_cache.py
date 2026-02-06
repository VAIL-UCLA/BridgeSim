#!/usr/bin/env python3
"""
Precompute TransFuser-style BEV features (fused_features) and cache them per token.

This script extracts BEV features from the TransFuser backbone for both
NAVSIM (with real LiDAR) and MetaDrive (with dummy LiDAR) domains.

Supported models:
- transfuser: Original TransFuser model
- diffusiondrive: DiffusionDrive model (TransFuser backbone + diffusion trajectory head)
- diffusiondrivev2: DiffusionDriveV2 model (TransFuser backbone + diffusion + RL-based scoring)

All three models share the same TransFuser backbone architecture that produces
identical fused_features with shape (512, 8, 8).

Cache layout: cache_dir/<domain>/<log_name>.h5
    - Each token is stored as a dataset inside the H5 file
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

# Add navsim to path
SCRIPT_DIR = Path(__file__).resolve().parent
NAVSIM_ROOT = SCRIPT_DIR.parent / "navsim"
sys.path.insert(0, str(NAVSIM_ROOT))

from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from bridgesim.modelzoo.navsim.common.dataclasses import AgentInput, SceneFilter, SensorConfig
from bridgesim.modelzoo.navsim.common.dataloader import SceneLoader

# Model type to import mapping - import configs lazily
MODEL_TYPES = ["transfuser", "diffusiondrive", "diffusiondrivev2"]

# ============================================================================
# Visualization Configuration
# ============================================================================

# Color palette for visualization (Ellis-5 and Tab-10 inspired)
VIS_COLORS = {
    "ego": "#DE7061",       # red
    "vehicle": "#699CDB",   # blue
    "pedestrian": "#b07aa1", # violet
    "bicycle": "#E38C47",   # orange
    "trajectory_gt": "#59a14f",   # green
    "trajectory_pred": "#DE7061", # red
    "lane": "#D3D3D3",      # light grey
    "intersection": "#D3D3D3",
    "crosswalk": "#b07aa1",
    "background": "#FFFFFF",
}

# BEV semantic class colors (based on TransfuserConfig defaults)
BEV_CLASS_COLORS = {
    0: (255, 255, 255),  # background - white
    1: (211, 211, 211),  # lane - light grey
    2: (211, 211, 211),  # intersection - light grey
    3: (105, 156, 219),  # vehicle - blue
    4: (176, 122, 161),  # pedestrian - violet
    5: (227, 140, 71),   # bicycle - orange
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def semantic_map_to_rgb(
    semantic_map: np.ndarray,
    num_classes: int = 6,
) -> np.ndarray:
    """
    Convert semantic BEV map to RGB image.

    Args:
        semantic_map: Semantic map (H, W) with class indices or (C, H, W) logits
        num_classes: Number of semantic classes

    Returns:
        RGB image (H, W, 3) as uint8
    """
    # Handle logits input
    if semantic_map.ndim == 3:
        semantic_map = semantic_map.argmax(axis=0)

    height, width = semantic_map.shape
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(num_classes):
        if label in BEV_CLASS_COLORS:
            rgb_map[semantic_map == label] = BEV_CLASS_COLORS[label]

    return rgb_map


def draw_bounding_boxes(
    image: np.ndarray,
    agent_states: np.ndarray,
    agent_labels: np.ndarray,
    pixel_size: float = 0.25,
    color: Tuple[int, int, int] = (105, 156, 219),
    thickness: int = 2,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """
    Draw agent bounding boxes on BEV image.

    Args:
        image: BEV image (H, W, 3)
        agent_states: Agent states (N, 5) with [x, y, heading, length, width]
        agent_labels: Agent confidence scores (N,)
        pixel_size: Meters per pixel
        color: RGB color for boxes
        thickness: Line thickness
        confidence_threshold: Minimum confidence to draw

    Returns:
        Image with bounding boxes drawn
    """
    height, width = image.shape[:2]
    center = np.array([height / 2.0, width / 2.0])

    image = image.copy()

    for i, (state, label) in enumerate(zip(agent_states, agent_labels)):
        if label < confidence_threshold:
            continue

        x, y, heading = state[:3]
        length, width_box = state[3], state[4]

        # Create oriented box
        box = OrientedBox(StateSE2(x, y, heading), length, width_box, 1.0)
        corners = np.array(box.geometry.exterior.coords).reshape((-1, 1, 2))

        # Convert to pixel coordinates
        corners_px = (corners / pixel_size) + center
        corners_px = corners_px.astype(np.int32)
        corners_px = np.flip(corners_px, axis=-1)  # x, y -> y, x for cv2

        cv2.polylines(image, [corners_px], isClosed=True, color=color, thickness=thickness)

    return image


def draw_trajectory(
    image: np.ndarray,
    trajectory: np.ndarray,
    pixel_size: float = 0.25,
    color: Tuple[int, int, int] = (89, 161, 79),
    point_radius: int = 4,
) -> np.ndarray:
    """
    Draw trajectory points on BEV image.

    Args:
        image: BEV image (H, W, 3)
        trajectory: Trajectory poses (T, 2) or (T, 3) with [x, y, (heading)]
        pixel_size: Meters per pixel
        color: RGB color for trajectory
        point_radius: Radius of trajectory points

    Returns:
        Image with trajectory drawn
    """
    height, width = image.shape[:2]
    center = np.array([height / 2.0, width / 2.0])

    image = image.copy()

    for pose in trajectory:
        x, y = pose[:2]
        px = int(x / pixel_size + center[0])
        py = int(y / pixel_size + center[1])
        cv2.circle(image, (py, px), point_radius, color, -1)

    return image


def draw_ego_vehicle(
    image: np.ndarray,
    pixel_size: float = 0.25,
    color: Tuple[int, int, int] = (222, 112, 97),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw ego vehicle at origin of BEV image.

    Args:
        image: BEV image (H, W, 3)
        pixel_size: Meters per pixel
        color: RGB color for ego
        thickness: Line thickness

    Returns:
        Image with ego vehicle drawn
    """
    height, width = image.shape[:2]
    center = np.array([height / 2.0, width / 2.0])

    image = image.copy()

    # Use Pacifica vehicle dimensions
    car_footprint = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(0, 0, 0),
        vehicle_parameters=get_pacifica_parameters(),
    )
    box = car_footprint.oriented_box
    corners = np.array([[c.x, c.y] for c in box.all_corners()])
    corners = np.vstack([corners, corners[0]])  # Close the polygon

    # Convert to pixel coordinates
    corners_px = (corners / pixel_size) + center
    corners_px = corners_px.astype(np.int32).reshape((-1, 1, 2))
    corners_px = np.flip(corners_px, axis=-1)

    cv2.polylines(image, [corners_px], isClosed=True, color=color, thickness=thickness)
    cv2.fillPoly(image, [corners_px], color=color)

    return image


def lidar_histogram_to_rgb(
    lidar_feature: np.ndarray,
) -> np.ndarray:
    """
    Convert LiDAR histogram to grayscale RGB for visualization.

    Args:
        lidar_feature: LiDAR BEV histogram (C, H, W) or (H, W)

    Returns:
        RGB image (H, W, 3) as uint8
    """
    if lidar_feature.ndim == 3:
        # Sum channels or take max
        lidar_2d = lidar_feature.max(axis=0)
    else:
        lidar_2d = lidar_feature

    # Normalize and convert to grayscale
    lidar_norm = (lidar_2d - lidar_2d.min()) / (lidar_2d.max() - lidar_2d.min() + 1e-8)
    lidar_gray = (255 - lidar_norm * 255).astype(np.uint8)

    # Convert to RGB
    rgb = np.stack([lidar_gray, lidar_gray, lidar_gray], axis=-1)

    return rgb


def create_bev_visualization(
    camera_image: np.ndarray,
    lidar_feature: np.ndarray,
    bev_semantic_map: Optional[np.ndarray] = None,
    agent_states: Optional[np.ndarray] = None,
    agent_labels: Optional[np.ndarray] = None,
    trajectory: Optional[np.ndarray] = None,
    gt_trajectory: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    title: str = "",
) -> np.ndarray:
    """
    Create comprehensive BEV visualization with all downstream outputs.

    Args:
        camera_image: Camera input (3, H, W) or (H, W, 3)
        lidar_feature: LiDAR BEV histogram
        bev_semantic_map: Semantic segmentation map
        agent_states: Detected agent states
        agent_labels: Agent confidence scores
        trajectory: Predicted trajectory
        gt_trajectory: Ground truth trajectory (optional)
        config: TransFuser config
        title: Title for the visualization

    Returns:
        Combined visualization image
    """
    pixel_size = config.bev_pixel_size if config else 0.25

    # Prepare camera image
    if camera_image.ndim == 3 and camera_image.shape[0] == 3:
        camera_image = camera_image.transpose(1, 2, 0)
    camera_rgb = (camera_image * 255).astype(np.uint8) if camera_image.max() <= 1 else camera_image.astype(np.uint8)

    # Prepare LiDAR visualization
    lidar_rgb = lidar_histogram_to_rgb(lidar_feature)
    lidar_rgb = cv2.resize(lidar_rgb, (256, 256))

    # Prepare semantic map visualization
    if bev_semantic_map is not None:
        sem_rgb = semantic_map_to_rgb(bev_semantic_map)
        # Flip for proper orientation (ego facing up)
        sem_rgb = sem_rgb[::-1, ::-1]
    else:
        sem_rgb = np.ones((256, 256, 3), dtype=np.uint8) * 200

    # Draw detections on semantic map
    vis_rgb = sem_rgb.copy()

    if agent_states is not None and agent_labels is not None:
        vis_rgb = draw_bounding_boxes(
            vis_rgb, agent_states, agent_labels,
            pixel_size=pixel_size,
            color=hex_to_rgb(VIS_COLORS["vehicle"]),
            confidence_threshold=0.5,
        )

    # Draw trajectories
    if gt_trajectory is not None:
        vis_rgb = draw_trajectory(
            vis_rgb, gt_trajectory,
            pixel_size=pixel_size,
            color=hex_to_rgb(VIS_COLORS["trajectory_gt"]),
            point_radius=5,
        )

    if trajectory is not None:
        vis_rgb = draw_trajectory(
            vis_rgb, trajectory,
            pixel_size=pixel_size,
            color=hex_to_rgb(VIS_COLORS["trajectory_pred"]),
            point_radius=4,
        )

    # Draw ego vehicle
    vis_rgb = draw_ego_vehicle(
        vis_rgb,
        pixel_size=pixel_size,
        color=hex_to_rgb(VIS_COLORS["ego"]),
    )

    # Flip for display
    vis_rgb = vis_rgb[::-1, ::-1]
    lidar_rgb = lidar_rgb[::-1, ::-1]

    # Create composite image
    # Layout: [Camera (512x128)] on top, [LiDAR (256x256) | BEV+Det (256x256)] below
    cam_h, cam_w = camera_rgb.shape[:2]
    target_cam_w = 512
    target_cam_h = int(cam_h * target_cam_w / cam_w)
    camera_resized = cv2.resize(camera_rgb, (target_cam_w, target_cam_h))

    # Create canvas
    canvas_w = 512
    canvas_h = target_cam_h + 256 + 30  # +30 for title
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Add title
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Place camera
    canvas[30:30+target_cam_h, :target_cam_w] = camera_resized

    # Place LiDAR and BEV
    y_offset = 30 + target_cam_h
    canvas[y_offset:y_offset+256, :256] = lidar_rgb
    canvas[y_offset:y_offset+256, 256:512] = cv2.resize(vis_rgb, (256, 256))

    # Add labels
    cv2.putText(canvas, "LiDAR BEV", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, "Detection + Tracking", (266, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return canvas


def create_domain_comparison_visualization(
    navsim_vis: np.ndarray,
    navsim_no_lidar_vis: np.ndarray,
    metadrive_vis: np.ndarray,
    token: str,
) -> np.ndarray:
    """
    Create side-by-side comparison of three domains.

    Args:
        navsim_vis: NAVSIM visualization
        navsim_no_lidar_vis: NAVSIM without LiDAR visualization
        metadrive_vis: MetaDrive visualization
        token: Scene token for title

    Returns:
        Combined comparison image
    """
    h = max(navsim_vis.shape[0], navsim_no_lidar_vis.shape[0], metadrive_vis.shape[0])
    w = navsim_vis.shape[1]

    # Create canvas for 3 columns
    canvas = np.ones((h + 40, w * 3, 3), dtype=np.uint8) * 255

    # Add main title
    cv2.putText(canvas, f"Token: {token[:20]}...", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Place visualizations
    canvas[40:40+navsim_vis.shape[0], :w] = navsim_vis
    canvas[40:40+navsim_no_lidar_vis.shape[0], w:w*2] = navsim_no_lidar_vis
    canvas[40:40+metadrive_vis.shape[0], w*2:w*3] = metadrive_vis

    return canvas


def save_visualization(
    image: np.ndarray,
    output_path: Path,
    token: str,
    suffix: str = "",
) -> None:
    """
    Save visualization image to disk.

    Args:
        image: Image to save (H, W, 3)
        output_path: Output directory
        token: Scene token
        suffix: Optional suffix for filename
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{token}{suffix}.png"
    filepath = output_path / filename

    # Convert RGB to BGR for cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filepath), image_bgr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute TransFuser BEV features for NAVSIM/MetaDrive."
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
        default=None,
        help="Path to MetaDrive sensor images directory (optional).",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="transfuser",
        choices=MODEL_TYPES,
        help="Model type: transfuser, diffusiondrive, or diffusiondrivev2 (default: transfuser).",
    )
    parser.add_argument(
        "--plan-anchor-path",
        type=Path,
        default=None,
        help="Path to plan anchor file (.npy) for DiffusionDrive/V2 models. Overrides config default.",
    )

    # Output
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Output directory for cached BEV features.",
    )

    # Data loading
    parser.add_argument(
        "--split-yaml",
        type=Path,
        default=None,
        help="Path to split YAML file (e.g., navtrain_part1.yaml) defining SceneFilter.",
    )
    parser.add_argument(
        "--log-names",
        type=str,
        nargs="+",
        default=None,
        help="Specific log names to process (overrides split-yaml if both provided).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (overrides split-yaml if provided).",
    )

    # Processing
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["navsim"],
        choices=["navsim", "metadrive"],
        help="Domains to process (default: navsim).",
    )

    # Distributed
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed processing.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tokens that already have cached BEV.",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of BEV features and model predictions.",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: cache_dir/visualizations).",
    )
    parser.add_argument(
        "--vis-num-samples",
        type=int,
        default=100,
        help="Number of samples to visualize (default: 100).",
    )
    parser.add_argument(
        "--vis-compare-domains",
        action="store_true",
        help="Create side-by-side domain comparison visualizations.",
    )
    parser.add_argument(
        "--vis-include-predictions",
        action="store_true",
        help="Include model predictions (semantic map, detections, trajectory) in visualizations.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scene_filter_from_yaml(
    yaml_path: Path,
    log_names_override: Optional[List[str]] = None,
    max_scenes_override: Optional[int] = None,
) -> SceneFilter:
    """
    Load SceneFilter from a YAML file.

    Args:
        yaml_path: Path to the YAML file
        log_names_override: Override log_names from YAML if provided
        max_scenes_override: Override max_scenes from YAML if provided

    Returns:
        SceneFilter instance
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract SceneFilter parameters (ignore Hydra-specific keys)
    scene_filter_kwargs = {
        k: v for k, v in config.items() if not k.startswith("_")
    }

    # Apply overrides
    if log_names_override is not None:
        scene_filter_kwargs["log_names"] = log_names_override
    if max_scenes_override is not None:
        scene_filter_kwargs["max_scenes"] = max_scenes_override

    return SceneFilter(**scene_filter_kwargs)


def load_model_and_config(
    checkpoint_path: Path,
    device: torch.device,
    model_type: str = "transfuser",
    plan_anchor_path: Optional[Path] = None,
) -> Tuple[nn.Module, Any]:
    """
    Load model from checkpoint based on model type.

    All three model types (transfuser, diffusiondrive, diffusiondrivev2) use
    the same TransFuser backbone that produces identical fused_features.

    Args:
        checkpoint_path: Path to checkpoint file (.ckpt)
        device: Device to load model on
        model_type: One of "transfuser", "diffusiondrive", "diffusiondrivev2"
        plan_anchor_path: Optional path to plan anchor file for DiffusionDrive/V2

    Returns:
        Tuple of (model, config)
    """
    print(f"Loading {model_type} model...")

    # Import model-specific classes based on model type
    if model_type == "transfuser":
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_model import TransfuserModel
        from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
        model = TransfuserModel(trajectory_sampling, config)

        # State dict key prefixes to strip
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    elif model_type == "diffusiondrive":
        from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
        from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        # Override plan_anchor_path if provided
        if plan_anchor_path is not None:
            config.plan_anchor_path = str(plan_anchor_path)
            print(f"Using plan anchor path: {plan_anchor_path}")
        model = V2TransfuserModel(config)

        # State dict key prefixes to strip for DiffusionDrive
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    elif model_type == "diffusiondrivev2":
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig

        config = TransfuserConfig()
        # Override plan_anchor_path if provided
        if plan_anchor_path is not None:
            config.plan_anchor_path = str(plan_anchor_path)
            print(f"Using plan anchor path: {plan_anchor_path}")
        model = V2TransfuserModel(config)

        # State dict key prefixes to strip for DiffusionDriveV2
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip prefixes if trained with Lightning/DDP
    clean_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in prefixes_to_strip:
            new_key = new_key.replace(prefix, "")
        clean_sd[new_key] = v

    # Load with strict=False to handle potential mismatches
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"{model_type} model loaded successfully.")
    return model, config


def create_dummy_lidar_bev(config: Any) -> torch.Tensor:
    """
    Create dummy LiDAR BEV representation for MetaDrive domain.

    Args:
        config: TransFuser config

    Returns:
        Zero tensor of shape (lidar_seq_len, H, W)
    """
    return torch.zeros(
        config.lidar_seq_len,
        config.lidar_resolution_height,
        config.lidar_resolution_width,
        dtype=torch.float32,
    )


def extract_fused_features(
    model: nn.Module,
    camera_feature: torch.Tensor,
    lidar_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract fused_features (BEV) from TransFuser backbone.

    Args:
        model: TransFuser model
        camera_feature: Camera input tensor (B, 3, 256, 1024)
        lidar_feature: LiDAR BEV tensor (B, 1, 256, 256)
        device: Device to run on

    Returns:
        fused_features tensor (B, 512, 8, 8)
    """
    camera_feature = camera_feature.to(device)
    lidar_feature = lidar_feature.to(device)

    with torch.no_grad():
        # Run backbone forward pass
        # Returns: (bev_feature_upscale, fused_features, image_feature_grid)
        _, fused_features, _ = model._backbone(camera_feature, lidar_feature)

    return fused_features


def get_predictions_from_fused_features(
    model: nn.Module,
    fused_features: torch.Tensor,
    status_feature: torch.Tensor,
    device: torch.device,
    model_type: str = "transfuser",
) -> Dict[str, torch.Tensor]:
    """
    Get model predictions (semantic map, detection, trajectory) from cached fused_features.

    This reconstructs the forward pass from fused_features by:
    1. Applying top_down to get bev_feature_upscale
    2. Running the transformer decoder and prediction heads

    Args:
        model: TransFuser model (or DiffusionDrive/DiffusionDriveV2)
        fused_features: Cached BEV features (B, 512, 8, 8)
        status_feature: Status feature tensor (B, 8) - can be zeros for visualization
        device: Device to run on
        model_type: One of "transfuser", "diffusiondrive", "diffusiondrivev2"

    Returns:
        Dictionary with predictions:
            - bev_semantic_map: (B, C, H, W) semantic segmentation logits
            - agent_states: (B, N, 5) detected agent states [x, y, heading, length, width]
            - agent_labels: (B, N) detection confidence scores
            - trajectory: (B, T, 3) predicted trajectory [x, y, heading] (TransFuser only)
    """
    fused_features = fused_features.to(device)
    status_feature = status_feature.to(device)

    batch_size = fused_features.shape[0]

    with torch.no_grad():
        # Step 1: Apply top_down to get bev_feature_upscale for semantic head
        # fused_features shape: (B, 512, 8, 8)
        bev_feature_upscale = model._backbone.top_down(fused_features)

        # Step 2: Follow transfuser_model.py lines 104-122
        # Downscale and flatten for transformer
        bev_feature = model._bev_downscale(fused_features).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)  # (B, 64, d_model)

        # Encode status feature
        status_encoding = model._status_encoding(status_feature)

        # Build key-value sequence
        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval = keyval + model._keyval_embedding.weight[None, ...]

        # Build query embeddings
        query = model._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)

        # Run transformer decoder
        query_out = model._tf_decoder(query, keyval)

        # Get semantic BEV map
        bev_semantic_map = model._bev_semantic_head(bev_feature_upscale)

        # Split queries for trajectory and agents
        trajectory_query, agents_query = query_out.split(model._query_splits, dim=1)

        # Get agent detection predictions (common to all model types)
        agents_output = model._agent_head(agents_query)

        # Get trajectory prediction (model-type specific)
        if model_type == "transfuser":
            # TransFuser: Simple MLP trajectory head
            trajectory_output = model._trajectory_head(trajectory_query)
            trajectory = trajectory_output["trajectory"].cpu()
        elif model_type in ["diffusiondrive", "diffusiondrivev2"]:
            # DiffusionDrive/V2: Full diffusion-based trajectory inference
            trajectory = _get_diffusion_trajectory_for_vis(
                model=model,
                fused_features=fused_features,
                bev_feature_upscale=bev_feature_upscale,
                trajectory_query=trajectory_query,
                agents_query=agents_query,
                status_encoding=status_encoding,
                batch_size=batch_size,
                device=device,
                model_type=model_type,
            )
        else:
            trajectory = None

    return {
        "bev_semantic_map": bev_semantic_map.cpu(),
        "agent_states": agents_output["agent_states"].cpu(),
        "agent_labels": torch.sigmoid(agents_output["agent_labels"]).cpu(),
        "trajectory": trajectory,
    }


def _get_diffusion_trajectory_for_vis(
    model: nn.Module,
    fused_features: torch.Tensor,
    bev_feature_upscale: torch.Tensor,
    trajectory_query: torch.Tensor,
    agents_query: torch.Tensor,
    status_encoding: torch.Tensor,
    batch_size: int,
    device: torch.device,
    model_type: str,
) -> Optional[torch.Tensor]:
    """
    Get trajectory from DiffusionDrive/V2 models for visualization.

    This implements the full diffusion inference pipeline for trajectory prediction.

    Args:
        model: DiffusionDrive or DiffusionDriveV2 model
        fused_features: Cached BEV features (B, 512, 8, 8)
        bev_feature_upscale: BEV feature from top_down (B, C, H, W)
        trajectory_query: Trajectory query (ego_query) from transformer decoder
        agents_query: Agent query from transformer decoder
        status_encoding: Encoded status feature
        batch_size: Batch size
        device: Device
        model_type: "diffusiondrive" or "diffusiondrivev2"

    Returns:
        Trajectory tensor (B, T, 3) or None if not available
    """
    try:
        import torch.nn.functional as F
        import numpy as np

        # Import helper functions from the model's module
        if model_type == "diffusiondrive":
            from bridgesim.modelzoo.navsim.agents.diffusiondrive.modules.blocks import gen_sineembed_for_position
        else:  # diffusiondrivev2
            from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.modules.blocks import gen_sineembed_for_position

        # Get trajectory head
        traj_head = model._trajectory_head

        # Build cross_bev_feature (following the model's forward pass)
        # This is needed for the diffusion decoder's cross-attention
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = fused_features.shape[2:]  # (8, 8)

        # Downscale and build keyval (same as main forward)
        bev_feature = model._bev_downscale(fused_features).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval = keyval + model._keyval_embedding.weight[None, ...]

        # Build cross_bev_feature
        concat_cross_bev = keyval[:, :-1].permute(0, 2, 1).contiguous()
        concat_cross_bev = concat_cross_bev.view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)

        cross_bev_feature = torch.cat([concat_cross_bev, bev_feature_upscale], dim=1)
        cross_bev_feature = model.bev_proj(cross_bev_feature.flatten(-2, -1).permute(0, 2, 1))
        cross_bev_feature = cross_bev_feature.permute(0, 2, 1).contiguous()
        cross_bev_feature = cross_bev_feature.view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])

        # Run diffusion inference
        step_num = 2
        diffusion_scheduler = traj_head.diffusion_scheduler
        diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # Initialize with plan anchor + noise
        plan_anchor = traj_head.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 20, 8, 2)
        img = traj_head.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((batch_size,), device=device, dtype=torch.long) * 8
        img = diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        ego_fut_mode = img.shape[1]

        # Diffusion denoising loop
        for k in roll_timesteps:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = traj_head.denorm_odo(x_boxes)

            # Project noisy trajectory to query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = traj_head.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(batch_size, ego_fut_mode, -1)

            # Embed timesteps
            timesteps = k
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(batch_size)
            time_embed = traj_head.time_mlp(timesteps)
            time_embed = time_embed.view(batch_size, 1, -1)

            # Run diffusion decoder
            poses_reg_list, poses_cls_list = traj_head.diff_decoder(
                traj_feature, noisy_traj_points, cross_bev_feature, bev_spatial_shape,
                agents_query, trajectory_query, time_embed, status_encoding, None
            )[:2]  # Handle both DiffusionDrive (2 returns) and V2 (3 returns)

            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]

            # Update diffusion state
            x_start = poses_reg[..., :2]
            x_start = traj_head.norm_odo(x_start)
            img = diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample

        # Select best trajectory based on classification score
        # Handle different shapes between DiffusionDrive and DiffusionDriveV2:
        # - DiffusionDrive: poses_cls is (bs, 20), poses_reg is (bs, 20, 8, 3)
        # - DiffusionDriveV2: poses_cls is (bs, num_groups, 20), poses_reg is (bs, 20*num_groups, 8, 3)
        if poses_cls.dim() == 3:
            # DiffusionDriveV2: flatten poses_cls from (bs, num_groups, 20) to (bs, num_groups*20)
            # Each element in poses_reg corresponds to one mode across all groups
            poses_cls_flat = poses_cls.view(batch_size, -1)  # (bs, num_groups*20)
            mode_idx = poses_cls_flat.argmax(dim=-1)  # (bs,)
        else:
            # DiffusionDrive: poses_cls is (bs, 20)
            mode_idx = poses_cls.argmax(dim=-1)  # (bs,)

        # Expand mode_idx for gathering: (bs,) -> (bs, 1, num_poses, 3)
        mode_idx = mode_idx[:, None, None, None].expand(-1, 1, traj_head._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)  # (bs, num_poses, 3)

        return best_reg.cpu()

    except Exception as e:
        import traceback
        print(f"Warning: Could not extract diffusion trajectory for visualization: {e}")
        traceback.print_exc()
        return None


def create_default_status_feature(batch_size: int = 1) -> torch.Tensor:
    """
    Create a default status feature tensor for visualization purposes.

    The status feature encodes driving command and ego state:
    - Driving command (one-hot, 4 dims): [follow, left, right, straight]
    - Velocity (2 dims): [vx, vy] normalized
    - Acceleration (2 dims): [ax, ay] normalized

    Args:
        batch_size: Number of samples in batch

    Returns:
        Status feature tensor (B, 8)
    """
    # Default: follow lane command, zero velocity/acceleration
    status = torch.zeros(batch_size, 8, dtype=torch.float32)
    status[:, 0] = 1.0  # Follow lane command
    return status


class PairedBEVDataset(Dataset):
    """
    Dataset for extracting paired TransFuser BEV features from NAVSIM and MetaDrive.

    Both domains use the same pickle logs but different sensor_blobs paths.
    For each token, outputs:
    - NAVSIM: camera + real LiDAR from navsim_sensor_path
    - MetaDrive: camera from metadrive_sensor_path + dummy LiDAR (zeros)
    """

    def __init__(
        self,
        navsim_scene_loader: SceneLoader,
        metadrive_scene_loader: SceneLoader,
        feature_builder: TransfuserFeatureBuilder,
        config: Any,
        cache_dir: Path,
        skip_existing: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            navsim_scene_loader: SceneLoader with navsim sensor path
            metadrive_scene_loader: SceneLoader with metadrive sensor path
            feature_builder: TransFuser feature builder
            config: TransFuser config
            cache_dir: Output cache directory
            skip_existing: Skip tokens that already have cached BEV for both domains
        """
        self.navsim_scene_loader = navsim_scene_loader
        self.metadrive_scene_loader = metadrive_scene_loader
        self.feature_builder = feature_builder
        self.config = config
        self.cache_dir = cache_dir
        self.skip_existing = skip_existing

        # Get all tokens (same for both loaders since they use same pickle logs)
        self.tokens = navsim_scene_loader.tokens
        self.tokens_per_log = navsim_scene_loader.get_tokens_list_per_log()

        # Build token to log_name mapping
        self.token_to_log: Dict[str, str] = {}
        for log_name, tokens in self.tokens_per_log.items():
            for token in tokens:
                self.token_to_log[token] = log_name

        # Filter out existing if requested
        if skip_existing:
            self.tokens = self._filter_existing_tokens()

        print(f"PairedBEVDataset: {len(self.tokens)} tokens to process")

    def _filter_existing_tokens(self) -> List[str]:
        """Filter out tokens that already have cached BEV for all three domains."""
        filtered = []
        domains = ["navsim", "navsim_no_lidar", "metadrive"]

        # Cache H5 file handles to avoid repeated opens
        h5_cache: Dict[str, Dict[str, set]] = {d: {} for d in domains}

        for domain in domains:
            domain_dir = self.cache_dir / domain
            if domain_dir.exists():
                for h5_file in domain_dir.glob("*.h5"):
                    log_name = h5_file.stem
                    try:
                        with h5py.File(h5_file, "r") as f:
                            h5_cache[domain][log_name] = set(f.keys())
                    except Exception:
                        h5_cache[domain][log_name] = set()

        for token in self.tokens:
            log_name = self.token_to_log.get(token, "unknown")
            # Check if token exists in all three domain H5 files
            all_exist = all(
                token in h5_cache[domain].get(log_name, set())
                for domain in domains
            )
            if not all_exist:
                filtered.append(token)

        print(f"Filtered {len(self.tokens) - len(filtered)} existing tokens")
        return filtered

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """
        Get paired features for a single token.

        Returns:
            Dictionary with:
                - token, log_name
                - navsim_camera, navsim_lidar (real LiDAR)
                - dummy_lidar (zeros, shared for navsim_no_lidar and metadrive)
                - metadrive_camera
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

            # Dummy lidar (zeros) - shared for navsim_no_lidar and metadrive
            dummy_lidar = create_dummy_lidar_bev(self.config)

            return {
                "token": token,
                "log_name": log_name,
                "navsim_camera": navsim_camera,
                "navsim_lidar": navsim_lidar,
                "metadrive_camera": metadrive_camera,
                "dummy_lidar": dummy_lidar,
            }

        except Exception as e:
            print(f"Error loading token {token}: {e}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Collate function that filters None samples and stacks tensors.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    tokens = [b["token"] for b in batch]
    log_names = [b["log_name"] for b in batch]

    navsim_cameras = torch.stack([b["navsim_camera"] for b in batch])
    navsim_lidars = torch.stack([b["navsim_lidar"] for b in batch])
    metadrive_cameras = torch.stack([b["metadrive_camera"] for b in batch])
    dummy_lidars = torch.stack([b["dummy_lidar"] for b in batch])

    return {
        "tokens": tokens,
        "log_names": log_names,
        "navsim_cameras": navsim_cameras,
        "navsim_lidars": navsim_lidars,
        "metadrive_cameras": metadrive_cameras,
        "dummy_lidars": dummy_lidars,
    }


def save_bev_cache(
    fused_features: torch.Tensor,
    token: str,
    log_name: str,
    domain: str,
    cache_dir: Path,
) -> None:
    """
    Save BEV fused_features to cache in H5 format.

    Each log is stored as a single H5 file with tokens as datasets.
    Cache structure: cache_dir/<domain>/<log_name>.h5
        - <token>: dataset containing fused_features

    Args:
        fused_features: BEV tensor (512, 8, 8)
        token: Scene token
        log_name: Log name
        domain: "navsim", "navsim_no_lidar", or "metadrive"
        cache_dir: Root cache directory
    """
    out_dir = cache_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"{log_name}.h5"

    # Convert tensor to numpy
    features_np = fused_features.cpu().numpy()

    # Open H5 file in append mode (creates if doesn't exist)
    with h5py.File(h5_path, "a") as f:
        # Skip if token already exists
        if token in f:
            return
        # Create dataset for this token
        f.create_dataset(
            token,
            data=features_np,
            compression="gzip",
            compression_opts=4,
        )


def build_scene_loaders(
    data_path: Path,
    navsim_sensor_path: Path,
    metadrive_sensor_path: Path,
    scene_filter: SceneFilter,
    navsim_sensor_config: SensorConfig,
    metadrive_sensor_config: SensorConfig,
) -> Tuple[SceneLoader, SceneLoader]:
    """
    Build SceneLoaders for both NAVSIM and MetaDrive domains.

    Args:
        data_path: Path to log pickles
        navsim_sensor_path: Path to NAVSIM sensor blobs
        metadrive_sensor_path: Path to MetaDrive sensor blobs
        scene_filter: Scene filter config
        navsim_sensor_config: Sensor config for NAVSIM (with lidar)
        metadrive_sensor_config: Sensor config for MetaDrive (without lidar)

    Returns:
        Tuple of (navsim_scene_loader, metadrive_scene_loader)
    """
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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup distributed if requested
    if args.distributed or "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Create cache directory
    if rank == 0:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # Load model
    model, config = load_model_and_config(
        args.checkpoint, device, args.model_type, args.plan_anchor_path
    )

    # Create feature builder
    feature_builder = TransfuserFeatureBuilder(config)

    # Setup scene filter
    if args.split_yaml is not None:
        print(f"Loading scene filter from YAML: {args.split_yaml}")
        scene_filter = load_scene_filter_from_yaml(
            args.split_yaml,
            log_names_override=args.log_names,
            max_scenes_override=args.max_scenes,
        )
    else:
        # Fallback to default SceneFilter with CLI args
        scene_filter = SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            has_route=True,
            log_names=args.log_names,
            max_scenes=args.max_scenes,
        )

    # Setup sensor config for NAVSIM (with lidar)
    navsim_sensor_config = SensorConfig(
        cam_f0=[3],  # Only load at current frame (index 3)
        cam_l0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=[3],
    )

    # Setup sensor config for MetaDrive (cameras only, no lidar)
    metadrive_sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,  # MetaDrive doesn't have lidar
    )

    # Validate MetaDrive sensor path
    if args.metadrive_sensor_path is None:
        raise ValueError("--metadrive-sensor-path is required")

    # Build scene loaders
    print("Building scene loaders...")
    navsim_loader, metadrive_loader = build_scene_loaders(
        data_path=args.navsim_log_path,
        navsim_sensor_path=args.navsim_sensor_path,
        metadrive_sensor_path=args.metadrive_sensor_path,
        scene_filter=scene_filter,
        navsim_sensor_config=navsim_sensor_config,
        metadrive_sensor_config=metadrive_sensor_config,
    )

    print(f"Total tokens: {len(navsim_loader.tokens)}")

    # Create dataset
    dataset = PairedBEVDataset(
        navsim_scene_loader=navsim_loader,
        metadrive_scene_loader=metadrive_loader,
        feature_builder=feature_builder,
        config=config,
        cache_dir=args.cache_dir,
        skip_existing=args.skip_existing,
    )

    # Distributed sharding
    if world_size > 1:
        indices = list(range(len(dataset)))
        indices = indices[rank::world_size]
        dataset.tokens = [dataset.tokens[i] for i in indices]
        print(f"Rank {rank}: processing {len(dataset.tokens)} tokens")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Setup visualization
    vis_output_dir = args.vis_output_dir or (args.cache_dir / "visualizations")
    vis_count = 0

    # Process batches
    progress = tqdm(dataloader, desc="Precomputing BEV features", disable=(rank != 0))
    for batch in progress:
        if batch is None:
            continue

        tokens = batch["tokens"]
        log_names = batch["log_names"]

        # Extract NAVSIM BEV features (with real LiDAR)
        navsim_fused = extract_fused_features(
            model,
            batch["navsim_cameras"],
            batch["navsim_lidars"],
            device,
        )

        # Extract NAVSIM BEV features with dummy LiDAR (isolates LiDAR effect)
        navsim_no_lidar_fused = extract_fused_features(
            model,
            batch["navsim_cameras"],
            batch["dummy_lidars"],
            device,
        )

        # Extract MetaDrive BEV features (with dummy LiDAR)
        metadrive_fused = extract_fused_features(
            model,
            batch["metadrive_cameras"],
            batch["dummy_lidars"],
            device,
        )

        # Save to cache
        batch_size = len(tokens)
        for i in range(batch_size):
            token = tokens[i]
            log_name = log_names[i]

            # Save NAVSIM BEV (with real LiDAR)
            save_bev_cache(
                navsim_fused[i],
                token,
                log_name,
                "navsim",
                args.cache_dir,
            )

            # Save NAVSIM BEV with dummy LiDAR
            save_bev_cache(
                navsim_no_lidar_fused[i],
                token,
                log_name,
                "navsim_no_lidar",
                args.cache_dir,
            )

            # Save MetaDrive BEV
            save_bev_cache(
                metadrive_fused[i],
                token,
                log_name,
                "metadrive",
                args.cache_dir,
            )

        # Generate visualizations if enabled
        if args.visualize and vis_count < args.vis_num_samples:
            for i in range(min(batch_size, args.vis_num_samples - vis_count)):
                token = tokens[i]

                # Get predictions from cached features if requested
                if args.vis_include_predictions:
                    status_feature = create_default_status_feature(1)

                    # NAVSIM predictions
                    navsim_preds = get_predictions_from_fused_features(
                        model, navsim_fused[i:i+1], status_feature, device,
                        model_type=args.model_type,
                    )
                    # NAVSIM no-lidar predictions
                    navsim_no_lidar_preds = get_predictions_from_fused_features(
                        model, navsim_no_lidar_fused[i:i+1], status_feature, device,
                        model_type=args.model_type,
                    )
                    # MetaDrive predictions
                    metadrive_preds = get_predictions_from_fused_features(
                        model, metadrive_fused[i:i+1], status_feature, device,
                        model_type=args.model_type,
                    )
                else:
                    navsim_preds = navsim_no_lidar_preds = metadrive_preds = None

                # Helper to safely extract trajectory (handles None case)
                def get_trajectory(preds):
                    if preds is None or preds["trajectory"] is None:
                        return None
                    return preds["trajectory"][0].numpy()

                # Create visualizations for each domain
                navsim_vis = create_bev_visualization(
                    camera_image=batch["navsim_cameras"][i].numpy(),
                    lidar_feature=batch["navsim_lidars"][i].numpy(),
                    bev_semantic_map=navsim_preds["bev_semantic_map"][0].numpy() if navsim_preds else None,
                    agent_states=navsim_preds["agent_states"][0].numpy() if navsim_preds else None,
                    agent_labels=navsim_preds["agent_labels"][0].numpy() if navsim_preds else None,
                    trajectory=get_trajectory(navsim_preds),
                    config=config,
                    title="NAVSIM (with LiDAR)",
                )

                navsim_no_lidar_vis = create_bev_visualization(
                    camera_image=batch["navsim_cameras"][i].numpy(),
                    lidar_feature=batch["dummy_lidars"][i].numpy(),
                    bev_semantic_map=navsim_no_lidar_preds["bev_semantic_map"][0].numpy() if navsim_no_lidar_preds else None,
                    agent_states=navsim_no_lidar_preds["agent_states"][0].numpy() if navsim_no_lidar_preds else None,
                    agent_labels=navsim_no_lidar_preds["agent_labels"][0].numpy() if navsim_no_lidar_preds else None,
                    trajectory=get_trajectory(navsim_no_lidar_preds),
                    config=config,
                    title="NAVSIM (no LiDAR)",
                )

                metadrive_vis = create_bev_visualization(
                    camera_image=batch["metadrive_cameras"][i].numpy(),
                    lidar_feature=batch["dummy_lidars"][i].numpy(),
                    bev_semantic_map=metadrive_preds["bev_semantic_map"][0].numpy() if metadrive_preds else None,
                    agent_states=metadrive_preds["agent_states"][0].numpy() if metadrive_preds else None,
                    agent_labels=metadrive_preds["agent_labels"][0].numpy() if metadrive_preds else None,
                    trajectory=get_trajectory(metadrive_preds),
                    config=config,
                    title="MetaDrive",
                )

                # Save individual visualizations
                save_visualization(navsim_vis, vis_output_dir / "navsim", token)
                save_visualization(navsim_no_lidar_vis, vis_output_dir / "navsim_no_lidar", token)
                save_visualization(metadrive_vis, vis_output_dir / "metadrive", token)

                # Create domain comparison if requested
                if args.vis_compare_domains:
                    comparison = create_domain_comparison_visualization(
                        navsim_vis, navsim_no_lidar_vis, metadrive_vis, token
                    )
                    save_visualization(comparison, vis_output_dir / "comparison", token)

                vis_count += 1

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    print("Done!")


if __name__ == "__main__":
    main()

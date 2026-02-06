#!/usr/bin/env python3
"""
Evaluate Image Feature Flow matching on validation data.

Workflow:
- Load validation scenes from NAVSIM log pickles.
- Extract image stem features and BEV features for 4 conditions:
  1. navsim: NavSim camera + real LiDAR
  2. navsim_no_lidar: NavSim camera + dummy LiDAR
  3. metadrive: MetaDrive camera + dummy LiDAR
  4. generated: Flow-sampled image features from metadrive, then continued through backbone
- Run model predictions on each BEV type.
- Compute metrics:
  - Image feature space: MSE, cosine similarity (generated vs navsim stem)
  - BEV feature space: MSE, cosine similarity (generated vs navsim BEV)
  - Downstream: ADE at 1s, 2s, 4s horizons
- Save visualizations and JSON summary.

Supports: TransFuser, DiffusionDrive, DiffusionDriveV2 (same backbone, different trajectory heads).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
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
    create_dummy_lidar_bev,
    get_predictions_from_fused_features,
    load_model_and_config,
    load_scene_filter_from_yaml,
    set_seed,
)
from train_image_feature_flow import (
    IMAGE_FEATURE_CHANNELS,
    IMAGE_FEATURE_HEIGHT,
    IMAGE_FEATURE_WIDTH,
    ImageFeatureFlowMatchingLit,
)


# ============================================================================
# Helper Functions for Feature Extraction
# ============================================================================


def extract_stem_features_only(
    model: nn.Module,
    camera_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract only the stem image features from camera input.

    This runs the image encoder stem layer only (before fusion blocks).

    Args:
        model: TransFuser model (has _backbone attribute)
        camera_feature: Camera input tensor (B, 3, 256, 1024)
        device: Device to run on

    Returns:
        image_features tensor after stem layer (B, 64, 128, 512)
    """
    camera_feature = camera_feature.to(device)

    with torch.no_grad():
        backbone = model._backbone
        image_features = camera_feature

        image_layers = iter(backbone.image_encoder.items())
        if len(backbone.image_encoder.return_layers) > 4:
            image_features = backbone.forward_layer_block(
                image_layers,
                backbone.image_encoder.return_layers,
                image_features
            )

    return image_features


def continue_from_stem_features(
    model: nn.Module,
    image_stem_features: torch.Tensor,
    lidar_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Continue backbone forward pass from post-stem image features.

    This processes lidar through its stem, then runs 4 fusion blocks
    with the provided image stem features.

    Args:
        model: TransFuser model
        image_stem_features: Image features after stem (B, 64, 128, 512)
        lidar_feature: LiDAR BEV input (B, 1, 256, 256)
        device: Device to run on

    Returns:
        fused_features: (B, 512, 8, 8) for trajectory prediction head
    """
    image_stem_features = image_stem_features.to(device)
    lidar_feature = lidar_feature.to(device)

    with torch.no_grad():
        backbone = model._backbone
        config = backbone.config

        # Handle latent lidar if needed
        if config.latent and lidar_feature is None:
            batch_size = image_stem_features.shape[0]
            lidar_feature = backbone.lidar_latent.repeat(batch_size, 1, 1, 1)

        image_features = image_stem_features
        lidar_features = lidar_feature

        # Process lidar through stem
        lidar_layers = iter(backbone.lidar_encoder.items())
        if len(backbone.lidar_encoder.return_layers) > 4:
            lidar_features = backbone.forward_layer_block(
                lidar_layers, backbone.lidar_encoder.return_layers, lidar_features
            )

        # Create fresh image encoder iterator and skip stem
        image_layers = iter(backbone.image_encoder.items())
        if len(backbone.image_encoder.return_layers) > 4:
            # Advance iterator past stem layer (but don't process - we already have stem features)
            for name, module in image_layers:
                if name in backbone.image_encoder.return_layers:
                    break

        # Run 4 fusion blocks
        for i in range(4):
            image_features = backbone.forward_layer_block(
                image_layers, backbone.image_encoder.return_layers, image_features
            )
            lidar_features = backbone.forward_layer_block(
                lidar_layers, backbone.lidar_encoder.return_layers, lidar_features
            )
            image_features, lidar_features = backbone.fuse_features(
                image_features, lidar_features, i
            )

        # Get fused features (same logic as backbone.forward)
        if config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features_pooled = backbone.global_pool_img(image_features)
            image_features_pooled = torch.flatten(image_features_pooled, 1)
            lidar_features_pooled = backbone.global_pool_lidar(lidar_features)
            lidar_features_pooled = torch.flatten(lidar_features_pooled, 1)

            if config.add_features:
                lidar_features_pooled = backbone.lidar_to_img_features_end(lidar_features_pooled)
                fused_features = image_features_pooled + lidar_features_pooled
            else:
                fused_features = torch.cat((image_features_pooled, lidar_features_pooled), dim=1)

    return fused_features


def extract_fused_features(
    model: nn.Module,
    camera_feature: torch.Tensor,
    lidar_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract fused BEV features from model backbone.

    Args:
        model: TransFuser model
        camera_feature: Camera input (B, 3, 256, 1024)
        lidar_feature: LiDAR BEV (B, 1, 256, 256)
        device: Device to run on

    Returns:
        fused_features: (B, 512, 8, 8) or flattened depending on config
    """
    camera_feature = camera_feature.to(device)
    lidar_feature = lidar_feature.to(device)

    with torch.no_grad():
        _, fused_features, _ = model._backbone(camera_feature, lidar_feature)

    return fused_features


def extract_all_features(
    model: nn.Module,
    camera_feature: torch.Tensor,
    lidar_feature: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract both stem features and fused BEV features in sequence.

    We cannot use hooks reliably across different model types, so we:
    1. First extract stem features (stem only)
    2. Then run full forward pass to get fused features

    Args:
        model: TransFuser model
        camera_feature: Camera input (B, 3, 256, 1024)
        lidar_feature: LiDAR BEV (B, 1, 256, 256)
        device: Device to run on

    Returns:
        Tuple of (stem_features, fused_features)
        - stem_features: (B, 64, 128, 512)
        - fused_features: (B, 512, 8, 8)
    """
    # Extract stem features first
    stem_features = extract_stem_features_only(model, camera_feature, device)

    # Then extract fused features (full forward pass)
    fused_features = extract_fused_features(model, camera_feature, lidar_feature, device)

    return stem_features, fused_features


# ============================================================================
# Flow Model Loading
# ============================================================================


def load_image_flow_model(
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[ImageFeatureFlowMatchingLit, nn.Module]:
    """
    Load image feature flow matching model from checkpoint.

    Args:
        ckpt_path: Path to Lightning checkpoint
        device: Device to load on
        use_ema: Whether to use EMA weights for sampling

    Returns:
        Tuple of (lit_model, sampler_model)
    """
    import pathlib
    # PyTorch 2.6+ requires explicit allowlisting of pathlib types for safe loading
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

    # Use strict=False because the checkpoint may contain backbone weights
    # (saved during training when backbone was loaded), but we don't need them
    # since we'll use the backbone from the main model for feature extraction.
    lit_model = ImageFeatureFlowMatchingLit.load_from_checkpoint(
        str(ckpt_path), map_location=device, strict=False
    )
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


# ============================================================================
# Metric Functions
# ============================================================================


def compute_image_feature_metrics(
    generated_features: torch.Tensor,
    target_features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute image feature-space metrics.

    Args:
        generated_features: (B, C, H, W) generated stem features
        target_features: (B, C, H, W) target navsim stem features

    Returns:
        Dictionary with:
            - mse: Mean squared error (per sample)
            - cosine_mean: Mean cosine similarity (per sample)
            - cosine_per_position: (H*W,) cosine sim per spatial position (averaged over batch)
    """
    B, C, H, W = generated_features.shape

    # MSE per sample
    mse = ((generated_features - target_features) ** 2).flatten(1).mean(dim=1)

    # Cosine similarity per spatial position
    gen_flat = generated_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
    tgt_flat = target_features.flatten(2).transpose(1, 2)  # (B, H*W, C)

    gen_norm = torch.nn.functional.normalize(gen_flat, p=2, dim=-1)
    tgt_norm = torch.nn.functional.normalize(tgt_flat, p=2, dim=-1)

    cos_sim = (gen_norm * tgt_norm).sum(dim=-1)  # (B, H*W)
    cosine_mean = cos_sim.mean(dim=-1)  # (B,)
    cosine_per_position = cos_sim.mean(dim=0)  # (H*W,)

    return {
        "mse": mse,
        "cosine_mean": cosine_mean,
        "cosine_per_position": cosine_per_position,
    }


def compute_bev_metrics(
    generated_bev: torch.Tensor,
    target_bev: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute BEV feature-space metrics.

    Args:
        generated_bev: (B, C, H, W) or (B, C) generated BEV features
        target_bev: (B, C, H, W) or (B, C) target navsim BEV features

    Returns:
        Dictionary with mse and cosine_mean (per sample)
    """
    # MSE per sample
    mse = ((generated_bev - target_bev) ** 2).flatten(1).mean(dim=1)

    # Cosine similarity
    gen_flat = generated_bev.flatten(1)  # (B, C*H*W) or (B, C)
    tgt_flat = target_bev.flatten(1)

    cosine = torch.nn.functional.cosine_similarity(gen_flat, tgt_flat, dim=1)

    return {
        "mse": mse,
        "cosine_mean": cosine,
    }


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


# ============================================================================
# Visualization Functions
# ============================================================================


def feature_to_image(
    features: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """
    Convert feature map to displayable image.

    Args:
        features: (C, H, W) feature tensor
        method: "mean" (average channels), "max" (max across channels), "pca" (first 3 PCA components)

    Returns:
        (H, W, 3) uint8 image
    """
    if method == "mean":
        # Average across channels
        img = features.mean(axis=0)
    elif method == "max":
        # Max across channels
        img = features.max(axis=0)
    else:
        # Default to mean
        img = features.mean(axis=0)

    # Normalize to [0, 255]
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)

    img = (img * 255).astype(np.uint8)

    # Apply colormap for visualization
    img_colored = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)

    return img_colored


def cosine_map_to_image(
    cosine_map: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert cosine similarity map to displayable image.

    Args:
        cosine_map: (H*W,) cosine similarities in [-1, 1]
        height: Height of feature map
        width: Width of feature map

    Returns:
        (H, W, 3) uint8 image with colormap
    """
    # Reshape to spatial
    cos_2d = cosine_map.reshape(height, width)

    # Normalize from [-1, 1] to [0, 255]
    # Typically cosine is positive for similar features
    cos_normalized = (cos_2d + 1) / 2  # [0, 1]
    cos_uint8 = (cos_normalized * 255).astype(np.uint8)

    # Apply colormap (red=low similarity, green=high similarity)
    img_colored = cv2.applyColorMap(cos_uint8, cv2.COLORMAP_JET)
    img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)

    return img_colored


def camera_to_image(camera: np.ndarray) -> np.ndarray:
    """
    Convert camera tensor to displayable image.

    Args:
        camera: (3, H, W) normalized camera tensor

    Returns:
        (H, W, 3) uint8 RGB image
    """
    if camera.shape[0] == 3:
        camera = camera.transpose(1, 2, 0)  # CHW -> HWC

    # Denormalize if needed (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if camera.max() <= 1.0:
        camera = camera * std + mean
        camera = np.clip(camera, 0, 1)
        camera = (camera * 255).astype(np.uint8)
    else:
        camera = camera.astype(np.uint8)

    return camera


def create_image_feature_visualization(
    navsim_camera: np.ndarray,
    metadrive_camera: np.ndarray,
    navsim_stem: np.ndarray,
    metadrive_stem: np.ndarray,
    generated_stem: np.ndarray,
    cosine_per_position: np.ndarray,
    title: str = "",
) -> np.ndarray:
    """
    Create comprehensive visualization for image features.

    Layout:
    +------------------+------------------+
    |  Title                              |
    +------------------+------------------+
    |  NavSim Camera   | MetaDrive Camera |
    +------------------+------------------+
    |  NavSim Stem     | MetaDrive Stem   |
    +------------------+------------------+
    |  Generated Stem  | Cosine Sim Map   |
    +------------------+------------------+

    Args:
        navsim_camera: (3, H, W) NavSim camera image
        metadrive_camera: (3, H, W) MetaDrive camera image
        navsim_stem: (C, H, W) NavSim stem features
        metadrive_stem: (C, H, W) MetaDrive stem features
        generated_stem: (C, H, W) Flow-transformed stem features
        cosine_per_position: (H*W,) per-pixel cosine similarity
        title: Title for the visualization

    Returns:
        Combined visualization image (H, W, 3) as uint8
    """
    # Convert cameras to images
    nav_cam_img = camera_to_image(navsim_camera)
    md_cam_img = camera_to_image(metadrive_camera)

    # Convert features to images
    nav_stem_img = feature_to_image(navsim_stem)
    md_stem_img = feature_to_image(metadrive_stem)
    gen_stem_img = feature_to_image(generated_stem)

    # Convert cosine map to image
    _, feat_h, feat_w = navsim_stem.shape
    cos_img = cosine_map_to_image(cosine_per_position, feat_h, feat_w)

    # Resize all to common width for layout
    target_width = 512
    cam_h, cam_w = nav_cam_img.shape[:2]
    cam_target_h = int(target_width * cam_h / cam_w)

    feat_target_h = int(target_width * feat_h / feat_w)

    # Resize images
    nav_cam_resized = cv2.resize(nav_cam_img, (target_width, cam_target_h))
    md_cam_resized = cv2.resize(md_cam_img, (target_width, cam_target_h))
    nav_stem_resized = cv2.resize(nav_stem_img, (target_width, feat_target_h))
    md_stem_resized = cv2.resize(md_stem_img, (target_width, feat_target_h))
    gen_stem_resized = cv2.resize(gen_stem_img, (target_width, feat_target_h))
    cos_resized = cv2.resize(cos_img, (target_width, feat_target_h))

    # Create rows
    row1 = np.concatenate([nav_cam_resized, md_cam_resized], axis=1)
    row2 = np.concatenate([nav_stem_resized, md_stem_resized], axis=1)
    row3 = np.concatenate([gen_stem_resized, cos_resized], axis=1)

    # Add labels
    def add_label(img: np.ndarray, label: str) -> np.ndarray:
        """Add text label to image."""
        img_copy = img.copy()
        cv2.putText(
            img_copy, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return img_copy

    row1_labeled = np.concatenate([
        add_label(nav_cam_resized, "NavSim Camera"),
        add_label(md_cam_resized, "MetaDrive Camera")
    ], axis=1)

    row2_labeled = np.concatenate([
        add_label(nav_stem_resized, "NavSim Stem"),
        add_label(md_stem_resized, "MetaDrive Stem")
    ], axis=1)

    row3_labeled = np.concatenate([
        add_label(gen_stem_resized, "Generated Stem"),
        add_label(cos_resized, "Cosine Similarity")
    ], axis=1)

    # Create title bar
    title_height = 40
    title_bar = np.ones((title_height, row1_labeled.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(
        title_bar, title, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )

    # Combine all rows
    combined = np.concatenate([title_bar, row1_labeled, row2_labeled, row3_labeled], axis=0)

    return combined


def save_visualization(
    image: np.ndarray,
    output_dir: Path,
    token: str,
    suffix: str = "",
) -> None:
    """
    Save visualization image to disk.

    Args:
        image: RGB image (H, W, 3)
        output_dir: Output directory
        token: Scene token for filename
        suffix: Optional suffix for filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{token}{suffix}.png"
    filepath = output_dir / filename

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filepath), image_bgr)


# ============================================================================
# Dataset
# ============================================================================


class ImageFeatureValidationDataset(Dataset):
    """
    Dataset for evaluating image feature flow matching on validation scenes.

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

        print(f"ImageFeatureValidationDataset: {len(self.tokens)} tokens to process")

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


# ============================================================================
# Scene Loader Builder
# ============================================================================


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


# ============================================================================
# Argument Parser
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Image Feature Flow matching on validation data."
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

    # Backbone model
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

    # Image flow model
    parser.add_argument(
        "--flow-ckpt",
        type=Path,
        required=True,
        help="Path to image feature flow model checkpoint.",
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
        default=Path("image_flow_eval"),
        help="Output directory for results.",
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
        help="Enable visualization of features and predictions.",
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


# ============================================================================
# Main Evaluation Loop
# ============================================================================


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

    # Load backbone model
    print("Loading backbone model...")
    model, config = load_model_and_config(
        args.checkpoint, device, args.model_type, args.plan_anchor_path
    )
    feature_builder = TransfuserFeatureBuilder(config)

    # Load image flow model
    print("Loading image flow model...")
    _, flow_sampler = load_image_flow_model(args.flow_ckpt, device, use_ema=args.use_ema)

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
    dataset = ImageFeatureValidationDataset(
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

    # Tracking metrics
    records: List[Dict] = []

    # Image feature metrics
    all_img_mse = []
    all_img_cosine = []

    # BEV metrics
    all_bev_mse = []
    all_bev_cosine = []

    # Trajectory ADE at different horizons for each domain
    all_ade_navsim = {"1s": [], "2s": [], "4s": []}
    all_ade_navsim_no_lidar = {"1s": [], "2s": [], "4s": []}
    all_ade_metadrive = {"1s": [], "2s": [], "4s": []}
    all_ade_generated = {"1s": [], "2s": [], "4s": []}

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

        # === Domain 1: NAVSIM (with LiDAR) ===
        navsim_stem, navsim_bev = extract_all_features(
            model, navsim_cameras, navsim_lidars, device
        )

        # === Domain 2: NAVSIM no-LiDAR ===
        navsim_no_lidar_stem, navsim_no_lidar_bev = extract_all_features(
            model, navsim_cameras, dummy_lidars, device
        )

        # === Domain 3: MetaDrive (extract stem only first) ===
        metadrive_stem = extract_stem_features_only(model, metadrive_cameras, device)

        # === Domain 4: Generated (flow-sampled from metadrive) ===
        generated_stem = flow_sampler.sample(metadrive_stem, sample_steps=args.sample_steps)

        # Continue metadrive and generated through backbone to get BEV
        metadrive_bev = continue_from_stem_features(model, metadrive_stem, dummy_lidars, device)
        generated_bev = continue_from_stem_features(model, generated_stem, dummy_lidars, device)

        # === Compute image feature metrics (generated vs navsim) ===
        img_metrics = compute_image_feature_metrics(generated_stem, navsim_stem)

        # === Compute BEV metrics (generated vs navsim) ===
        bev_metrics = compute_bev_metrics(generated_bev, navsim_bev)

        # === Get trajectory predictions ===
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

        # === Process each sample in batch ===
        for i in range(batch_size):
            token = tokens[i]
            log_name = log_names[i]
            gt_traj = gt_trajectories[i].cpu()

            # Image feature metrics
            img_mse_i = img_metrics["mse"][i].item()
            img_cos_i = img_metrics["cosine_mean"][i].item()
            all_img_mse.append(img_mse_i)
            all_img_cosine.append(img_cos_i)

            # BEV metrics
            bev_mse_i = bev_metrics["mse"][i].item()
            bev_cos_i = bev_metrics["cosine_mean"][i].item()
            all_bev_mse.append(bev_mse_i)
            all_bev_cosine.append(bev_cos_i)

            # Helper to get trajectory safely
            def get_traj(preds, idx):
                if preds is None or preds["trajectory"] is None:
                    return None
                return preds["trajectory"][idx]

            navsim_traj = get_traj(navsim_preds, i)
            navsim_no_lidar_traj = get_traj(navsim_no_lidar_preds, i)
            metadrive_traj = get_traj(metadrive_preds, i)
            generated_traj = get_traj(generated_preds, i)

            # Compute ADE at multiple horizons
            ade_navsim = compute_trajectory_l2_loss(navsim_traj, gt_traj) if navsim_traj is not None else None
            ade_navsim_no_lidar = compute_trajectory_l2_loss(navsim_no_lidar_traj, gt_traj) if navsim_no_lidar_traj is not None else None
            ade_metadrive = compute_trajectory_l2_loss(metadrive_traj, gt_traj) if metadrive_traj is not None else None
            ade_generated = compute_trajectory_l2_loss(generated_traj, gt_traj) if generated_traj is not None else None

            # Track ADE metrics
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

            # Record
            record = {
                "token": token,
                "log_name": log_name,
                "img_mse": img_mse_i,
                "img_cosine": img_cos_i,
                "bev_mse": bev_mse_i,
                "bev_cosine": bev_cos_i,
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
            }
            records.append(record)

            # === Visualization ===
            if args.visualize and vis_count < args.vis_num_samples:
                vis = create_image_feature_visualization(
                    navsim_camera=navsim_cameras[i].cpu().numpy(),
                    metadrive_camera=metadrive_cameras[i].cpu().numpy(),
                    navsim_stem=navsim_stem[i].cpu().numpy(),
                    metadrive_stem=metadrive_stem[i].cpu().numpy(),
                    generated_stem=generated_stem[i].cpu().numpy(),
                    cosine_per_position=img_metrics["cosine_per_position"].cpu().numpy(),
                    title=f"Token: {token[:20]}... | ImgCos: {img_cos_i:.3f} | BEVCos: {bev_cos_i:.3f}",
                )
                save_visualization(vis, vis_output_dir, token)
                vis_count += 1

        # Update progress bar
        if all_img_cosine:
            progress.set_postfix({
                "img_cos": f"{np.mean(all_img_cosine):.3f}",
                "bev_cos": f"{np.mean(all_bev_cosine):.3f}",
                "ade4s_gen": f"{np.mean(all_ade_generated['4s']):.3f}" if all_ade_generated["4s"] else "N/A",
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
        "image_feature_metrics": {
            "mse": compute_stats(all_img_mse),
            "cosine_mean": compute_stats(all_img_cosine),
        },
        "bev_metrics": {
            "mse": compute_stats(all_bev_mse),
            "cosine_mean": compute_stats(all_bev_cosine),
        },
        "trajectory_ade": {
            "navsim": {
                "1s": compute_stats(all_ade_navsim["1s"]),
                "2s": compute_stats(all_ade_navsim["2s"]),
                "4s": compute_stats(all_ade_navsim["4s"]),
            },
            "navsim_no_lidar": {
                "1s": compute_stats(all_ade_navsim_no_lidar["1s"]),
                "2s": compute_stats(all_ade_navsim_no_lidar["2s"]),
                "4s": compute_stats(all_ade_navsim_no_lidar["4s"]),
            },
            "metadrive": {
                "1s": compute_stats(all_ade_metadrive["1s"]),
                "2s": compute_stats(all_ade_metadrive["2s"]),
                "4s": compute_stats(all_ade_metadrive["4s"]),
            },
            "generated": {
                "1s": compute_stats(all_ade_generated["1s"]),
                "2s": compute_stats(all_ade_generated["2s"]),
                "4s": compute_stats(all_ade_generated["4s"]),
            },
        },
        "records": records,
    }

    # Save summary
    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(records)}")

    def fmt_stats(stats: Dict) -> str:
        """Format stats as string."""
        if stats["mean"] is None:
            return "N/A"
        return f"{stats['mean']:.4f} +/- {stats['std']:.4f}"

    print(f"\nImage Feature Metrics (generated vs navsim):")
    print(f"  MSE:    {fmt_stats(summary['image_feature_metrics']['mse'])}")
    print(f"  Cosine: {fmt_stats(summary['image_feature_metrics']['cosine_mean'])}")

    print(f"\nBEV Feature Metrics (generated vs navsim):")
    print(f"  MSE:    {fmt_stats(summary['bev_metrics']['mse'])}")
    print(f"  Cosine: {fmt_stats(summary['bev_metrics']['cosine_mean'])}")

    print(f"\nADE (Average Displacement Error) in meters (lower is better):")
    print(f"{'Domain':<25} {'ADE@1s':<20} {'ADE@2s':<20} {'ADE@4s':<20}")
    print(f"{'-'*85}")

    for domain in ["navsim", "navsim_no_lidar", "generated", "metadrive"]:
        domain_display = {
            "navsim": "NavSim (with LiDAR)",
            "navsim_no_lidar": "NavSim (no LiDAR)",
            "generated": "Generated (flow)",
            "metadrive": "MetaDrive",
        }[domain]
        ade = summary["trajectory_ade"][domain]
        print(f"{domain_display:<25} {fmt_stats(ade['1s']):<20} {fmt_stats(ade['2s']):<20} {fmt_stats(ade['4s']):<20}")

    print(f"\nResults saved to: {summary_path}")
    if args.visualize:
        print(f"Visualizations saved to: {vis_output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

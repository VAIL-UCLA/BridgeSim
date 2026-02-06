#!/usr/bin/env python3
"""
Precompute TransFuser image features (after stem layer) and cache them as .pt files.

This script extracts image features from the TransFuser backbone's stem layer
(line 174 of transfuser_backbone.py) for both NAVSIM and MetaDrive domains.

The stem image features are the output of the image encoder's stem layer,
before the 4 fusion blocks with LiDAR.

Supported models:
- transfuser: Original TransFuser model
- diffusiondrive: DiffusionDrive model (TransFuser backbone)
- diffusiondrivev2: DiffusionDriveV2 model (TransFuser backbone)

All three models share the same TransFuser backbone architecture.

Cache layout: cache_dir/<domain>/<log_name>/<token>.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add navsim to path
SCRIPT_DIR = Path(__file__).resolve().parent
NAVSIM_ROOT = SCRIPT_DIR.parent / "navsim"
sys.path.insert(0, str(NAVSIM_ROOT))

from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from bridgesim.modelzoo.navsim.common.dataclasses import AgentInput, SceneFilter, SensorConfig
from bridgesim.modelzoo.navsim.common.dataloader import SceneLoader

# Model type to import mapping
MODEL_TYPES = ["transfuser", "diffusiondrive", "diffusiondrivev2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute TransFuser stem image features for NAVSIM/MetaDrive."
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
        help="Path to plan anchor file (.npy) for DiffusionDrive/V2 models.",
    )

    # Output
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Output directory for cached image features.",
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
        help="Skip tokens that already have cached features.",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of image features.",
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
    """Load SceneFilter from a YAML file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    scene_filter_kwargs = {
        k: v for k, v in config.items() if not k.startswith("_")
    }

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
    """Load model from checkpoint based on model type."""
    print(f"Loading {model_type} model...")

    if model_type == "transfuser":
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_model import TransfuserModel
        from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
        model = TransfuserModel(trajectory_sampling, config)
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    elif model_type == "diffusiondrive":
        from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
        from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

        config = TransfuserConfig()
        if plan_anchor_path is not None:
            config.plan_anchor_path = str(plan_anchor_path)
        model = V2TransfuserModel(config)
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    elif model_type == "diffusiondrivev2":
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel
        from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig

        config = TransfuserConfig()
        if plan_anchor_path is not None:
            config.plan_anchor_path = str(plan_anchor_path)
        model = V2TransfuserModel(config)
        prefixes_to_strip = ["agent._transfuser_model.", "_transfuser_model."]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    clean_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in prefixes_to_strip:
            new_key = new_key.replace(prefix, "")
        clean_sd[new_key] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}..." if len(missing) > 5 else f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    print(f"{model_type} model loaded successfully.")
    return model, config


def extract_stem_image_features(
    model: nn.Module,
    camera_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract image features after the stem layer (line 174 of transfuser_backbone.py).

    This extracts the image features after only the stem layer of the image encoder,
    before any fusion with LiDAR features.

    Args:
        model: TransFuser model
        camera_feature: Camera input tensor (B, 3, 256, 1024)
        device: Device to run on

    Returns:
        image_features tensor after stem layer
    """
    camera_feature = camera_feature.to(device)

    with torch.no_grad():
        backbone = model._backbone
        image_features = camera_feature

        # Run through stem layer only (if the encoder has one)
        # This corresponds to line 173-174 in transfuser_backbone.py
        image_layers = iter(backbone.image_encoder.items())
        if len(backbone.image_encoder.return_layers) > 4:
            image_features = backbone.forward_layer_block(
                image_layers,
                backbone.image_encoder.return_layers,
                image_features
            )

    return image_features


def save_image_feature_cache(
    image_features: torch.Tensor,
    token: str,
    log_name: str,
    domain: str,
    cache_dir: Path,
) -> None:
    """
    Save image features to cache as .pt file.

    Cache structure: cache_dir/<domain>/<log_name>/<token>.pt

    Args:
        image_features: Image feature tensor
        token: Scene token
        log_name: Log name
        domain: "navsim" or "metadrive"
        cache_dir: Root cache directory
    """
    out_dir = cache_dir / domain / log_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{token}.pt"

    torch.save({
        "image_features": image_features.cpu(),
        "token": token,
        "log_name": log_name,
    }, out_path)


# ============================================================================
# Visualization Functions
# ============================================================================

def feature_to_heatmap(
    features: np.ndarray,
    method: str = "mean",
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Convert feature map to heatmap visualization.

    Args:
        features: Feature tensor (C, H, W)
        method: Aggregation method - "mean", "max", or "std"
        colormap: OpenCV colormap to use

    Returns:
        RGB heatmap image (H, W, 3) as uint8
    """
    if method == "mean":
        agg = features.mean(axis=0)
    elif method == "max":
        agg = features.max(axis=0)
    elif method == "std":
        agg = features.std(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to [0, 255]
    agg_min, agg_max = agg.min(), agg.max()
    if agg_max - agg_min > 1e-8:
        agg_norm = (agg - agg_min) / (agg_max - agg_min)
    else:
        agg_norm = np.zeros_like(agg)
    agg_uint8 = (agg_norm * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(agg_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def feature_to_pca_rgb(features: np.ndarray) -> np.ndarray:
    """
    Convert feature map to RGB using PCA (reduce channels to 3).

    Args:
        features: Feature tensor (C, H, W)

    Returns:
        RGB image (H, W, 3) as uint8
    """
    C, H, W = features.shape

    # Reshape to (C, H*W) for PCA
    flat = features.reshape(C, -1).T  # (H*W, C)

    # Center the data
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    # Compute covariance and eigenvectors
    try:
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Take top 3 components
        top3_idx = np.argsort(eigenvalues)[-3:][::-1]
        top3_vecs = eigenvectors[:, top3_idx]

        # Project data onto top 3 components
        projected = centered @ top3_vecs  # (H*W, 3)

        # Normalize each channel to [0, 255]
        rgb = np.zeros((H * W, 3), dtype=np.uint8)
        for i in range(3):
            ch = projected[:, i]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                ch_norm = (ch - ch_min) / (ch_max - ch_min)
            else:
                ch_norm = np.zeros_like(ch)
            rgb[:, i] = (ch_norm * 255).astype(np.uint8)

        return rgb.reshape(H, W, 3)

    except Exception:
        # Fallback to simple channel selection if PCA fails
        if C >= 3:
            # Take evenly spaced channels
            indices = [0, C // 2, C - 1]
            selected = features[indices].transpose(1, 2, 0)  # (H, W, 3)
        else:
            # Duplicate channels
            selected = np.stack([features[0]] * 3, axis=-1)

        # Normalize
        s_min, s_max = selected.min(), selected.max()
        if s_max - s_min > 1e-8:
            selected = (selected - s_min) / (s_max - s_min)
        return (selected * 255).astype(np.uint8)


def create_feature_visualization(
    camera_image: np.ndarray,
    image_features: np.ndarray,
    title: str = "",
) -> np.ndarray:
    """
    Create comprehensive visualization of image features.

    Layout:
    - Top: Original camera image
    - Bottom left: Mean heatmap
    - Bottom middle: Max heatmap
    - Bottom right: PCA RGB

    Args:
        camera_image: Camera input (3, H, W) or (H, W, 3), normalized [0, 1] or [0, 255]
        image_features: Feature tensor (C, H, W)
        title: Title for the visualization

    Returns:
        Combined visualization image (H, W, 3) as uint8
    """
    # Prepare camera image
    if camera_image.ndim == 3 and camera_image.shape[0] == 3:
        camera_image = camera_image.transpose(1, 2, 0)
    if camera_image.max() <= 1.0:
        camera_rgb = (camera_image * 255).astype(np.uint8)
    else:
        camera_rgb = camera_image.astype(np.uint8)

    # Get feature map dimensions
    C, feat_H, feat_W = image_features.shape

    # Create feature visualizations
    heatmap_mean = feature_to_heatmap(image_features, method="mean")
    heatmap_max = feature_to_heatmap(image_features, method="max")
    pca_rgb = feature_to_pca_rgb(image_features)

    # Resize feature maps to match a reasonable display size
    display_h, display_w = 128, 256
    heatmap_mean = cv2.resize(heatmap_mean, (display_w, display_h))
    heatmap_max = cv2.resize(heatmap_max, (display_w, display_h))
    pca_rgb = cv2.resize(pca_rgb, (display_w, display_h))

    # Resize camera image
    cam_h, cam_w = camera_rgb.shape[:2]
    target_cam_w = display_w * 3
    target_cam_h = int(cam_h * target_cam_w / cam_w)
    camera_resized = cv2.resize(camera_rgb, (target_cam_w, target_cam_h))

    # Create canvas
    title_h = 30
    label_h = 20
    canvas_w = display_w * 3
    canvas_h = title_h + target_cam_h + label_h + display_h + 10
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Add title
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add feature shape info
    shape_text = f"Features: {C}x{feat_H}x{feat_W}"
    cv2.putText(canvas, shape_text, (canvas_w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Place camera image
    y_offset = title_h
    canvas[y_offset:y_offset + target_cam_h, :target_cam_w] = camera_resized

    # Add labels for feature visualizations
    y_offset = title_h + target_cam_h + 5
    cv2.putText(canvas, "Mean Activation", (10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, "Max Activation", (display_w + 10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(canvas, "PCA RGB", (display_w * 2 + 10, y_offset + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Place feature visualizations
    y_offset = title_h + target_cam_h + label_h
    canvas[y_offset:y_offset + display_h, 0:display_w] = heatmap_mean
    canvas[y_offset:y_offset + display_h, display_w:display_w * 2] = heatmap_max
    canvas[y_offset:y_offset + display_h, display_w * 2:display_w * 3] = pca_rgb

    return canvas


def create_domain_comparison_visualization(
    navsim_vis: np.ndarray,
    metadrive_vis: np.ndarray,
    token: str,
) -> np.ndarray:
    """
    Create side-by-side comparison of NAVSIM and MetaDrive features.

    Args:
        navsim_vis: NAVSIM visualization
        metadrive_vis: MetaDrive visualization
        token: Scene token for title

    Returns:
        Combined comparison image
    """
    h = max(navsim_vis.shape[0], metadrive_vis.shape[0])
    w = navsim_vis.shape[1]

    # Create canvas for 2 columns
    canvas = np.ones((h + 40, w * 2, 3), dtype=np.uint8) * 255

    # Add main title
    cv2.putText(canvas, f"Token: {token[:30]}...", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Place visualizations
    canvas[40:40 + navsim_vis.shape[0], :w] = navsim_vis
    canvas[40:40 + metadrive_vis.shape[0], w:w * 2] = metadrive_vis

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
        image: Image to save (H, W, 3) RGB
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


class ImageFeatureDataset(Dataset):
    """
    Dataset for extracting image features from NAVSIM and MetaDrive.

    Both domains use the same pickle logs but different sensor_blobs paths.
    """

    def __init__(
        self,
        navsim_scene_loader: SceneLoader,
        metadrive_scene_loader: Optional[SceneLoader],
        feature_builder: TransfuserFeatureBuilder,
        config: Any,
        cache_dir: Path,
        skip_existing: bool = False,
        process_metadrive: bool = False,
    ):
        self.navsim_scene_loader = navsim_scene_loader
        self.metadrive_scene_loader = metadrive_scene_loader
        self.feature_builder = feature_builder
        self.config = config
        self.cache_dir = cache_dir
        self.skip_existing = skip_existing
        self.process_metadrive = process_metadrive

        self.tokens = navsim_scene_loader.tokens
        self.tokens_per_log = navsim_scene_loader.get_tokens_list_per_log()

        self.token_to_log: Dict[str, str] = {}
        for log_name, tokens in self.tokens_per_log.items():
            for token in tokens:
                self.token_to_log[token] = log_name

        if skip_existing:
            self.tokens = self._filter_existing_tokens()

        print(f"ImageFeatureDataset: {len(self.tokens)} tokens to process")

    def _filter_existing_tokens(self) -> List[str]:
        """Filter out tokens that already have cached features."""
        filtered = []
        domains = ["navsim"]
        if self.process_metadrive:
            domains.append("metadrive")

        for token in self.tokens:
            log_name = self.token_to_log.get(token, "unknown")
            all_exist = all(
                (self.cache_dir / domain / log_name / f"{token}.pt").exists()
                for domain in domains
            )
            if not all_exist:
                filtered.append(token)

        print(f"Filtered {len(self.tokens) - len(filtered)} existing tokens")
        return filtered

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        token = self.tokens[idx]
        log_name = self.token_to_log.get(token, "unknown")

        try:
            # NAVSIM domain
            navsim_agent_input = self.navsim_scene_loader.get_agent_input_from_token(token)
            navsim_camera = self.feature_builder._get_camera_feature(navsim_agent_input)

            result = {
                "token": token,
                "log_name": log_name,
                "navsim_camera": navsim_camera,
            }

            # MetaDrive domain (optional)
            if self.process_metadrive and self.metadrive_scene_loader is not None:
                metadrive_agent_input = self.metadrive_scene_loader.get_agent_input_from_token(token)
                metadrive_camera = self.feature_builder._get_camera_feature(metadrive_agent_input)
                result["metadrive_camera"] = metadrive_camera

            return result

        except Exception as e:
            print(f"Error loading token {token}: {e}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """Collate function that filters None samples and stacks tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    tokens = [b["token"] for b in batch]
    log_names = [b["log_name"] for b in batch]
    navsim_cameras = torch.stack([b["navsim_camera"] for b in batch])

    result = {
        "tokens": tokens,
        "log_names": log_names,
        "navsim_cameras": navsim_cameras,
    }

    if "metadrive_camera" in batch[0]:
        metadrive_cameras = torch.stack([b["metadrive_camera"] for b in batch])
        result["metadrive_cameras"] = metadrive_cameras

    return result


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
        scene_filter = SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            has_route=True,
            log_names=args.log_names,
            max_scenes=args.max_scenes,
        )

    # Sensor config for cameras only
    sensor_config = SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=False,
        cam_l2=False,
        cam_r0=[3],
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,  # No LiDAR needed for stem features
    )

    # Build scene loaders
    print("Building scene loaders...")
    navsim_loader = SceneLoader(
        data_path=args.navsim_log_path,
        original_sensor_path=args.navsim_sensor_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    process_metadrive = "metadrive" in args.domains and args.metadrive_sensor_path is not None
    metadrive_loader = None
    if process_metadrive:
        metadrive_loader = SceneLoader(
            data_path=args.navsim_log_path,
            original_sensor_path=args.metadrive_sensor_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

    print(f"Total tokens: {len(navsim_loader.tokens)}")

    # Create dataset
    dataset = ImageFeatureDataset(
        navsim_scene_loader=navsim_loader,
        metadrive_scene_loader=metadrive_loader,
        feature_builder=feature_builder,
        config=config,
        cache_dir=args.cache_dir,
        skip_existing=args.skip_existing,
        process_metadrive=process_metadrive,
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
    progress = tqdm(dataloader, desc="Precomputing stem image features", disable=(rank != 0))
    for batch in progress:
        if batch is None:
            continue

        tokens = batch["tokens"]
        log_names = batch["log_names"]

        # Extract NAVSIM stem image features
        navsim_features = extract_stem_image_features(
            model,
            batch["navsim_cameras"],
            device,
        )

        # Extract MetaDrive stem image features (if processing)
        metadrive_features = None
        if process_metadrive and "metadrive_cameras" in batch:
            metadrive_features = extract_stem_image_features(
                model,
                batch["metadrive_cameras"],
                device,
            )

        # Save to cache
        batch_size_actual = len(tokens)
        for i in range(batch_size_actual):
            token = tokens[i]
            log_name = log_names[i]

            # Save NAVSIM features
            save_image_feature_cache(
                navsim_features[i],
                token,
                log_name,
                "navsim",
                args.cache_dir,
            )

            # Save MetaDrive features
            if metadrive_features is not None:
                save_image_feature_cache(
                    metadrive_features[i],
                    token,
                    log_name,
                    "metadrive",
                    args.cache_dir,
                )

        # Generate visualizations if enabled
        if args.visualize and vis_count < args.vis_num_samples:
            for i in range(min(batch_size_actual, args.vis_num_samples - vis_count)):
                token = tokens[i]

                # Create NAVSIM visualization
                navsim_vis = create_feature_visualization(
                    camera_image=batch["navsim_cameras"][i].numpy(),
                    image_features=navsim_features[i].cpu().numpy(),
                    title=f"NAVSIM - {token[:20]}",
                )
                save_visualization(navsim_vis, vis_output_dir / "navsim", token)

                # Create MetaDrive visualization (if processing)
                if metadrive_features is not None and "metadrive_cameras" in batch:
                    metadrive_vis = create_feature_visualization(
                        camera_image=batch["metadrive_cameras"][i].numpy(),
                        image_features=metadrive_features[i].cpu().numpy(),
                        title=f"MetaDrive - {token[:20]}",
                    )
                    save_visualization(metadrive_vis, vis_output_dir / "metadrive", token)

                    # Create domain comparison if requested
                    if args.vis_compare_domains:
                        comparison = create_domain_comparison_visualization(
                            navsim_vis, metadrive_vis, token
                        )
                        save_visualization(comparison, vis_output_dir / "comparison", token)

                vis_count += 1

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    print("Done!")


if __name__ == "__main__":
    main()

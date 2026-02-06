#!/usr/bin/env python3
"""
PyTorch Lightning training scaffold for image feature flow matching.

Key behaviors:
- Extracts image features ON-THE-FLY from raw sensor images using TransFuser backbone.
- No pre-cached H5 files needed - features computed during training.
- Uses DiTNonSquare for non-square feature maps (64 channels, 64 height, 256 width).
- Source domain: MetaDrive
- Target domain: NAVSIM

Assumptions:
- NAVSIM and MetaDrive sensor images available at specified paths.
- TransFuser checkpoint available for feature extraction.
- Image stem features shape: (64, 64, 256) after ResNet34 stem.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset

# Add navsim to path
SCRIPT_DIR = Path(__file__).resolve().parent
NAVSIM_ROOT = SCRIPT_DIR.parent / "navsim"
sys.path.insert(0, str(NAVSIM_ROOT))

from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from bridgesim.modelzoo.navsim.common.dataclasses import SceneFilter, SensorConfig
from bridgesim.modelzoo.navsim.common.dataloader import SceneLoader

from image_feature_flow_model import ImageFeatureFlowMatchingModel


# Image feature dimensions (after ResNet34 stem with input 3x256x1024)
# conv1 (7x7, stride 2): 256->128, 1024->512
IMAGE_FEATURE_CHANNELS = 64
IMAGE_FEATURE_HEIGHT = 128
IMAGE_FEATURE_WIDTH = 512


@dataclass
class TokenEntry:
    """Entry representing a scene token."""
    token: str
    log_name: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LogitNormalSampler:
    """
    Logit-normal timestep sampler for flow matching.
    Follows https://arxiv.org/pdf/2403.03206.pdf

    Samples from a normal distribution and passes through sigmoid.
    This biases sampling towards t=0.5 (or shifted by normal_mean).
    """
    def __init__(self, normal_mean: float = 0.0, normal_std: float = 1.0):
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps in [0, 1] using logit-normal distribution."""
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(batch_size,),
            device=device,
        )
        x_logistic = torch.sigmoid(x_normal)
        return x_logistic


def build_optimizer(params, name: str, lr: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name}")


def build_lr_scheduler(
    optimizer, name: str, *, warmup_steps: int = -1, t_max: int = 100
):
    name = name.lower()
    if name == "none":
        return None
    if name == "customized":
        def fn(step: int):
            if warmup_steps > 0:
                return min(step / warmup_steps, 1.0)
            return 1.0
        return LambdaLR(optimizer, fn)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=t_max)
    raise ValueError(f"Unsupported lr scheduler {name}")


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


# Model type to import mapping
MODEL_TYPES = ["transfuser", "diffusiondrive", "diffusiondrivev2"]


def load_model_and_config(
    checkpoint_path: Path,
    device: torch.device,
    model_type: str = "transfuser",
    plan_anchor_path: Optional[Path] = None,
) -> Tuple[nn.Module, Any]:
    """Load model from checkpoint based on model type."""
    print(f"Loading {model_type} model for feature extraction...")

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

    print(f"{model_type} model loaded successfully (frozen for feature extraction).")
    return model, config


def extract_stem_image_features(
    model: nn.Module,
    camera_feature: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract image features after the stem layer (line 174 of transfuser_backbone.py).

    Args:
        model: TransFuser model
        camera_feature: Camera input tensor (B, 3, 256, 1024)
        device: Device to run on

    Returns:
        image_features tensor after stem layer (B, 64, 64, 256)
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


class CameraImagePairDataset(Dataset):
    """
    Dataset that loads raw camera images from both domains.
    Feature extraction is done in the Lightning module (on GPU in main process).
    This allows num_workers > 0 without CUDA multiprocessing issues.
    """

    def __init__(
        self,
        entries: List[TokenEntry],
        navsim_scene_loader: SceneLoader,
        metadrive_scene_loader: SceneLoader,
        feature_builder: TransfuserFeatureBuilder,
    ):
        self.entries = entries
        self.navsim_loader = navsim_scene_loader
        self.metadrive_loader = metadrive_scene_loader
        self.feature_builder = feature_builder

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        token = entry.token

        try:
            # Load raw camera images from both domains (CPU only, no CUDA)
            navsim_agent_input = self.navsim_loader.get_agent_input_from_token(token)
            metadrive_agent_input = self.metadrive_loader.get_agent_input_from_token(token)

            navsim_camera = self.feature_builder._get_camera_feature(navsim_agent_input)
            metadrive_camera = self.feature_builder._get_camera_feature(metadrive_agent_input)

            # Return raw camera tensors (3, 256, 1024) - feature extraction in Lightning module
            return metadrive_camera, navsim_camera

        except Exception as e:
            print(f"Error loading token {token}: {e}")
            return None


def collate_fn(batch: List):
    """Collate function that filters None entries."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    metadrive_cameras, navsim_cameras = zip(*batch)
    metadrive_batch = torch.stack(metadrive_cameras, dim=0)
    navsim_batch = torch.stack(navsim_cameras, dim=0)
    return metadrive_batch, navsim_batch


class ImageFeatureFlowMatchingLit(pl.LightningModule):
    """PyTorch Lightning module for image feature flow matching.

    Feature extraction from raw camera images happens in this module (on GPU),
    not in the dataloader workers, to avoid CUDA multiprocessing issues.
    """

    def __init__(
        self,
        feature_height: int = IMAGE_FEATURE_HEIGHT,
        feature_width: int = IMAGE_FEATURE_WIDTH,
        feature_channels: int = IMAGE_FEATURE_CHANNELS,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        lr_scheduler_name: str = "customized",
        warmup_steps: int = 500,
        lr_t_max: int = 1000,
        ema_decay: float = 0.9999,
        dit_variant: str = "DiT-B/4",
        # Timestep sampling parameters
        t_sampler: str = "logit_normal",  # "uniform" or "logit_normal"
        t_normal_mean: float = 0.0,
        t_normal_std: float = 1.0,
        # Backbone parameters for feature extraction
        checkpoint_path: Optional[Path] = None,
        model_type: str = "transfuser",
        plan_anchor_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.warmup_steps = warmup_steps
        self.lr_t_max = lr_t_max
        self.ema_decay = ema_decay
        self.t_sampler_name = t_sampler

        # Store backbone loading params (loaded lazily on first forward)
        self._checkpoint_path = checkpoint_path
        self._model_type = model_type
        self._plan_anchor_path = plan_anchor_path
        self._backbone: Optional[nn.Module] = None

        # Timestep sampler
        if t_sampler == "logit_normal":
            self.t_sampler = LogitNormalSampler(normal_mean=t_normal_mean, normal_std=t_normal_std)
        else:
            self.t_sampler = None  # Use uniform sampling

        # Flow model configured for image feature dimensions
        self.flow_model = ImageFeatureFlowMatchingModel(
            feature_height=feature_height,
            feature_width=feature_width,
            feature_channels=feature_channels,
            dit_variant=dit_variant,
        )

        # EMA model
        self.ema_flow_model = deepcopy(self.flow_model)
        for p in self.ema_flow_model.parameters():
            p.requires_grad_(False)

    def _ensure_backbone_loaded(self) -> nn.Module:
        """Lazy-load the backbone for feature extraction."""
        if self._backbone is None:
            if self._checkpoint_path is None:
                raise RuntimeError("checkpoint_path must be provided for feature extraction")
            self._backbone, _ = load_model_and_config(
                self._checkpoint_path,
                self.device,
                self._model_type,
                self._plan_anchor_path,
            )
        return self._backbone

    def _extract_features(self, camera_batch: torch.Tensor) -> torch.Tensor:
        """Extract stem features from raw camera images on GPU."""
        backbone = self._ensure_backbone_loaded()
        return extract_stem_image_features(backbone, camera_batch, self.device)

    def on_train_epoch_end(self):
        """Periodic garbage collection during training."""
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.flow_model.parameters(),
            name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            name=self.lr_scheduler_name,
            warmup_steps=self.warmup_steps,
            t_max=self.lr_t_max,
        )
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def _ema_update(self):
        if self.ema_flow_model is None:
            return
        ema_params = dict(self.ema_flow_model.named_parameters())
        for name, param in self.flow_model.named_parameters():
            ema_param = ema_params[name]
            ema_param.data.mul_(self.ema_decay).add_(
                (1 - self.ema_decay) * param.data
            )

    def _compute_cosine_similarity(
        self,
        src_bchw: torch.Tensor,
        tgt_bchw: torch.Tensor,
    ) -> dict:
        """
        Compute cosine similarity between source and target features.
        """
        B, C, H, W = src_bchw.shape

        src_flat = src_bchw.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tgt_flat = tgt_bchw.flatten(2).transpose(1, 2)  # (B, H*W, C)

        src_norm = torch.nn.functional.normalize(src_flat, p=2, dim=-1)
        tgt_norm = torch.nn.functional.normalize(tgt_flat, p=2, dim=-1)

        cos_sim = (src_norm * tgt_norm).sum(dim=-1)  # (B, H*W)
        per_position_mean = cos_sim.mean(dim=0)
        mean_cos_sim = cos_sim.mean()

        return {
            "mean": mean_cos_sim,
            "per_position_mean": per_position_mean,
        }

    def _flow_step(
        self,
        src_features: torch.Tensor,
        tgt_features: torch.Tensor,
        stage: str,
    ):
        """Shared flow-matching computation for train/val."""
        # src_features and tgt_features are already (B, C, H, W)
        batch_size = src_features.shape[0]

        # Sample timesteps using configured sampler
        if self.t_sampler is not None:
            t = self.t_sampler.sample(batch_size, device=self.device)
        else:
            t = torch.rand(batch_size, device=self.device)

        # Interpolate and compute target velocity
        x_t = self.flow_model.psi(t, noise=src_features, x1=tgt_features)
        target_velocity = self.flow_model.dt_psi(t, noise=src_features, x1=tgt_features)

        # Predict velocity
        v_pred = self.flow_model(x_t, t)

        # MSE loss
        loss = (v_pred - target_velocity).pow(2).flatten(1).mean()

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._ema_update()

    def _run_stage(self, batch, stage: str):
        if batch is None:
            return torch.zeros((), device=self.device)

        # batch contains raw camera images: (metadrive_cameras, navsim_cameras)
        src_cameras, tgt_cameras = batch
        src_cameras = src_cameras.to(self.device)
        tgt_cameras = tgt_cameras.to(self.device)

        # Extract features on GPU (no gradient for backbone)
        with torch.no_grad():
            src_features = self._extract_features(src_cameras)
            tgt_features = self._extract_features(tgt_cameras)

        return self._flow_step(src_features, tgt_features, stage=stage)

    def training_step(self, batch, batch_idx: int):
        return self._run_stage(batch, stage="train")

    def validation_step(self, batch, batch_idx: int):
        return self._run_stage(batch, stage="val")

    def test_step(self, batch, batch_idx: int):
        if batch is None:
            return

        # batch contains raw camera images: (metadrive_cameras, navsim_cameras)
        src_cameras, tgt_cameras = batch
        src_cameras = src_cameras.to(self.device)
        tgt_cameras = tgt_cameras.to(self.device)

        # Extract features on GPU (no gradient for backbone)
        with torch.no_grad():
            src_features = self._extract_features(src_cameras)
            tgt_features = self._extract_features(tgt_cameras)

        # Sample from flow model
        sampler_model = (
            self.ema_flow_model if self.ema_flow_model is not None else self.flow_model
        )
        gen_features = sampler_model.sample(src_features)

        # Compute reconstruction loss
        loss = (gen_features - tgt_features).pow(2).flatten(1).mean()

        self.log(
            "test/loss",
            loss.detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Compute cosine similarity between generated and target features
        cos_sim = self._compute_cosine_similarity(gen_features, tgt_features)

        self.log(
            "test/cosine_sim_mean",
            cos_sim["mean"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )


class CameraImagePairDataModule(pl.LightningDataModule):
    """
    Data module for camera image pairs.
    Loads raw sensor images; feature extraction happens in the Lightning module.
    Supports separate train/val data paths.
    """

    def __init__(
        self,
        # Train data paths
        train_navsim_log_path: Path,
        train_navsim_sensor_path: Path,
        train_metadrive_sensor_path: Path,
        train_split_yaml: Optional[Path] = None,
        # Val data paths (optional - if not provided, uses train paths with val_split_yaml)
        val_navsim_log_path: Optional[Path] = None,
        val_navsim_sensor_path: Optional[Path] = None,
        val_metadrive_sensor_path: Optional[Path] = None,
        val_split_yaml: Optional[Path] = None,
        # Model config (for feature builder only, backbone loaded in Lightning module)
        model_type: str = "transfuser",
        plan_anchor_path: Optional[Path] = None,
        # Data loading
        log_names: Optional[List[str]] = None,
        batch_size: int = 16,
        num_workers: int = 4,  # Can use workers now since no CUDA in dataset
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Train paths
        self.train_navsim_log_path = train_navsim_log_path
        self.train_navsim_sensor_path = train_navsim_sensor_path
        self.train_metadrive_sensor_path = train_metadrive_sensor_path
        self.train_split_yaml = train_split_yaml
        # Val paths (default to train paths if not provided)
        self.val_navsim_log_path = val_navsim_log_path or train_navsim_log_path
        self.val_navsim_sensor_path = val_navsim_sensor_path or train_navsim_sensor_path
        self.val_metadrive_sensor_path = val_metadrive_sensor_path or train_metadrive_sensor_path
        self.val_split_yaml = val_split_yaml
        # Model config
        self.model_type = model_type
        self.plan_anchor_path = plan_anchor_path
        # Data loading
        self.log_names = log_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        self.train_entries: List[TokenEntry] = []
        self.val_entries: List[TokenEntry] = []
        self.feature_builder: Optional[TransfuserFeatureBuilder] = None
        # Separate loaders for train and val
        self.train_navsim_loader: Optional[SceneLoader] = None
        self.train_metadrive_loader: Optional[SceneLoader] = None
        self.val_navsim_loader: Optional[SceneLoader] = None
        self.val_metadrive_loader: Optional[SceneLoader] = None

    def _build_scene_filter(self, split_yaml: Optional[Path], max_samples: Optional[int]) -> SceneFilter:
        """Build scene filter from YAML or defaults."""
        if split_yaml is not None:
            print(f"Loading scene filter from YAML: {split_yaml}")
            return load_scene_filter_from_yaml(
                split_yaml,
                log_names_override=self.log_names,
                max_scenes_override=max_samples,
            )
        else:
            return SceneFilter(
                num_history_frames=4,
                num_future_frames=10,
                has_route=True,
                log_names=self.log_names,
                max_scenes=max_samples,
            )

    def _build_entries_from_loader(self, loader: SceneLoader) -> List[TokenEntry]:
        """Build TokenEntry list from a SceneLoader."""
        tokens_per_log = loader.get_tokens_list_per_log()
        token_to_log: Dict[str, str] = {}
        for log_name, tokens in tokens_per_log.items():
            for token in tokens:
                token_to_log[token] = log_name

        return [
            TokenEntry(token=token, log_name=token_to_log.get(token, "unknown"))
            for token in loader.tokens
        ]

    def setup(self, stage: Optional[str] = None) -> None:
        # Create feature builder for camera preprocessing (no backbone needed here)
        if self.model_type == "transfuser":
            from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_config import TransfuserConfig
            config = TransfuserConfig()
        elif self.model_type == "diffusiondrive":
            from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
            config = TransfuserConfig()
            if self.plan_anchor_path is not None:
                config.plan_anchor_path = str(self.plan_anchor_path)
        elif self.model_type == "diffusiondrivev2":
            from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
            config = TransfuserConfig()
            if self.plan_anchor_path is not None:
                config.plan_anchor_path = str(self.plan_anchor_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.feature_builder = TransfuserFeatureBuilder(config)

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
            lidar_pc=False,
        )

        # Build TRAIN scene loaders
        print("Building TRAIN scene loaders...")
        train_scene_filter = self._build_scene_filter(self.train_split_yaml, self.max_train_samples)
        self.train_navsim_loader = SceneLoader(
            data_path=self.train_navsim_log_path,
            original_sensor_path=self.train_navsim_sensor_path,
            scene_filter=train_scene_filter,
            sensor_config=sensor_config,
        )
        self.train_metadrive_loader = SceneLoader(
            data_path=self.train_navsim_log_path,
            original_sensor_path=self.train_metadrive_sensor_path,
            scene_filter=train_scene_filter,
            sensor_config=sensor_config,
        )
        self.train_entries = self._build_entries_from_loader(self.train_navsim_loader)
        print(f"Loaded {len(self.train_entries)} train entries")

        # Build VAL scene loaders
        print("Building VAL scene loaders...")
        val_scene_filter = self._build_scene_filter(self.val_split_yaml, self.max_val_samples)
        self.val_navsim_loader = SceneLoader(
            data_path=self.val_navsim_log_path,
            original_sensor_path=self.val_navsim_sensor_path,
            scene_filter=val_scene_filter,
            sensor_config=sensor_config,
        )
        self.val_metadrive_loader = SceneLoader(
            data_path=self.val_navsim_log_path,
            original_sensor_path=self.val_metadrive_sensor_path,
            scene_filter=val_scene_filter,
            sensor_config=sensor_config,
        )
        self.val_entries = self._build_entries_from_loader(self.val_navsim_loader)
        print(f"Loaded {len(self.val_entries)} val entries")

    def train_dataloader(self):
        return DataLoader(
            CameraImagePairDataset(
                self.train_entries,
                navsim_scene_loader=self.train_navsim_loader,
                metadrive_scene_loader=self.train_metadrive_loader,
                feature_builder=self.feature_builder,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            CameraImagePairDataset(
                self.val_entries,
                navsim_scene_loader=self.val_navsim_loader,
                metadrive_scene_loader=self.val_metadrive_loader,
                feature_builder=self.feature_builder,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightning training for image feature flow matching."
    )
    # Train data paths
    parser.add_argument(
        "--train-navsim-log-path",
        type=Path,
        required=True,
        help="Path to NAVSIM log pickles directory for training.",
    )
    parser.add_argument(
        "--train-navsim-sensor-path",
        type=Path,
        required=True,
        help="Path to NAVSIM sensor blobs directory for training.",
    )
    parser.add_argument(
        "--train-metadrive-sensor-path",
        type=Path,
        required=True,
        help="Path to MetaDrive sensor images directory for training.",
    )
    parser.add_argument(
        "--train-split-yaml",
        type=Path,
        default=None,
        help="Path to split YAML file defining SceneFilter for training.",
    )

    # Val data paths (optional - defaults to train paths if not provided)
    parser.add_argument(
        "--val-navsim-log-path",
        type=Path,
        default=None,
        help="Path to NAVSIM log pickles directory for validation (default: same as train).",
    )
    parser.add_argument(
        "--val-navsim-sensor-path",
        type=Path,
        default=None,
        help="Path to NAVSIM sensor blobs directory for validation (default: same as train).",
    )
    parser.add_argument(
        "--val-metadrive-sensor-path",
        type=Path,
        default=None,
        help="Path to MetaDrive sensor images directory for validation (default: same as train).",
    )
    parser.add_argument(
        "--val-split-yaml",
        type=Path,
        default=None,
        help="Path to split YAML file defining SceneFilter for validation.",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to TransFuser model checkpoint (.ckpt).",
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

    # Data loading
    parser.add_argument(
        "--log-names",
        type=str,
        nargs="+",
        default=None,
        help="Specific log names to process.",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data workers (0 recommended for on-the-fly extraction).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="customized",
        choices=["none", "customized", "cosine"],
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-t-max", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)

    # Devices
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--devices", default="auto",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
    )

    # Checkpointing
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from checkpoint path.",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=1,
    )

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ImageFeatureFlow")
    parser.add_argument("--model-name", type=str, default="transfuser")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # DiT
    parser.add_argument(
        "--dit-variant",
        type=str,
        default="DiT-B/4",
        help="DiT architecture variant (e.g., DiT-S/4, DiT-B/4, DiT-L/4).",
    )

    # Timestep sampling (logit-normal from https://arxiv.org/pdf/2403.03206.pdf)
    parser.add_argument(
        "--t-sampler",
        type=str,
        default="logit_normal",
        choices=["uniform", "logit_normal"],
        help="Timestep sampling method: uniform or logit_normal.",
    )
    parser.add_argument(
        "--t-normal-mean",
        type=float,
        default=0.0,
        help="Mean for logit-normal timestep sampling.",
    )
    parser.add_argument(
        "--t-normal-std",
        type=float,
        default=1.0,
        help="Std for logit-normal timestep sampling.",
    )

    # Data limits
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Maximum number of training samples.")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Maximum number of validation samples.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision("high")

    # Validate required paths
    if not args.train_navsim_log_path.exists():
        raise SystemExit(f"Train NAVSIM log path not found: {args.train_navsim_log_path}")
    if not args.train_navsim_sensor_path.exists():
        raise SystemExit(f"Train NAVSIM sensor path not found: {args.train_navsim_sensor_path}")
    if not args.train_metadrive_sensor_path.exists():
        raise SystemExit(f"Train MetaDrive sensor path not found: {args.train_metadrive_sensor_path}")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    # Validate optional val paths if provided
    if args.val_navsim_log_path and not args.val_navsim_log_path.exists():
        raise SystemExit(f"Val NAVSIM log path not found: {args.val_navsim_log_path}")
    if args.val_navsim_sensor_path and not args.val_navsim_sensor_path.exists():
        raise SystemExit(f"Val NAVSIM sensor path not found: {args.val_navsim_sensor_path}")
    if args.val_metadrive_sensor_path and not args.val_metadrive_sensor_path.exists():
        raise SystemExit(f"Val MetaDrive sensor path not found: {args.val_metadrive_sensor_path}")

    datamodule = CameraImagePairDataModule(
        # Train paths
        train_navsim_log_path=args.train_navsim_log_path,
        train_navsim_sensor_path=args.train_navsim_sensor_path,
        train_metadrive_sensor_path=args.train_metadrive_sensor_path,
        train_split_yaml=args.train_split_yaml,
        # Val paths
        val_navsim_log_path=args.val_navsim_log_path,
        val_navsim_sensor_path=args.val_navsim_sensor_path,
        val_metadrive_sensor_path=args.val_metadrive_sensor_path,
        val_split_yaml=args.val_split_yaml,
        # Model config (for feature builder only)
        model_type=args.model_type,
        plan_anchor_path=args.plan_anchor_path,
        # Data loading
        log_names=args.log_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    lit_model = ImageFeatureFlowMatchingLit(
        feature_height=IMAGE_FEATURE_HEIGHT,
        feature_width=IMAGE_FEATURE_WIDTH,
        feature_channels=IMAGE_FEATURE_CHANNELS,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler_name=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        lr_t_max=args.lr_t_max,
        ema_decay=args.ema_decay,
        dit_variant=args.dit_variant,
        # Timestep sampling parameters
        t_sampler=args.t_sampler,
        t_normal_mean=args.t_normal_mean,
        t_normal_std=args.t_normal_std,
        # Backbone parameters for feature extraction
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        plan_anchor_path=args.plan_anchor_path,
    )

    # Build run name
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model_name}-imgflow-{args.dit_variant}"

    # Include run name in checkpoint directory
    ckpt_dir = args.ckpt_dir / args.wandb_run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{step}",
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_weights_only=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = None
    if args.wandb:
        wandb.login(key="caad08df59bfd0cb22f3613849ad66faeb65d4b0")
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(ckpt_dir),
            config=vars(args),
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=32,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(
        lit_model,
        datamodule=datamodule,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )


if __name__ == "__main__":
    main()

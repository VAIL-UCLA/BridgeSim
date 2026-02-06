#!/usr/bin/env python3
"""
UniAD feature extraction + training scaffold for Bench2Drive/MetaDrive pairs.

Goals:
- Mirror the high-level training structure of CrossFlow/train_t2i.py.
- Load Bench2Drive/MetaDrive pairs, run UniAD backbone/neck to produce features.
- Provide train/val DataLoader construction and per-epoch loops.
- Leave flow-matching model/optimizer logic as TODO hooks.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local calibration package is importable when running directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.datasets import UniADPairDataModule, collate_uniad_pairs
from calibration.precompute_uniad_image_features import load_model_and_pipeline


@dataclass
class RunConfig:
    pairs: Path
    config: Path
    checkpoint: Path
    device: str = "cuda"
    batch_size: int = 1
    num_workers: int = 4
    train_ratio: float = 0.8
    split_seed: int = 42
    phase: str = "train"  # train | val | both
    epochs: int = 1
    pin_memory: bool = True
    shuffle_train: bool = True


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="UniAD feature extractor + scaffold for flow-matching training.")
    parser.add_argument("--pairs", type=Path, required=True, help="Path to pairs JSONL (build_bench2drive_metadrive_pairs.py).")
    parser.add_argument("--config", type=Path, required=True, help="Path to UniAD config (.py).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to UniAD checkpoint (.pth).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for feature extraction.")
    parser.add_argument("--batch-size", type=int, default=1, help="Pairs per batch.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "val", "both"], help="Which split(s) to run.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to iterate (affects train/val loops).")
    args = parser.parse_args()
    return RunConfig(
        pairs=args.pairs,
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        phase=args.phase,
        epochs=args.epochs,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_backbone_on_device(batch: Dict, model, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run UniAD backbone+neck on a collated batch, keeping features on device.
    Mirrors precompute_uniad_image_features.run_backbone_from_batch but without
    moving outputs to CPU (suitable for training on features).
    """
    for key, data in batch.items():
        if key == "img_metas":
            continue
        if torch.is_tensor(data):
            batch[key] = data.to(device, non_blocking=True)
        elif isinstance(data, (list, tuple)) and data and torch.is_tensor(data[0]):
            moved = list(data)
            moved[0] = moved[0].to(device, non_blocking=True)
            batch[key] = moved

    imgs = batch["img"][0] if isinstance(batch["img"], (list, tuple)) else batch["img"]
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)
    with torch.no_grad():
        assert imgs.dim() == 5, f"Expected 5D imgs, got {imgs.shape}"
        B, N, C, H, W = imgs.size()
        flat = imgs.view(B * N, C, H, W)
        if getattr(model, "use_grid_mask", False):
            flat = model.grid_mask(flat)

        backbone_feats = model.img_backbone(flat)
        if isinstance(backbone_feats, dict):
            backbone_feats = list(backbone_feats.values())

        neck_feats = backbone_feats
        if getattr(model, "with_img_neck", False):
            neck_feats = model.img_neck(backbone_feats)

        def _reshape(feat_list):
            reshaped = []
            for f in feat_list:
                _, c, h, w = f.size()
                reshaped.append(f.view(B, N, c, h, w).detach())
            return reshaped

        backbone_feats = _reshape(backbone_feats)
        neck_feats = _reshape(neck_feats)
    return backbone_feats, neck_feats


def slice_per_pair(feat_list: List[torch.Tensor], idx: int) -> List[torch.Tensor]:
    """Select the idx-th pair across views for each feature level."""
    return [feat[idx : idx + 1] for feat in feat_list]


def prepare_feature_pairs(
    pairs_batch,
    md_backbone: List[torch.Tensor],
    md_neck: List[torch.Tensor],
    b2d_backbone: List[torch.Tensor],
    b2d_neck: List[torch.Tensor],
) -> List[Dict]:
    feature_pairs = []
    for idx, pair in enumerate(pairs_batch):
        feature_pairs.append(
            {
                "pair": pair,
                "metadrive": {
                    "backbone": slice_per_pair(md_backbone, idx),
                    "neck": slice_per_pair(md_neck, idx),
                },
                "bench2drive": {
                    "backbone": slice_per_pair(b2d_backbone, idx),
                    "neck": slice_per_pair(b2d_neck, idx),
                },
            }
        )
    return feature_pairs


def make_loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_uniad_pairs,
    )


def train_one_epoch(cfg: RunConfig, loader: DataLoader, model, device: torch.device) -> Dict[str, float]:
    """
    Training scaffold. Replace TODO sections with flow-matching loss/optimizer logic.
    """
    model.eval()  # UniAD remains frozen; flow model (to be added) would be train()
    metrics = {}
    for batch in tqdm(loader, desc="train", dynamic_ncols=True):
        if batch is None:
            continue
        pairs_batch, md_batch, b2d_batch = batch
        md_backbone, md_neck = run_backbone_on_device(md_batch, model, device)
        b2d_backbone, b2d_neck = run_backbone_on_device(b2d_batch, model, device)

        feature_pairs = prepare_feature_pairs(pairs_batch, md_backbone, md_neck, b2d_backbone, b2d_neck)

        # TODO: flow_model_forward(feature_pairs) -> loss
        # TODO: loss.backward(); optimizer.step(); lr_scheduler.step()
        _ = feature_pairs
    return metrics


def eval_one_epoch(cfg: RunConfig, loader: DataLoader, model, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", dynamic_ncols=True):
            if batch is None:
                continue
            pairs_batch, md_batch, b2d_batch = batch
            md_backbone, md_neck = run_backbone_on_device(md_batch, model, device)
            b2d_backbone, b2d_neck = run_backbone_on_device(b2d_batch, model, device)
            feature_pairs = prepare_feature_pairs(pairs_batch, md_backbone, md_neck, b2d_backbone, b2d_neck)

            # TODO: evaluate flow_model on feature_pairs and log metrics.
            _ = feature_pairs
    return metrics


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.split_seed)
    device = torch.device(cfg.device)

    model, pipeline = load_model_and_pipeline(cfg.config, cfg.checkpoint, device)
    data_module = UniADPairDataModule(
        pairs=cfg.pairs,
        pipeline=pipeline,
        train_ratio=cfg.train_ratio,
        split_seed=cfg.split_seed,
    )

    train_loader = (
        make_loader(
            data_module.get_split("train"),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=cfg.shuffle_train,
            pin_memory=cfg.pin_memory,
        )
        if cfg.phase in {"train", "both"}
        else None
    )
    val_loader = (
        make_loader(
            data_module.get_split("val"),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=cfg.pin_memory,
        )
        if cfg.phase in {"val", "both"}
        else None
    )

    for epoch in range(cfg.epochs):
        if train_loader:
            _ = train_one_epoch(cfg, train_loader, model, device)
        if val_loader:
            _ = eval_one_epoch(cfg, val_loader, model, device)

        # TODO: add checkpointing/logging if needed.


if __name__ == "__main__":
    main()

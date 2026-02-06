#!/usr/bin/env python3
"""
Precompute MetaDrive/Bench2Drive BEV features and store them on disk for reuse.

This mirrors the BEV extraction flow in train_bev_flow_pl.py: builds samples via
PairDataset/PairDataModule utilities and runs UniAD get_bevs. Outputs per-domain
files under the provided cache directory keyed by scenario/frame.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_bev_flow_pl import PairDataset, collate_fn
from train_uniad_bev import (
    FrameBundle,
    load_frame_bundles,
    load_infos,
    load_inference_pipeline,
    load_model_and_pipeline,
    extract_bev_embedding_from_batch,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute BEV features for MetaDrive/Bench2Drive pairs.")
    parser.add_argument("--pairs", type=Path, required=True, help="Frame pairs JSONL.")
    parser.add_argument("--info-pkls", nargs="+", required=True, help="Info pkl(s) from prepare_B2D_nautilus.py.")
    parser.add_argument("--config", type=Path, required=True, help="UniAD config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="UniAD checkpoint path.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Output directory for cached BEVs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--default-command", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of pairs to process.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device. Defaults to cuda:LOCAL_RANK or cpu.")
    parser.add_argument("--distributed", action="store_true", help="Enable torch.distributed for multi-GPU sharding.")
    parser.add_argument("--local-rank", type=int, default=-1, help="torch.distributed launch utility compatibility.")
    return parser.parse_args()


def save_bev(bev: torch.Tensor, bundle: FrameBundle, domain: str, cache_dir: Path) -> None:
    out_dir = cache_dir / domain / bundle.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.frame}.pt"
    torch.save({"bev": bev.cpu(), "scenario": bundle.scenario, "frame": bundle.frame}, out_path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Distributed setup (single-node multi-GPU)
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
        local_rank = 0
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cache_dir = args.cache_dir
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # Build UniAD model + inference pipeline
    uni_model, pipeline = load_model_and_pipeline(str(args.config), str(args.checkpoint))
    for p in uni_model.parameters():
        p.requires_grad_(False)
    uni_model.eval().to(device)

    info_index = load_infos([Path(p) for p in args.info_pkls])

    # Load bundles
    bundles = load_frame_bundles(args.pairs)
    # Shard bundles across ranks
    bundles = bundles[rank::world_size]
    if args.max_samples is not None:
        bundles = bundles[: args.max_samples]

    dataset = PairDataset(
        bundles,
        pipeline=pipeline,
        info_index=info_index,
        default_command=args.default_command,
        fps=args.fps,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    progress = tqdm(dataloader, desc="Precompute BEVs", disable=(rank != 0))
    for batch in progress:
        if batch is None:
            continue
        bundles_batch, md_batch, b2d_batch = batch
        md_bev = extract_bev_embedding_from_batch(md_batch, uni_model, device)
        b2d_bev = extract_bev_embedding_from_batch(b2d_batch, uni_model, device)

        # md_bev / b2d_bev shapes can be (H*W, B, C) or (B, C, H, W); align to per-sample saves.
        batch_size = md_bev.shape[1] if md_bev.dim() == 3 else md_bev.shape[0]
        for idx in range(batch_size):
            md_slice = md_bev[:, idx : idx + 1] if md_bev.dim() == 3 else md_bev[idx : idx + 1]
            b2d_slice = b2d_bev[:, idx : idx + 1] if b2d_bev.dim() == 3 else b2d_bev[idx : idx + 1]
            bundle = bundles_batch[idx]
            # Skip if already present
            md_out = cache_dir / "metadrive" / bundle.scenario / f"{bundle.frame}.pt"
            b2d_out = cache_dir / "bench2drive" / bundle.scenario / f"{bundle.frame}.pt"
            if not md_out.exists():
                save_bev(md_slice, bundle, "metadrive", cache_dir)
            if not b2d_out.exists():
                save_bev(b2d_slice, bundle, "bench2drive", cache_dir)

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

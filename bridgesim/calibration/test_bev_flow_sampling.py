#!/usr/bin/env python3
"""
Evaluate the BEV flow sampler on cached MetaDrive/Bench2Drive pairs.

Workflow:
- Load val bundles from a pairs JSONL and split file.
- Read cached BEVs for MetaDrive/Bench2Drive.
- Sample Bench2Drive BEVs from MetaDrive BEVs using the EMA flow model.
- Run UniAD heads on generated (and GT) BEVs for detection/seg/occ.
- Save per-frame visuals (source/gen/target) and optional generated BEV tensors.
- Emit a JSON summary with simple reconstruction metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_bev_flow_pl import (
    FlowMatchingLit,
    PairDataset,
    collate_fn,
    load_split,
    set_seed,
    split_bundles,
)
from train_uniad_bev import (
    FrameBundle,
    load_frame_bundles,
    load_model_and_pipeline,
    run_heads_from_bev,
    visualize_bev,
)


def collapse_bev_feature(
    bev_feature: torch.Tensor, mode: str = "mean", channel: int = 0, normalize: bool = False
) -> torch.Tensor:
    """
    Reduce BEV feature (C,H,W) to (H,W) for visualization.

    mode: mean | sum | single
    """
    if bev_feature.ndim == 4:
        bev_feature = bev_feature[0]
    if bev_feature.ndim != 3:
        raise ValueError(f"Expected BEV feature with 3 dims, got shape {tuple(bev_feature.shape)}")

    feature = bev_feature
    if normalize:
        denom = (feature.max() - feature.min()).clamp(min=1e-6)
        feature = (feature - feature.min()) / denom

    if mode == "mean":
        reduced = feature.mean(dim=0)
    elif mode == "sum":
        reduced = feature.sum(dim=0)
    elif mode == "single":
        if channel < 0 or channel >= feature.shape[0]:
            raise ValueError(f"Channel {channel} out of range for feature with {feature.shape[0]} channels")
        reduced = feature[channel]
    else:
        raise ValueError(f"Unsupported mode '{mode}' (use mean | sum | single)")

    return reduced.cpu()


def save_bev_feature_grid(
    features: Dict[str, torch.Tensor],
    output_path: Path,
    mode: str = "mean",
    channel: int = 0,
    cmap: str = "viridis",
    normalize: bool = False,
) -> None:
    """Visualize BEV features with shared color scale."""
    processed = []
    titles = []
    for title, tensor in features.items():
        if tensor is None:
            continue
        reduced = collapse_bev_feature(tensor, mode=mode, channel=channel, normalize=normalize)
        processed.append(reduced)
        titles.append(title)

    if not processed:
        return

    vmin = min(p.min().item() for p in processed)
    vmax = max(p.max().item() for p in processed)

    fig, axes = plt.subplots(1, len(processed), figsize=(6 * len(processed), 5))
    if len(processed) == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, img, title in zip(axes, processed, titles):
        im = ax.imshow(img.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_axis_off()
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_triplet_with_bevfeat(
    images: Dict[str, Path],
    bev_features: Dict[str, torch.Tensor],
    output_path: Path,
    cmap: str = "viridis",
    normalize: bool = False,
) -> None:
    """Save a grid with visuals (row 1), BEV mean (row 2), and BEV channel 0 (row 3)."""
    titles = list(bev_features.keys())
    columns = []
    for title in titles:
        img_path = images.get(title)
        img = plt.imread(str(img_path)) if img_path is not None and img_path.exists() else None
        feat = bev_features.get(title)
        if feat is None and img is None:
            continue
        columns.append((title, img, feat))

    if not columns:
        return

    # Prepare feature rows
    def reduce_row(mode: str, channel: int = 0) -> List[Optional[torch.Tensor]]:
        reduced_list: List[Optional[torch.Tensor]] = []
        for _, _, feat in columns:
            if feat is None:
                reduced_list.append(None)
                continue
            reduced_list.append(collapse_bev_feature(feat, mode=mode, channel=channel, normalize=normalize))
        return reduced_list

    mean_row = reduce_row("mean")
    ch0_row = reduce_row("single", channel=0)

    # Compute vmin/vmax per feature row
    def row_limits(row: List[Optional[torch.Tensor]]) -> Tuple[float, float]:
        vals = [r for r in row if r is not None]
        if not vals:
            return 0.0, 0.0
        return min(r.min().item() for r in vals), max(r.max().item() for r in vals)

    mean_vmin, mean_vmax = row_limits(mean_row)
    ch0_vmin, ch0_vmax = row_limits(ch0_row)

    n_rows = 1 + 2  # visuals + two feature reductions
    n_cols = len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]  # pragma: no cover - not expected
    if n_cols == 1:
        axes = [[ax] for ax in axes]  # type: ignore[list-item]

    # Row 0: visuals
    for col_idx, (title, img, _) in enumerate(columns):
        ax = axes[0][col_idx]
        if img is not None:
            ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title)

    # Row 1: mean
    for col_idx, reduced in enumerate(mean_row):
        ax = axes[1][col_idx]
        ax.set_axis_off()
        if reduced is None:
            continue
        im = ax.imshow(reduced.numpy(), cmap=cmap, vmin=mean_vmin, vmax=mean_vmax, origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{columns[col_idx][0]} (mean)")

    # Row 2: channel 0
    for col_idx, reduced in enumerate(ch0_row):
        ax = axes[2][col_idx]
        ax.set_axis_off()
        if reduced is None:
            continue
        im = ax.imshow(reduced.numpy(), cmap=cmap, vmin=ch0_vmin, vmax=ch0_vmax, origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{columns[col_idx][0]} (channel 0)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test BEV flow sampling on cached pairs (val split).")
    parser.add_argument("--pairs", type=Path, required=True, help="Frame pairs JSONL.")
    parser.add_argument("--split", type=Path, required=True, help="JSON split with key 'val'.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory with precomputed BEV cache.")
    parser.add_argument("--flow-ckpt", type=Path, required=True, help="Lightning checkpoint for flow matcher.")
    parser.add_argument("--uniad-config", type=Path, required=True, help="UniAD config path (.py).")
    parser.add_argument("--uniad-checkpoint", type=Path, required=True, help="UniAD checkpoint path (.pth).")
    parser.add_argument("--out-dir", type=Path, default=Path("bev_flow_eval"), help="Output directory for visuals/summary.")
    parser.add_argument(
        "--save-gen-bev-dir",
        type=Path,
        default=None,
        help="Optional directory to save generated Bench2Drive BEVs.",
    )
    parser.add_argument("--sample-steps", type=int, default=50, help="Euler sampling steps for flow matching.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of val samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 for clearer per-frame outputs).")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=None,
        help="Score threshold for visualization; defaults to UniAD model setting if omitted.",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization to speed up runs.")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="Force using non-EMA flow weights.")
    parser.add_argument(
        "--viz-bev-features",
        action="store_true",
        help="Save BEV feature heatmaps (channel-reduced) for source/gen/target.",
    )
    parser.add_argument(
        "--bevfeat-mode",
        choices=["mean", "sum", "single"],
        default="mean",
        help="How to reduce channels for BEV feature visualization.",
    )
    parser.add_argument(
        "--bevfeat-channel",
        type=int,
        default=0,
        help="Channel index when bevfeat-mode=single.",
    )
    parser.add_argument(
        "--bevfeat-normalize",
        action="store_true",
        help="Normalize BEV features before reducing for visualization.",
    )
    parser.add_argument("--bevfeat-cmap", type=str, default="viridis", help="Matplotlib cmap for BEV feature viz.")
    parser.set_defaults(use_ema=True)
    return parser.parse_args()


def build_val_loader(
    pairs_path: Path,
    split_path: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int] = None,
) -> Tuple[List[FrameBundle], DataLoader]:
    bundles = load_frame_bundles(pairs_path)
    val_set = load_split(split_path)
    _, val_bundles = split_bundles(bundles, val_set)
    if max_samples is not None:
        val_bundles = val_bundles[:max_samples]

    dataset = PairDataset(val_bundles, cache_dir=cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return val_bundles, dataloader


def load_flow_models(ckpt_path: Path, device: torch.device, use_ema: bool):
    lit_model = FlowMatchingLit.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit_model.eval()
    lit_model.to(device)
    sampler = lit_model.ema_flow_model if use_ema and lit_model.ema_flow_model is not None else lit_model.flow_model
    sampler = sampler.to(device)
    sampler.eval()
    return lit_model, sampler


def load_uniad(config_path: Path, checkpoint_path: Path, device: torch.device):
    model, _ = load_model_and_pipeline(str(config_path), str(checkpoint_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model


def save_triplet_image(md_png: Path, gen_png: Path, tgt_png: Path, output_path: Path) -> None:
    images: List[Tuple[str, Optional[Path]]] = [
        ("MetaDrive (source)", md_png),
        ("Generated Bench2Drive", gen_png),
        ("GT Bench2Drive", tgt_png),
    ]
    loaded = []
    titles = []
    for title, path in images:
        if path is not None and path.exists():
            loaded.append(plt.imread(str(path)))
            titles.append(title)
    if not loaded:
        return
    fig, axes = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 6))
    if len(loaded) == 1:
        axes = [axes]  # type: ignore[list-item]
    for ax, img, title in zip(axes, loaded, titles):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_json_summary(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    val_bundles, dataloader = build_val_loader(
        args.pairs,
        args.split,
        args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"[data] Loaded {len(val_bundles)} val bundles")

    lit_model, sampler_model = load_flow_models(args.flow_ckpt, device, use_ema=args.use_ema)
    uni_model = load_uniad(args.uniad_config, args.uniad_checkpoint, device)
    score_thresh = (
        args.score_thresh
        if args.score_thresh is not None
        else getattr(getattr(uni_model, "track_base", None), "filter_score_thresh", 0.2)
    )

    out_dir = args.out_dir
    gen_bev_dir = args.save_gen_bev_dir

    records: List[Dict] = []
    progress = tqdm(dataloader, desc="Sampling BEVs", total=len(dataloader))

    for batch in progress:
        if batch is None:
            continue
        bundles, md_bev, b2d_bev = batch
        md_bev = md_bev.to(device)
        b2d_bev = b2d_bev.to(device)

        # Sample Bench2Drive BEV from MetaDrive BEV.
        gen_bev = sampler_model.sample(md_bev, sample_steps=args.sample_steps)
        gen_bchw = lit_model.flow_model.reshape_bev(gen_bev)
        tgt_bchw = lit_model.flow_model.reshape_bev(b2d_bev)

        batch_size = gen_bchw.shape[0]
        for idx in range(batch_size):
            bundle = bundles[idx]
            base_out = out_dir / bundle.scenario
            base_out.mkdir(parents=True, exist_ok=True)

            sample_gen = gen_bchw[idx : idx + 1]
            sample_tgt = tgt_bchw[idx : idx + 1]
            sample_md = lit_model.flow_model.reshape_bev(md_bev[idx : idx + 1])
            # sample_gen = sample_md + 10 * (sample_gen - sample_md)

            mse = torch.mean((sample_gen - sample_tgt) ** 2).item()
            mse_ori = torch.mean((sample_gen - sample_md) ** 2).item()
            cos = torch.nn.functional.cosine_similarity(
                sample_gen.flatten(1), sample_tgt.flatten(1), dim=1
            ).mean().item()

            gen_bev_path = None
            if gen_bev_dir is not None:
                gen_bev_path = gen_bev_dir / "bench2drive" / bundle.scenario / f"{bundle.frame}.pt"
                gen_bev_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"bev": sample_gen.cpu(), "scenario": bundle.scenario, "frame": bundle.frame},
                    gen_bev_path,
                )

            md_png = base_out / f"{bundle.frame}_md.png"
            gen_png = base_out / f"{bundle.frame}_gen.png"
            tgt_png = base_out / f"{bundle.frame}_b2d.png"
            triplet_png = base_out / f"{bundle.frame}_triplet.png"
            bevfeat_png = base_out / f"{bundle.frame}_bevfeat.png"
            combined_png = base_out / f"{bundle.frame}_triplet_bevfeat.png"

            gen_seg = gen_occ = tgt_seg = tgt_occ = None
            if not args.no_viz:
                try:
                    _, md_seg, md_occ = run_heads_from_bev(sample_md, uni_model, score_thresh=score_thresh)
                except Exception as e:  # pragma: no cover - visualization convenience
                    print(f"[warn] heads(md) failed for {bundle.scenario}/{bundle.frame}: {e}")
                try:
                    _, gen_seg, gen_occ = run_heads_from_bev(sample_gen, uni_model, score_thresh=score_thresh)
                except Exception as e:  # pragma: no cover - visualization convenience
                    print(f"[warn] heads(gen) failed for {bundle.scenario}/{bundle.frame}: {e}")
                try:
                    _, tgt_seg, tgt_occ = run_heads_from_bev(sample_tgt, uni_model, score_thresh=score_thresh)
                except Exception as e:  # pragma: no cover - visualization convenience
                    print(f"[warn] heads(gt) failed for {bundle.scenario}/{bundle.frame}: {e}")

                visualize_bev(
                    sample_md,
                    uni_model,
                    score_thresh=score_thresh,
                    output_png=md_png,
                    seg_data=md_seg,
                    occ_data=md_occ,
                )
                visualize_bev(
                    sample_gen,
                    uni_model,
                    score_thresh=score_thresh,
                    output_png=gen_png,
                    seg_data=gen_seg,
                    occ_data=gen_occ,
                )
                visualize_bev(
                    sample_tgt,
                    uni_model,
                    score_thresh=score_thresh,
                    output_png=tgt_png,
                    seg_data=tgt_seg,
                    occ_data=tgt_occ,
                )
                if args.viz_bev_features:
                    save_bev_feature_grid(
                        {
                            "MetaDrive (source)": sample_md,
                            "Generated Bench2Drive": sample_gen,
                            "GT Bench2Drive": sample_tgt,
                        },
                        bevfeat_png,
                        mode=args.bevfeat_mode,
                        channel=args.bevfeat_channel,
                        cmap=args.bevfeat_cmap,
                        normalize=args.bevfeat_normalize,
                    )
                    save_triplet_with_bevfeat(
                        {
                            "MetaDrive (source)": md_png,
                            "Generated Bench2Drive": gen_png,
                            "GT Bench2Drive": tgt_png,
                        },
                        {
                            "MetaDrive (source)": sample_md,
                            "Generated Bench2Drive": sample_gen,
                            "GT Bench2Drive": sample_tgt,
                        },
                        combined_png,
                        cmap=args.bevfeat_cmap,
                        normalize=args.bevfeat_normalize,
                    )
                else:
                    save_triplet_image(md_png, gen_png, tgt_png, triplet_png)

            records.append(
                {
                    "scenario": bundle.scenario,
                    "frame": bundle.frame,
                    "mse": mse,
                    "cosine": cos,
                    "gen_bev_path": str(gen_bev_path) if gen_bev_path else None,
                    "md_png": str(md_png) if md_png.exists() else None,
                    "gen_png": str(gen_png) if gen_png.exists() else None,
                    "tgt_png": str(tgt_png) if tgt_png.exists() else None,
                    "triplet_png": str(combined_png if args.viz_bev_features else triplet_png)
                    if (combined_png if args.viz_bev_features else triplet_png).exists()
                    else None,
                    "bevfeat_png": str(bevfeat_png) if bevfeat_png.exists() else None,
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json_summary(out_dir / "summary.json", records)
    print(f"[done] Wrote summary for {len(records)} samples to {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()

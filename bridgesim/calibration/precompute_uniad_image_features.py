#!/usr/bin/env python3
"""
Precompute UniAD image-backbone features for Bench2Drive/MetaDrive image pairs.

The script:
1) Loads a UniAD model and builds the inference-only preprocessing pipeline
   (mirroring leaderboard/offline_evaluator_md_1.py but skipping file loaders).
2) Iterates over the pairs JSONL produced by build_bench2drive_metadrive_pairs.py.
3) For each (MetaDrive, Bench2Drive) image pair, applies the UniAD pipeline,
   runs the image backbone + neck to obtain multi-level features, and caches
   them into an HDF5 file per scenario.

Groups inside each HDF5 file follow: {frame}/{camera}/{kind}, where kind is
either "metadrive" or "bench2drive", and datasets feat0, feat1, ... store the
per-level feature maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELZOO_BENCH2DRIVE = REPO_ROOT / "modelzoo" / "bench2drive"
sys.path.insert(0, str(MODELZOO_BENCH2DRIVE))

from bridgesim.modelzoo.bench2drive.mmcv import Config
from bridgesim.modelzoo.bench2drive.mmcv.datasets.pipelines import Compose
from bridgesim.modelzoo.bench2drive.mmcv.models import build_model
from bridgesim.modelzoo.bench2drive.mmcv.parallel.collate import collate as mm_collate_to_batch_form
from bridgesim.modelzoo.bench2drive.mmcv.utils import load_checkpoint


@dataclass
class PairEntry:
    scenario: str
    frame: str
    camera: str
    bench2drive_path: Path
    metadrive_path: Path


def load_pairs(pairs_path: Path) -> List[PairEntry]:
    """Load image pairs from a JSONL produced by build_bench2drive_metadrive_pairs.py."""
    entries: List[PairEntry] = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            scenario = payload.get("scenario") or payload.get("scenario_name")
            if not scenario:
                raise ValueError(f"Missing 'scenario' key on line {line_no}")
            pairs = payload.get("pairs") or []
            for pair in pairs:
                try:
                    entries.append(
                        PairEntry(
                            scenario=scenario,
                            frame=str(pair["frame"]),
                            camera=str(pair["camera"]),
                            bench2drive_path=Path(pair["bench2drive_path"]),
                            metadrive_path=Path(pair["metadrive_path"]),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"Missing field {exc} in pair on line {line_no}") from exc
    return entries


def find_bench2drive_zoo_root(cfg_path: Path) -> Path:
    """Get the Bench2Drive modelzoo root directory."""
    # First check if config is in the new modelzoo location
    for parent in cfg_path.resolve().parents:
        if parent.name == "bench2drive":
            return parent
    # Fallback to modelzoo/bench2drive
    if MODELZOO_BENCH2DRIVE.exists():
        return MODELZOO_BENCH2DRIVE
    raise RuntimeError(f"Failed to locate Bench2Drive modelzoo root from {cfg_path}")


def load_model_and_pipeline(cfg_path: Path, checkpoint_path: Path, device: torch.device):
    """Load UniAD model and an inference pipeline without file loading."""
    cfg = Config.fromfile(str(cfg_path))
    bench_zoo_root = find_bench2drive_zoo_root(cfg_path)

    # Fix anchor_info_path relative to modelzoo/bench2drive
    motion_head = cfg.model.get("motion_head") if hasattr(cfg, "model") else None
    if motion_head and "anchor_info_path" in motion_head:
        anchor_path = Path(motion_head["anchor_info_path"])
        if not anchor_path.is_absolute():
            anchor_path = bench_zoo_root / anchor_path
        cfg.model["motion_head"]["anchor_info_path"] = str(anchor_path)

    # Import plugins if requested
    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir:
            plugin_dir_path = Path(plugin_dir)
            if not plugin_dir_path.is_absolute():
                plugin_dir_path = bench_zoo_root / plugin_dir_path
            sys.path.insert(0, str(plugin_dir_path.parent))
            module_path = ".".join(plugin_dir_path.with_suffix("").parts[-2:])
            __import__(module_path)

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    _ = load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=True)
    model.to(device)
    model.eval()

    # Build inference-only pipeline without file loaders
    inference_only_pipeline_cfg: List[Dict] = []
    for pipeline_cfg in cfg.inference_only_pipeline:
        if pipeline_cfg["type"] in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles"):
            continue
        inference_only_pipeline_cfg.append(pipeline_cfg)
    pipeline = Compose(inference_only_pipeline_cfg)
    return model, pipeline


def build_dummy_meta() -> Dict[str, object]:
    """Provide minimal meta keys required by the pipeline."""
    return {
        "timestamp": 0.0,
        "l2g_r_mat": np.eye(3, dtype=np.float32),
        "l2g_t": np.zeros(3, dtype=np.float32),
        "command": 0,
    }


def run_backbone_from_batch(batch: Dict, model, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run backbone+neck on a collated batch produced by mm_collate_to_batch_form."""
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
                reshaped.append(f.view(B, N, c, h, w).detach().cpu())
            return reshaped

        backbone_feats = _reshape(backbone_feats)
        neck_feats = _reshape(neck_feats)
    return backbone_feats, neck_feats


def cache_group_name(frame: str, camera: str, kind: str, stage: str | None = None) -> str:
    frame_key = frame if frame.isdigit() is False else f"{int(frame):06d}"
    suffix = f"/{stage}" if stage else ""
    return f"{frame_key}/{camera}/{kind}{suffix}"


def save_features(
    h5_file: h5py.File,
    group_name: str,
    feats: List[torch.Tensor],
    *,
    source_path: Path,
    overwrite: bool,
) -> None:
    if group_name in h5_file:
        if not overwrite:
            return
        del h5_file[group_name]
    grp = h5_file.create_group(group_name)
    for idx, feat in enumerate(feats):
        grp.create_dataset(f"feat{idx}", data=feat.numpy())
    grp.attrs["source_path"] = str(source_path)


def process_pairs(
    pairs: Iterable[PairEntry],
    pipeline: Compose,
    model,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    ) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    open_files: Dict[str, h5py.File] = {}

    def get_cache_file(scenario: str) -> h5py.File:
        if scenario not in open_files:
            path = cache_root / f"{scenario}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            open_files[scenario] = h5py.File(path, "a")
        return open_files[scenario]

    class PairDataset(Dataset):
        def __init__(self, entries: List[PairEntry], cache_dir: Path, pipe: Compose, overwrite_flag: bool):
            self.entries = entries
            self.cache_dir = cache_dir
            self.pipe = pipe
            self.overwrite = overwrite_flag

        def __len__(self) -> int:
            return len(self.entries)

        def _is_cached(self, pair: PairEntry) -> bool:
            if self.overwrite:
                return False
            h5_path = self.cache_dir / f"{pair.scenario}.h5"
            if not h5_path.is_file():
                return False
            try:
                with h5py.File(h5_path, "r") as f:
                    md_groups = [
                        cache_group_name(pair.frame, pair.camera, "metadrive", stage) for stage in ("backbone", "neck")
                    ]
                    b2d_groups = [
                        cache_group_name(pair.frame, pair.camera, "bench2drive", stage)
                        for stage in ("backbone", "neck")
                    ]
                    return all(g in f for g in md_groups + b2d_groups)
            except OSError:
                return False

        def _process_image(self, img_path: Path):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            sample = build_dummy_meta()
            sample["img"] = [img]
            return self.pipe(sample)

        def __getitem__(self, idx: int):
            pair = self.entries[idx]
            if not pair.metadrive_path.is_file() or not pair.bench2drive_path.is_file():
                return None
            if self._is_cached(pair):
                return None
            md_proc = self._process_image(pair.metadrive_path)
            b2d_proc = self._process_image(pair.bench2drive_path)
            if md_proc is None or b2d_proc is None:
                return None
            return pair, md_proc, b2d_proc

    def _pair_collate(batch_items: List):
        batch_items = [b for b in batch_items if b is not None]
        if not batch_items:
            return None
        pairs_batch, md_samples, b2d_samples = zip(*batch_items)
        md_batch = mm_collate_to_batch_form(md_samples, samples_per_gpu=len(md_samples))
        b2d_batch = mm_collate_to_batch_form(b2d_samples, samples_per_gpu=len(b2d_samples))
        return list(pairs_batch), md_batch, b2d_batch

    try:
        dataset = PairDataset(list(pairs), cache_root, pipeline, overwrite)
        pair_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_pair_collate,
        )

        for batch_data in tqdm(pair_loader, desc="Extracting features", dynamic_ncols=True):
            if batch_data is None:
                continue
            batch_pairs, md_batch, b2d_batch = batch_data

            md_backbone, md_neck = run_backbone_from_batch(md_batch, model, device)
            b2d_backbone, b2d_neck = run_backbone_from_batch(b2d_batch, model, device)

            for idx, pair in enumerate(batch_pairs):
                h5_f = get_cache_file(pair.scenario)

                def pick_sample(feat_list: List[torch.Tensor]) -> List[torch.Tensor]:
                    return [feat[idx : idx + 1] for feat in feat_list]

                save_features(
                    h5_f,
                    cache_group_name(pair.frame, pair.camera, "metadrive", "backbone"),
                    pick_sample(md_backbone),
                    source_path=pair.metadrive_path,
                    overwrite=overwrite,
                )
                save_features(
                    h5_f,
                    cache_group_name(pair.frame, pair.camera, "metadrive", "neck"),
                    pick_sample(md_neck),
                    source_path=pair.metadrive_path,
                    overwrite=overwrite,
                )
                save_features(
                    h5_f,
                    cache_group_name(pair.frame, pair.camera, "bench2drive", "backbone"),
                    pick_sample(b2d_backbone),
                    source_path=pair.bench2drive_path,
                    overwrite=overwrite,
                )
                save_features(
                    h5_f,
                    cache_group_name(pair.frame, pair.camera, "bench2drive", "neck"),
                    pick_sample(b2d_neck),
                    source_path=pair.bench2drive_path,
                    overwrite=overwrite,
                )
    finally:
        for f in open_files.values():
            f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute UniAD image-backbone features for Bench2Drive/MetaDrive pairs."
    )
    parser.add_argument("--pairs", type=Path, default=Path("/closed-loop-e2e/Bench2Drive/converted-base/pairs.jsonl"), help="Path to pairs JSONL (from build_bench2drive_metadrive_pairs.py).")
    parser.add_argument("--config", type=Path, default=MODELZOO_BENCH2DRIVE / "adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py", help="Path to UniAD config (.py).")
    parser.add_argument("--checkpoint", type=Path, default=Path("/closed-loop-e2e/weights/uniad_base_b2d.pth"), help="Path to UniAD checkpoint (.pth).")
    parser.add_argument("--cache-root", type=Path, default=Path("/closed-loop-e2e/Bench2Drive/base-uniad-feature-pairs"), help="Directory to store HDF5 feature files.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device for feature extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of pairs to process together for feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for image preprocessing.",
    )
    parser.add_argument("--overwrite", action="store_true", default=True, help="Recompute and overwrite existing feature groups.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    pairs = load_pairs(args.pairs)
    model, pipeline = load_model_and_pipeline(args.config, args.checkpoint, device)

    process_pairs(
        pairs=pairs,
        pipeline=pipeline,
        model=model,
        cache_root=args.cache_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()



# Example usage:
# python bridgesim/calibration/precompute_uniad_image_features.py \
#   --pairs /path/to/pairs.jsonl \
#   --config bridgesim/modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
#   --checkpoint /path/to/uniad_base_b2d.pth \
#   --cache-root /path/to/feature-pairs \
#   --device cuda --overwrite

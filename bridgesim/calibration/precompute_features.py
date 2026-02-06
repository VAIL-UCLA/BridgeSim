from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import h5py

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm

from calibration.dataloader import RasterRealPairDataset, feature_cache_file, feature_cache_group


def _collate(batch):
    """
    Keep metadata as a list while stacking real/rendered images.
    """
    meta = []
    real_imgs = []
    rendered_imgs = []
    for item in batch:
        meta.append(
            {
                "log_id": item["log_id"],
                "frame_idx": item["frame_idx"],
                "cam_name": item["cam_name"],
            }
        )
        real_imgs.append(item["real_image"])
        rendered_imgs.append(item["rendered_image"])
    real_tensor = torch.stack(real_imgs, dim=0)
    rendered_tensor = torch.stack(rendered_imgs, dim=0)
    return meta, real_tensor, rendered_tensor


def _tokens_to_map(x: torch.Tensor, height: int, width: int, patch_size: int = 16) -> torch.Tensor:
    gh, gw = height // patch_size, width // patch_size
    extra = x.shape[1] - gh * gw
    patches = x[:, extra:]
    return patches.transpose(1, 2).reshape(x.shape[0], -1, gh, gw)


def preprocess_and_cache(
    manifest_path: Path,
    log_root: Path,
    sensor_root: Path,
    rendered_root: Path,
    cache_root: Path,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(backbone_name, trust_remote_code=True).to(device)
    model.eval()

    dataset = RasterRealPairDataset(
        manifest_path=manifest_path,
        log_root=log_root,
        sensor_root=sensor_root,
        rendered_root=rendered_root,
        load_images=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=_collate,
    )

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    open_files: dict[str, h5py.File] = {}

    def get_cache_file(log_id: str) -> Optional[h5py.File]:
        if log_id not in open_files:
            path = feature_cache_file(cache_root=cache_root, log_id=log_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                open_files[log_id] = h5py.File(path, "a")
            except OSError:
                path.unlink(missing_ok=True)
                open_files[log_id] = h5py.File(path, "a")
        return open_files[log_id]

    with torch.no_grad():
        try:
            for meta_batch, real_imgs, rendered_imgs in tqdm(
                loader, desc="Caching features", dynamic_ncols=True
            ):
                keep_indices = []
                for i, meta in enumerate(meta_batch):
                    cache_file = get_cache_file(meta["log_id"])
                    if cache_file is None:
                        continue
                    real_group = feature_cache_group(meta["frame_idx"], meta["cam_name"], "real")
                    rendered_group = feature_cache_group(meta["frame_idx"], meta["cam_name"], "rendered")
                    if real_group in cache_file and rendered_group in cache_file:
                        continue
                    keep_indices.append(i)

                if not keep_indices:
                    continue

                meta_batch = [meta_batch[i] for i in keep_indices]
                real_imgs = real_imgs[keep_indices].to(device, non_blocking=True)
                rendered_imgs = rendered_imgs[keep_indices].to(device, non_blocking=True)

                real_tokens = model(pixel_values=real_imgs)["last_hidden_state"]
                rendered_tokens = model(pixel_values=rendered_imgs)["last_hidden_state"]

                _, _, img_h, img_w = real_imgs.shape
                real_maps = _tokens_to_map(real_tokens, img_h, img_w)
                rendered_maps = _tokens_to_map(rendered_tokens, img_h, img_w)

                real_feats = tuple(feat.cpu() for feat in (real_tokens, real_maps))
                rendered_feats = tuple(feat.cpu() for feat in (rendered_tokens, rendered_maps))
                for i, meta in enumerate(meta_batch):
                    for kind, feats in (("real", real_feats), ("rendered", rendered_feats)):
                        per_sample_feats = tuple(level[i] for level in feats)
                        cache_file = get_cache_file(meta["log_id"])
                        if cache_file is None:
                            continue
                        group_name = feature_cache_group(meta["frame_idx"], meta["cam_name"], kind)
                        if group_name in cache_file:
                            del cache_file[group_name]
                        grp = cache_file.create_group(group_name)
                        for idx, feat in enumerate(per_sample_feats):
                            grp.create_dataset(f"feat{idx}", data=feat.numpy())
        finally:
            for f in open_files.values():
                f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and cache image backbone features.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL.")
    parser.add_argument("--log-root", type=Path, required=True, help="Root containing log pickle files.")
    parser.add_argument("--sensor-root", type=Path, required=True, help="Root containing real sensor blobs.")
    parser.add_argument("--rendered-root", type=Path, required=True, help="Root containing rendered sensor blobs.")
    parser.add_argument("--cache-root", type=Path, required=True, help="Destination directory for cached features.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="facebook/dinov3-vith16plus-pretrain-lvd1689m",
        help="Backbone name passed to transformers.AutoModel.from_pretrained.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_and_cache(
        manifest_path=args.manifest,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        rendered_root=args.rendered_root,
        cache_root=args.cache_root,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

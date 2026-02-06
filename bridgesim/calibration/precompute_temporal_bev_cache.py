#!/usr/bin/env python3
"""
Precompute temporal UniAD BEV features and cache them per current frame.

Behavior
- Loads frame bundles that already include history lists (prev_frames/prev_*).
- Builds temporal clips following train_temporal_bev history logic
  (relative can_bus, prev_bev_exists flags, interval filtering).
- Runs UniAD temporal get_bevs without sigma jitter and saves only the BEV
  corresponding to the current frame.
- Cache layout matches the original per-frame script:
  cache_dir/<domain>/<scenario>/<frame>.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_uniad_bev import (
    load_model_and_pipeline,
    load_infos,
    _match_info_entry,
    invert_pose,
    visualize_bev,
    run_heads_from_bev,
)

try:
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form
except ImportError:
    from mmcv.parallel import collate as mm_collate_to_batch_form  # type: ignore


CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
CAM_DIR_TO_NAME = {
    "rgb_front": "CAM_FRONT",
    "rgb_front_left": "CAM_FRONT_LEFT",
    "rgb_front_right": "CAM_FRONT_RIGHT",
    "rgb_back": "CAM_BACK",
    "rgb_back_left": "CAM_BACK_LEFT",
    "rgb_back_right": "CAM_BACK_RIGHT",
}
CAM_NAME_TO_DIR = {v: k for k, v in CAM_DIR_TO_NAME.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute temporal BEV features for MetaDrive/Bench2Drive.")
    parser.add_argument("--pairs", type=Path, required=True, help="Frame pairs JSONL with history.")
    parser.add_argument("--info-pkls", nargs="+", required=True, help="Info pkl(s) from prepare_B2D_nautilus.py.")
    parser.add_argument("--config", type=Path, required=True, help="UniAD config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="UniAD checkpoint path.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Output directory for cached BEVs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--default-command", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--history-length", type=int, default=2, help="Number of past frames (queue_length-1).")
    parser.add_argument("--sample-interval", type=int, default=5, help="Expected stride between frames.")
    parser.add_argument(
        "--allow-short-history",
        action="store_true",
        default=True,
        help="Allow fewer past frames at episode starts (enabled by default).",
    )
    parser.add_argument(
        "--no-allow-short-history",
        dest="allow_short_history",
        action="store_false",
        help="Require full history_length history.",
    )
    parser.add_argument("--viz", action="store_true", help="Enable BEV visualization (rank 0 only).")
    parser.add_argument("--viz-dir", type=Path, default=Path("bev_viz_temporal_cache"))
    parser.add_argument("--viz-score-thresh", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of pairs to process.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device. Defaults to cuda:LOCAL_RANK or cpu.")
    parser.add_argument("--distributed", action="store_true", help="Enable torch.distributed for multi-GPU sharding.")
    parser.add_argument("--local-rank", type=int, default=-1, help="torch.distributed launch utility compatibility.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class FrameBundle:
    scenario: str
    frame: str
    bench2drive: Dict[str, str]
    metadrive: Dict[str, str]
    prev_frames: List[str]
    prev_bench2drive: List[Dict[str, str]]
    prev_metadrive: List[Dict[str, str]]


def load_frame_bundles(path: Path) -> List[FrameBundle]:
    bundles: List[FrameBundle] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                bundles.append(
                    FrameBundle(
                        scenario=str(payload["scenario"]),
                        frame=str(payload["frame"]),
                        bench2drive=payload["bench2drive"],
                        metadrive=payload["metadrive"],
                        prev_frames=[str(x) for x in payload.get("prev_frames") or []],
                        prev_bench2drive=payload.get("prev_bench2drive") or [],
                        prev_metadrive=payload.get("prev_metadrive") or [],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} on line {line_no}") from exc
    return bundles


def _timestamp_from_frame(frame_id: str, fps: float) -> float:
    try:
        return int(frame_id) / fps
    except ValueError:
        return 0.0


def _load_multiview_images(cam_paths: Dict[str, str]) -> Optional[Tuple[List[np.ndarray], List[str]]]:
    import cv2

    imgs: List[np.ndarray] = []
    filenames: List[str] = []
    for cam_name in CAM_NAMES:
        dir_name = CAM_NAME_TO_DIR.get(cam_name)
        if dir_name is None or dir_name not in cam_paths:
            return None
        path = Path(cam_paths[dir_name])
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        imgs.append(img)
        filenames.append(str(path))
    return imgs, filenames


def build_single_frame_sample(
    scenario: str,
    frame_id: str,
    cam_paths: Dict[str, str],
    info_index: Dict[Tuple[str, int], Dict],
    default_command: int,
    fps: float,
) -> Optional[Dict]:
    from pyquaternion import Quaternion  # local import

    fake_bundle = FrameBundle(scenario, frame_id, {}, {}, [], [], [])
    info_entry = _match_info_entry(fake_bundle, info_index)  # type: ignore[arg-type]
    if info_entry is None:
        return None
    imgs_and_names = _load_multiview_images(cam_paths)
    if imgs_and_names is None:
        return None
    imgs, filenames = imgs_and_names

    ego_yaw = float(np.nan_to_num(info_entry["ego_yaw"], nan=np.pi / 2))
    ego_translation = np.array(info_entry["ego_translation"], dtype=np.float32)
    ego_vel = np.array(info_entry["ego_vel"], dtype=np.float32)
    ego_accel = np.array(info_entry["ego_accel"], dtype=np.float32)
    ego_rotation_rate = np.array(info_entry["ego_rotation_rate"], dtype=np.float32)
    lidar2ego = np.array(info_entry["sensors"]["LIDAR_TOP"]["lidar2ego"], dtype=np.float32)

    lidar2img_list: List[np.ndarray] = []
    lidar2cam_list: List[np.ndarray] = []
    cam_intrinsics: List[np.ndarray] = []
    for cam in CAM_NAMES:
        cam_info = info_entry["sensors"][cam]
        intrinsic = np.array(cam_info["intrinsic"], dtype=np.float32)
        intrinsic_pad = np.eye(4, dtype=np.float32)
        intrinsic_pad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        cam2ego = np.array(cam_info["cam2ego"], dtype=np.float32)
        lidar2cam = invert_pose(cam2ego) @ lidar2ego
        lidar2img = intrinsic_pad @ lidar2cam
        lidar2img_list.append(lidar2img)
        lidar2cam_list.append(lidar2cam)
        cam_intrinsics.append(intrinsic_pad)

    ego2world = np.eye(4, dtype=np.float32)
    ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_yaw).rotation_matrix.astype(np.float32)
    ego2world[0:3, 3] = ego_translation
    lidar2global = ego2world @ lidar2ego
    l2g_r_mat = lidar2global[0:3, 0:3]
    l2g_t = lidar2global[0:3, 3]

    yaw_deg = ego_yaw / np.pi * 180.0
    rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_yaw))
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = ego_translation
    can_bus[3:7] = rotation
    can_bus[7:10] = ego_vel
    can_bus[10:13] = ego_accel
    can_bus[13:16] = ego_rotation_rate
    can_bus[16] = ego_yaw
    can_bus[17] = yaw_deg

    command = info_entry.get("command_near", default_command)
    if command < 0:
        command = default_command
    command = int(command) - 1  # UniAD expects 0-indexed
    timestamp = info_entry.get("timestamp", _timestamp_from_frame(frame_id, fps))
    frame_idx = int(info_entry["frame_idx"])
    stacked = np.stack(imgs, axis=-1)
    sample = {
        "img": imgs,
        "lidar2img": np.stack(lidar2img_list, axis=0),
        "lidar2cam": np.stack(lidar2cam_list, axis=0),
        "timestamp": timestamp,
        "l2g_r_mat": l2g_r_mat,
        "l2g_t": l2g_t,
        "command": command,
        "ego_fut_cmd": np.array([command], dtype=np.int64),
        "scene_token": info_entry["folder"],
        "frame_idx": frame_idx,
        "can_bus": can_bus,
        "folder": info_entry["folder"],
        "filename": filenames,
        "ori_shape": stacked.shape,
        "img_shape": stacked.shape,
        "pad_shape": stacked.shape,
        "scale_factor": np.ones(4, dtype=np.float32),
        "camera_intrinsics": np.stack(cam_intrinsics, axis=0),
        "cam2imgs": np.stack(cam_intrinsics, axis=0),
    }
    return sample


def build_sequence(
    bundle: FrameBundle,
    domain: str,
    info_index: Dict[Tuple[str, int], Dict],
    default_command: int,
    fps: float,
    history_length: int,
    sample_interval: int,
    allow_short_history: bool,
) -> Optional[List[Dict]]:
    frames: List[Tuple[str, Dict[str, str], int]] = []
    prev_list = getattr(bundle, f"prev_{domain}")
    prev_frames = bundle.prev_frames or []
    if len(prev_frames) != len(prev_list):
        prev_frames = prev_frames[: len(prev_list)]
    for idx, (frame_id, cam_paths) in enumerate(zip(prev_frames, prev_list)):
        frames.append((frame_id, cam_paths, idx))
    frames.append((bundle.frame, getattr(bundle, domain), len(frames)))

    if len(frames) < history_length + 1 and not allow_short_history:
        return None
    frames = frames[-(history_length + 1) :]

    filtered: List[Tuple[str, Dict[str, str]]] = []
    last_num: Optional[int] = None
    for fid, cam_paths, _idx in frames:
        if last_num is None:
            filtered.append((fid, cam_paths))
            try:
                last_num = int(fid)
            except ValueError:
                last_num = None
            continue
        try:
            cur_num = int(fid)
        except ValueError:
            filtered.append((fid, cam_paths))
            last_num = None
            continue
        if cur_num - last_num == sample_interval or allow_short_history:
            filtered.append((fid, cam_paths))
            last_num = cur_num
            continue
        if not allow_short_history:
            return None
        filtered.append((fid, cam_paths))
        last_num = cur_num

    samples: List[Dict] = []
    for fid, cam_map in filtered:
        s = build_single_frame_sample(bundle.scenario, fid, cam_map, info_index, default_command, fps)
        if s is None:
            if allow_short_history and fid != bundle.frame:
                continue
            return None
        samples.append(s)
    if not samples:
        return None

    prev_pos = None
    prev_yaw_deg = None
    for idx, sample in enumerate(samples):
        meta = sample
        if idx == 0:
            meta["prev_bev_exists"] = False
            prev_pos = meta["can_bus"][:3].copy()
            prev_yaw_deg = float(meta["can_bus"][17])
            meta["can_bus"] = meta["can_bus"].copy()
            meta["can_bus"][:3] = 0
            meta["can_bus"][17] = 0
        else:
            meta["prev_bev_exists"] = True
            cur_pos = meta["can_bus"][:3].copy()
            cur_yaw_deg = float(meta["can_bus"][17])
            rel = meta["can_bus"].copy()
            rel[:3] = cur_pos - prev_pos  # type: ignore[operator]
            rel[17] = cur_yaw_deg - prev_yaw_deg  # type: ignore[operator]
            meta["can_bus"] = rel
            prev_pos = cur_pos
            prev_yaw_deg = cur_yaw_deg
    return samples


class TemporalPairDataset(Dataset):
    def __init__(
        self,
        bundles: List[FrameBundle],
        info_index: Dict[Tuple[str, int], Dict],
        default_command: int,
        fps: float,
        history_length: int,
        sample_interval: int,
        allow_short_history: bool,
    ):
        self.bundles = bundles
        self.info_index = info_index
        self.default_command = default_command
        self.fps = fps
        self.history_length = history_length
        self.sample_interval = sample_interval
        self.allow_short_history = allow_short_history

    def __len__(self) -> int:
        return len(self.bundles)

    def __getitem__(self, idx: int):
        bundle = self.bundles[idx]
        md_seq = build_sequence(
            bundle,
            "metadrive",
            self.info_index,
            self.default_command,
            self.fps,
            self.history_length,
            self.sample_interval,
            self.allow_short_history,
        )
        b2d_seq = build_sequence(
            bundle,
            "bench2drive",
            self.info_index,
            self.default_command,
            self.fps,
            self.history_length,
            self.sample_interval,
            self.allow_short_history,
        )
        if md_seq is None or b2d_seq is None:
            return None
        t = min(len(md_seq), len(b2d_seq))
        md_seq = md_seq[-t:]
        b2d_seq = b2d_seq[-t:]
        return bundle, md_seq, b2d_seq


def collate_fn(batch: List):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    bundles, md_seqs, b2d_seqs = zip(*batch)
    min_t = min(len(seq) for seq in md_seqs)
    md_seqs = [seq[-min_t:] for seq in md_seqs]
    b2d_seqs = [seq[-min_t:] for seq in b2d_seqs]
    return list(bundles), md_seqs, b2d_seqs


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, data in batch.items():
        if key == "img_metas":
            continue
        if torch.is_tensor(data[0]):
            data[0] = data[0].to(device, non_blocking=True)
    return batch


def _unwrap_img_metas(img_metas, batch_size: int):
    if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
        img_metas = img_metas[0]

    if isinstance(img_metas, (list, tuple)):
        tmp = []
        for m in img_metas:
            if hasattr(m, "data"):
                tmp.append(m.data[0])
            else:
                tmp.append(m)
        img_metas = tmp
    else:
        if hasattr(img_metas, "data"):
            img_metas = img_metas.data
        img_metas = img_metas if isinstance(img_metas, list) else [img_metas]

    if len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
        img_metas = list(img_metas[0])

    assert len(img_metas) == batch_size, f"Unexpected img_metas length {len(img_metas)} vs batch {batch_size}"
    return img_metas


def run_get_bevs_sequence_batch(model, pipeline, seqs: List[List[Dict]], device: torch.device):
    if not seqs:
        return None
    time_steps = len(seqs[0])
    batch_size = len(seqs)
    prev_bev = None
    bev_out = None
    for t in range(time_steps):
        processed_list = [pipeline(seq[t]) for seq in seqs]
        batch = mm_collate_to_batch_form(processed_list, samples_per_gpu=batch_size)
        batch = move_to_device(batch, device)

        imgs = batch["img"][0] if isinstance(batch["img"], (list, tuple)) else batch["img"]
        if imgs.dim() == 4 and batch_size == 1:
            imgs = imgs.unsqueeze(0)
        img_metas = _unwrap_img_metas(batch["img_metas"], batch_size)
        with torch.no_grad():
            bev_out, prev_bev = model.get_bevs(imgs, img_metas, prev_bev)
    return bev_out


def save_bev(bev: torch.Tensor, bundle: FrameBundle, domain: str, cache_dir: Path) -> None:
    out_dir = cache_dir / domain / bundle.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bundle.frame}.pt"
    torch.save({"bev": bev.cpu(), "scenario": bundle.scenario, "frame": bundle.frame}, out_path)


def visualize_with_heads(bev: torch.Tensor, uni_model, output_png: Path, score_thresh: float, domain_label: str):
    """Run UniAD heads on BEV and save visualization (best-effort)."""
    try:
        _, seg_out, occ_out = run_heads_from_bev(bev, uni_model, score_thresh=score_thresh)
    except Exception as e:
        print(f"[viz] Heads failed for {domain_label}: {e}")
        seg_out, occ_out = None, None
    visualize_bev(
        bev,
        uni_model,
        score_thresh=score_thresh,
        output_png=output_png,
        seg_data=seg_out,
        occ_data=occ_out,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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

    uni_model, pipeline = load_model_and_pipeline(str(args.config), str(args.checkpoint))
    for p in uni_model.parameters():
        p.requires_grad_(False)
    uni_model.eval().to(device)

    info_index = load_infos([Path(p) for p in args.info_pkls])

    bundles = load_frame_bundles(args.pairs)
    bundles = bundles[rank::world_size]
    if args.max_samples is not None:
        bundles = bundles[: args.max_samples]

    dataset = TemporalPairDataset(
        bundles,
        info_index=info_index,
        default_command=args.default_command,
        fps=args.fps,
        history_length=args.history_length,
        sample_interval=args.sample_interval,
        allow_short_history=args.allow_short_history,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    progress = tqdm(dataloader, desc="Precompute temporal BEVs", disable=(rank != 0))
    for batch in progress:
        if batch is None:
            continue
        bundles_batch, md_seqs, b2d_seqs = batch

        md_bev = run_get_bevs_sequence_batch(uni_model, pipeline, md_seqs, device)
        b2d_bev = run_get_bevs_sequence_batch(uni_model, pipeline, b2d_seqs, device)

        if md_bev is None or b2d_bev is None:
            continue

        md_batch_size = md_bev.shape[1] if md_bev.dim() == 3 else md_bev.shape[0]
        for idx in range(md_batch_size):
            md_slice = md_bev[:, idx : idx + 1] if md_bev.dim() == 3 else md_bev[idx : idx + 1]
            b2d_slice = b2d_bev[:, idx : idx + 1] if b2d_bev.dim() == 3 else b2d_bev[idx : idx + 1]
            bundle = bundles_batch[idx]
            md_out = cache_dir / "metadrive" / bundle.scenario / f"{bundle.frame}.pt"
            b2d_out = cache_dir / "bench2drive" / bundle.scenario / f"{bundle.frame}.pt"
            if not md_out.exists():
                save_bev(md_slice, bundle, "metadrive", cache_dir)
            if not b2d_out.exists():
                save_bev(b2d_slice, bundle, "bench2drive", cache_dir)
            if args.viz:
                md_png = args.viz_dir / "metadrive" / bundle.scenario / f"{bundle.frame}_md.png"
                b2d_png = args.viz_dir / "bench2drive" / bundle.scenario / f"{bundle.frame}_b2d.png"
                md_png.parent.mkdir(parents=True, exist_ok=True)
                b2d_png.parent.mkdir(parents=True, exist_ok=True)
                visualize_with_heads(md_slice, uni_model, md_png, args.viz_score_thresh, "metadrive")
                visualize_with_heads(b2d_slice, uni_model, b2d_png, args.viz_score_thresh, "bench2drive")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

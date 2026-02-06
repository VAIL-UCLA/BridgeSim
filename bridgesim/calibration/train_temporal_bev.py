#!/usr/bin/env python3
"""
PyTorch Lightning training scaffold for temporal UniAD BEV flow matching (on-the-fly BEV extraction).

Key behaviors:
- Loads MetaDrive/Bench2Drive frame pairs with temporal history and splits using a provided val list.
- Builds temporal clips (history_length + current) with UniAD-style metas (relative can_bus, prev_bev_exists).
- Extracts BEVs on the fly with a single batched temporal forward (MD + B2D concatenated).
- Uses the same FlowMatching model/optimizer/scheduler/EMA setup as train_bev_flow_pl.py.
- Assumes single-node multi-GPU with full precision (no AMP) to match the original scaffold.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import wandb
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from flow_matching_model import FlowMatchingModel
from train_uniad_bev import (
    load_model_and_pipeline,
    load_infos,
    _match_info_entry,
    invert_pose,
    visualize_bev,
    run_heads_from_bev,
)
from train_bev_flow_pl import load_split, split_bundles, build_optimizer, build_lr_scheduler  # reuse helpers

try:
    from mmcv.parallel.collate import collate as mm_collate_to_batch_form
except ImportError:
    from mmcv.parallel import collate as mm_collate_to_batch_form  # type: ignore

# ---- Data structures ----

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


@dataclass
class FrameBundle:
    scenario: str
    frame: str
    bench2drive: Dict[str, str]
    metadrive: Dict[str, str]
    prev_frames: List[str]
    prev_bench2drive: List[Dict[str, str]]
    prev_metadrive: List[Dict[str, str]]


# ---- Parsing ----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal UniAD BEV flow matching (on-the-fly).")
    parser.add_argument("--pairs", type=Path, required=True, help="Frame pairs JSONL with history.")
    parser.add_argument("--info-pkls", nargs="+", required=True, help="Info pkl(s) from prepare_B2D_nautilus.py.")
    parser.add_argument("--config", type=Path, required=True, help="UniAD config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="UniAD checkpoint path.")
    parser.add_argument("--val-split", type=Path, required=True, help="JSON split file with 'val' list.")
    parser.add_argument("--default-command", type=int, default=4, help="Fallback driving command (1-indexed in data).")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--history-length", type=int, default=2, help="Number of past frames (queue_length-1).")
    parser.add_argument("--sample-interval", type=int, default=5, help="Expected stride between frames.")
    parser.add_argument("--allow-short-history", action="store_true", help="Allow fewer past frames at episode starts.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr-scheduler", type=str, default="customized")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-t-max", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--dit-variant", type=str, default="DiT-B/2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, help="Optional cap on number of bundles.")
    parser.add_argument("--devices", type=int, default=-1, help="Lightning devices (-1 for all).")
    parser.add_argument("--strategy", type=str, default="auto", help="Lightning strategy.")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--precision", type=str, default="32-true", help="Use full precision to match baseline.")
    parser.add_argument("--log-wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb-project", type=str, default="temporal-bev-flow")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints_temporal_bev"))
    parser.add_argument("--sigma", type=float, default=0.05, help="Uniform jitter magnitude applied to prev BEV.")
    parser.add_argument("--viz", action="store_true", help="Enable visualization (rank 0 only).")
    parser.add_argument("--viz-dir", type=Path, default=Path("bev_viz_temporal"))
    parser.add_argument("--viz-score-thresh", type=float, default=0.2)
    return parser.parse_args()


# ---- Utilities ----

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frame_bundles(path: Path) -> List[FrameBundle]:
    bundles: List[FrameBundle] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
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
    fake_bundle = FrameBundle(scenario, frame_id, {}, {}, [], [], [])  # for _match_info_entry
    info_entry = _match_info_entry(fake_bundle, info_index)  # type: ignore
    if info_entry is None:
        return None
    imgs_and_names = _load_multiview_images(cam_paths)
    if imgs_and_names is None:
        return None
    imgs, filenames = imgs_and_names
    from pyquaternion import Quaternion  # local import

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
        "flip": False,
        "pcd_horizontal_flip": False,
        "pcd_vertical_flip": False,
        "cam_intrinsic": cam_intrinsics,
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
):
    hist_ids = list(bundle.prev_frames) if history_length > 0 else []
    hist_bench = list(bundle.prev_bench2drive) if history_length > 0 else []
    hist_meta = list(bundle.prev_metadrive) if history_length > 0 else []

    frames = []
    for idx, fid in enumerate(hist_ids):
        cam_map = hist_meta[idx] if domain == "metadrive" else hist_bench[idx]
        frames.append((fid, cam_map, idx))

    # Random drop strategy akin to prepare_train_data: shuffle and drop one when more than needed.
    if len(frames) > history_length:
        tmp = frames.copy()
        random.shuffle(tmp)
        tmp = tmp[1:]
        tmp = sorted(tmp, key=lambda x: x[2])
        tmp = tmp[-history_length:]
        frames = tmp
    else:
        frames = frames[-history_length:]

    frames.append((bundle.frame, getattr(bundle, domain), len(frames)))

    # Enforce interval when numeric IDs, but allow shortening at episode starts.
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

    # Adjust can_bus to be relative like union2one; set prev_bev_exists flags.
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
            rel[:3] = cur_pos - prev_pos
            rel[17] = cur_yaw_deg - prev_yaw_deg
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


def run_get_bevs_sequence_batch(model, pipeline, seqs: List[List[Dict]], device: torch.device, sigma_range: float):
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
            if prev_bev is not None:
                half_batch_ind = int(batch_size / 2)
                sigma = torch.rand(half_batch_ind, device=prev_bev.device) * sigma_range + (1 - sigma_range)
                while sigma.ndim < prev_bev.ndim:
                    sigma = sigma.unsqueeze(-1)
                prev_bev[:half_batch_ind] = (sigma * (1e-5 - 1) + 1) * prev_bev[:half_batch_ind] + sigma * prev_bev[half_batch_ind:]
            bev_out, prev_bev = model.get_bevs(imgs, img_metas, prev_bev)
    return bev_out


# ---- Lightning modules ----

class TemporalBEVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_bundles: List[FrameBundle],
        val_bundles: List[FrameBundle],
        info_index: Dict[Tuple[str, int], Dict],
        default_command: int,
        fps: float,
        history_length: int,
        sample_interval: int,
        allow_short_history: bool,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_bundles = train_bundles
        self.val_bundles = val_bundles
        self.info_index = info_index
        self.default_command = default_command
        self.fps = fps
        self.history_length = history_length
        self.sample_interval = sample_interval
        self.allow_short_history = allow_short_history
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        ds = TemporalPairDataset(
            self.train_bundles,
            self.info_index,
            self.default_command,
            self.fps,
            self.history_length,
            self.sample_interval,
            self.allow_short_history,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = TemporalPairDataset(
            self.val_bundles,
            self.info_index,
            self.default_command,
            self.fps,
            self.history_length,
            self.sample_interval,
            self.allow_short_history,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


class TemporalFlowMatchingLit(pl.LightningModule):
    def __init__(
        self,
        uni_model,
        pipeline,
        lr: float,
        weight_decay: float,
        optimizer_name: str,
        lr_scheduler_name: str,
        warmup_steps: int,
        lr_t_max: int,
        ema_decay: float,
        dit_variant: str,
        visualize: bool,
        viz_dir: Path,
        viz_score_thresh: float,
        sigma: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["uni_model", "pipeline"])
        self.uni_model = uni_model
        self.pipeline = pipeline
        self.flow_model = FlowMatchingModel(dit_variant=dit_variant)
        self.ema_flow_model = deepcopy(self.flow_model)
        for p in self.ema_flow_model.parameters():
            p.requires_grad_(False)
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.warmup_steps = warmup_steps
        self.lr_t_max = lr_t_max
        self.ema_decay = ema_decay
        self.visualize_flag = visualize
        self.viz_dir = viz_dir
        self.viz_score_thresh = viz_score_thresh
        self.sigma = sigma

    def setup(self, stage: Optional[str] = None) -> None:
        # Ensure UniAD backbone lives on the correct device.
        if self.uni_model is not None:
            self.uni_model.to(self.device)

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _ema_update(self):
        if self.ema_flow_model is None:
            return
        ema_params = dict(self.ema_flow_model.named_parameters())
        for name, param in self.flow_model.named_parameters():
            ema_param = ema_params[name]
            ema_param.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param.data)

    def _flow_step(self, bundles, md_bev: torch.Tensor, b2d_bev: torch.Tensor, stage: str):
        src_bchw = self.flow_model.reshape_bev(md_bev)
        tgt_bchw = self.flow_model.reshape_bev(b2d_bev)
        t = torch.rand(src_bchw.shape[0], device=self.device)

        x_t = self.flow_model.psi(t, noise=src_bchw, x1=tgt_bchw)
        target_velocity = self.flow_model.dt_psi(t, noise=src_bchw, x1=tgt_bchw)

        v_pred = self.flow_model(x_t, t)
        loss = (v_pred - target_velocity).pow(2).flatten(1).mean()

        log_name = f"{stage}/loss"
        # torch.cuda.synchronize()  # Force sync here
        # print(f"Loss computed: {loss.item()}", flush=True) 
        batch_size = len(bundles)
        self.log(
            log_name,
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        # if self.visualize_flag:
        #     if self.uni_model is not None:
        #         self.uni_model.eval()
        #     batch_size = len(bundles)
        #     for idx in range(batch_size):
        #         md_slice = md_bev[:, idx : idx + 1]
        #         b2d_slice = b2d_bev[:, idx : idx + 1]
        #         bundle = bundles[idx]
        #         out_dir = self.viz_dir / bundle.scenario
        #         out_dir.mkdir(parents=True, exist_ok=True)
        #         try:
        #             det_out, seg_out, occ_out = run_heads_from_bev(md_slice, self.uni_model, score_thresh=0.2)
        #         except Exception as e:
        #             print(f"[viz] Failed to run heads for bench2drive {bundle.scenario}/{bundle.frame}: {e}")
        #             det_out, seg_out, occ_out = None, None, None

        #         visualize_bev(
        #             md_slice,
        #             self.uni_model,
        #             score_thresh=self.viz_score_thresh,
        #             output_png=out_dir / f"{bundle.frame}_md.png",
        #             seg_data=seg_out,
        #             occ_data=occ_out,
        #         )

        #         try:
        #             det_out, seg_out, occ_out = run_heads_from_bev(b2d_slice, self.uni_model, score_thresh=0.2)
        #         except Exception as e:
        #             print(f"[viz] Failed to run heads for bench2drive {bundle.scenario}/{bundle.frame}: {e}")
        #             det_out, seg_out, occ_out = None, None, None
        #         visualize_bev(
        #             b2d_slice,
        #             self.uni_model,
        #             score_thresh=self.viz_score_thresh,
        #             output_png=out_dir / f"{bundle.frame}_b2d.png",
        #             seg_data=seg_out,
        #             occ_data=occ_out,
        #         )
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._ema_update()

    def _run_stage(self, batch, stage: str):
        if batch is None:
            return torch.zeros((), device=self.device)
        bundles_batch, md_seqs, b2d_seqs = batch
        combined_seqs = list(md_seqs) + list(b2d_seqs)
        with torch.no_grad():
            combined_bev = run_get_bevs_sequence_batch(
                self.uni_model, self.pipeline, combined_seqs, self.device, self.sigma
            )
        md_bev, b2d_bev = torch.chunk(combined_bev, 2, dim=1)
        return self._flow_step(bundles_batch, md_bev, b2d_bev, stage=stage)

    def training_step(self, batch, batch_idx):
        return self._run_stage(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._run_stage(batch, stage="val")

    def test_step(self, batch, batch_idx):
        if batch is None:
            return
        bundles_batch, md_seqs, b2d_seqs = batch
        combined_seqs = list(md_seqs) + list(b2d_seqs)
        with torch.no_grad():
            combined_bev = run_get_bevs_sequence_batch(
                self.uni_model, self.pipeline, combined_seqs, self.device, self.sigma
            )
        md_bev, b2d_bev = torch.chunk(combined_bev, 2, dim=0)

        sampler_model = self.ema_flow_model if self.ema_flow_model is not None else self.flow_model
        gen_bev = sampler_model.sample(md_bev)

        gen_bchw = self.flow_model.reshape_bev(gen_bev)
        tgt_bchw = self.flow_model.reshape_bev(b2d_bev)
        loss = (gen_bchw - tgt_bchw).pow(2).flatten(1).mean()
        self.log("test/loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.visualize_flag and self.global_rank == 0:
            if self.uni_model is not None:
                self.uni_model.eval()
            batch_size = len(bundles_batch)
            for idx in range(batch_size):
                md_slice = md_bev[idx : idx + 1]
                b2d_slice = b2d_bev[idx : idx + 1]
                bundle = bundles_batch[idx]
                out_dir = self.viz_dir / bundle.scenario
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    det_out, seg_out, occ_out = run_heads_from_bev(md_slice, self.uni_model, score_thresh=0.2)
                except Exception as e:
                    print(f"[viz] Failed to run heads for bench2drive {bundle.scenario}/{bundle.frame}: {e}")
                    det_out, seg_out, occ_out = None, None, None

                visualize_bev(
                    md_slice,
                    self.uni_model,
                    score_thresh=self.viz_score_thresh,
                    output_png=out_dir / f"{bundle.frame}_md.png",
                    seg_data=seg_out,
                    occ_data=occ_out,
                )

                try:
                    det_out, seg_out, occ_out = run_heads_from_bev(b2d_slice, self.uni_model, score_thresh=0.2)
                except Exception as e:
                    print(f"[viz] Failed to run heads for bench2drive {bundle.scenario}/{bundle.frame}: {e}")
                    det_out, seg_out, occ_out = None, None, None
                visualize_bev(
                    b2d_slice,
                    self.uni_model,
                    score_thresh=self.viz_score_thresh,
                    output_png=out_dir / f"{bundle.frame}_b2d.png",
                    seg_data=seg_out,
                    occ_data=occ_out,
                )

    def configure_optimizers(self):
        optimizer = build_optimizer(self.flow_model.parameters(), self.optimizer_name, self.lr, self.weight_decay)
        scheduler = build_lr_scheduler(
            optimizer,
            self.lr_scheduler_name,
            warmup_steps=self.warmup_steps,
            t_max=self.lr_t_max,
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ---- Main ----


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    bundles = load_frame_bundles(args.pairs)
    if args.max_samples:
        bundles = bundles[: args.max_samples]
    val_set = load_split(args.val_split)
    train_b, val_b = split_bundles(bundles, val_set)
    info_index = load_infos([Path(p) for p in args.info_pkls])

    uni_model, pipeline = load_model_and_pipeline(str(args.config), str(args.checkpoint))
    uni_model.eval()
    for p in uni_model.parameters():
        p.requires_grad_(False)

    dm = TemporalBEVDataModule(
        train_bundles=train_b,
        val_bundles=val_b,
        info_index=info_index,
        default_command=args.default_command,
        fps=args.fps,
        history_length=max(0, args.history_length),
        sample_interval=max(1, args.sample_interval),
        allow_short_history=args.allow_short_history,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    lit_model = TemporalFlowMatchingLit(
        uni_model=uni_model,
        pipeline=pipeline,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        lr_scheduler_name=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        lr_t_max=args.lr_t_max,
        ema_decay=args.ema_decay,
        dit_variant=args.dit_variant,
        visualize=args.viz,
        viz_dir=args.viz_dir,
        viz_score_thresh=args.viz_score_thresh,
        sigma=args.sigma,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="temporal-bev-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    logger = None
    if args.log_wandb:
        wandb.login(key="caad08df59bfd0cb22f3613849ad66faeb65d4b0")
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            save_dir=str(args.ckpt_dir),
            config=vars(args),
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[ckpt_cb],
        logger=logger,
        enable_model_summary=False,
        gradient_clip_val=0.0,
    )

    trainer.fit(lit_model, dm)


if __name__ == "__main__":
    main()

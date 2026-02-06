#!/usr/bin/env python3
"""
UniAD BEV extraction scaffold for MetaDrive/Bench2Drive paired frames.

This step focuses solely on:
1) Loading an offline frame-pairs JSONL (from build_bench2drive_metadrive_frame_pairs.py).
2) Building UniAD-ready samples (images + minimal meta/extrinsics).
3) Running UniAD to cache or inspect BEV embeddings for both domains.

Flow-matching/learning is intentionally left as a placeholder to be filled in
after BEV extraction is verified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Bench2Drive model imports from modelzoo
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELZOO_BENCH2DRIVE = REPO_ROOT / "modelzoo" / "bench2drive"
sys.path.insert(0, str(MODELZOO_BENCH2DRIVE))

from bridgesim.modelzoo.bench2drive.mmcv import Config
from bridgesim.modelzoo.bench2drive.mmcv.core.bbox import get_box_type
from bridgesim.modelzoo.bench2drive.mmcv.datasets.pipelines import Compose
from bridgesim.modelzoo.bench2drive.mmcv.models import build_model
from bridgesim.modelzoo.bench2drive.mmcv.parallel.collate import collate as mm_collate_to_batch_form
from bridgesim.modelzoo.bench2drive.mmcv.utils import load_checkpoint

LIDAR2IMG = {
    "CAM_FRONT": np.array(
        [
            [1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
            [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
            [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_FRONT_LEFT": np.array(
        [
            [6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
            [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
            [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_FRONT_RIGHT": np.array(
        [
            [1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
            [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
            [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK": np.array(
        [
            [-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
            [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
            [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK_LEFT": np.array(
        [
            [-1.14251841e03, 8.00000000e02, 0.00000000e00, -6.84385123e02],
            [-4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
            [-9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK_RIGHT": np.array(
        [
            [3.60989788e02, -1.34723223e03, 0.00000000e00, -1.04238127e02],
            [4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
            [9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
}
LIDAR2CAM = {
    "CAM_FRONT": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]]
    ),
    "CAM_FRONT_LEFT": np.array(
        [
            [0.57357644, 0.81915204, 0.0, -0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [-0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_FRONT_RIGHT": np.array(
        [
            [0.57357644, -0.81915204, 0.0, 0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_BACK": np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]]
    ),
    "CAM_BACK_LEFT": np.array(
        [
            [-0.34202014, 0.93969262, 0.0, -0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [-0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_BACK_RIGHT": np.array(
        [
            [-0.34202014, -0.93969262, 0.0, 0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}
LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, -0.39], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]]
)
CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def _bench2drive_zoo_root(cfg_path: Path) -> Path:
    """Get the Bench2Drive modelzoo root directory."""
    # First check if config is in the new modelzoo location
    for parent in cfg_path.resolve().parents:
        if parent.name == "bench2drive":
            return parent
    # Fallback to modelzoo/bench2drive
    if MODELZOO_BENCH2DRIVE.exists():
        return MODELZOO_BENCH2DRIVE
    raise RuntimeError(f"Bench2Drive modelzoo root not found for config {cfg_path}")


def load_model_and_pipeline(config_path: str, checkpoint_path: str):
    cfg_path = Path(config_path)
    cfg = Config.fromfile(str(cfg_path))
    bench2drive_zoo_root = _bench2drive_zoo_root(cfg_path)

    motion_head = cfg.model.get("motion_head") if hasattr(cfg, "model") else None
    if motion_head and "anchor_info_path" in motion_head:
        anchor_path = Path(motion_head["anchor_info_path"])
        if not anchor_path.is_absolute():
            anchor_path = bench2drive_zoo_root / anchor_path
        cfg.model["motion_head"]["anchor_info_path"] = str(anchor_path)

    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir:
            plugin_dir_path = Path(plugin_dir)
            if not plugin_dir_path.is_absolute():
                plugin_dir_path = bench2drive_zoo_root / plugin_dir_path
            sys.path.insert(0, str(plugin_dir_path.parent))
            module_path = ".".join(plugin_dir_path.with_suffix("").parts[-2:])
            __import__(module_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    _ = load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=True)
    model.eval()

    inference_only_pipeline_cfg: List[Dict] = []
    for pipeline_cfg in cfg.inference_only_pipeline:
        if pipeline_cfg["type"] in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles"):
            continue
        inference_only_pipeline_cfg.append(pipeline_cfg)
    inference_pipeline = Compose(inference_only_pipeline_cfg)
    return model, inference_pipeline


def load_inference_pipeline(config_path: str):
    """Build inference-only pipeline without loading the model or checkpoint."""
    cfg_path = Path(config_path)
    cfg = Config.fromfile(str(cfg_path))
    bench2drive_zoo_root = _bench2drive_zoo_root(cfg_path)

    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir:
            plugin_dir_path = Path(plugin_dir)
            if not plugin_dir_path.is_absolute():
                plugin_dir_path = bench2drive_zoo_root / plugin_dir_path
            sys.path.insert(0, str(plugin_dir_path.parent))
            module_path = ".".join(plugin_dir_path.with_suffix("").parts[-2:])
            __import__(module_path)

    inference_only_pipeline_cfg: List[Dict] = []
    for pipeline_cfg in cfg.inference_only_pipeline:
        if pipeline_cfg["type"] in ("LoadMultiViewImageFromFilesInCeph", "LoadMultiViewImageFromFiles"):
            continue
        inference_only_pipeline_cfg.append(pipeline_cfg)
    return Compose(inference_only_pipeline_cfg)


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
class ExtractConfig:
    pairs: Path
    config: Path
    checkpoint: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_root: Optional[Path] = None
    default_command: int = 4  # LANEFOLLOW fallback (0-indexed expected by UniAD)
    fps: float = 20.0
    max_samples: Optional[int] = None
    save_cpu: bool = True
    seed: int = 42
    info_pkls: List[Path] = None
    visualize: bool = False
    viz_dir: Optional[Path] = None
    viz_score_thresh: float = 0.2


def parse_args() -> ExtractConfig:
    parser = argparse.ArgumentParser(
        description="Extract UniAD BEV embeddings for MetaDrive/Bench2Drive paired frames."
    )
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL from build_bench2drive_metadrive_frame_pairs.py.")
    parser.add_argument("--config", type=Path, required=True, help="Path to UniAD config (.py).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to UniAD checkpoint (.pth).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, help="Optional directory to save BEV tensors (per scenario/frame).")
    parser.add_argument("--default-command", type=int, default=4, help="Fallback driving command if unavailable.")
    parser.add_argument("--fps", type=float, default=20.0, help="Frame rate used to derive timestamps.")
    parser.add_argument("--max-samples", type=int, help="Limit number of frame bundles processed.")
    parser.add_argument("--save-cpu", action="store_true", default=True, help="Save BEVs on CPU to reduce size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--info-pkls",
        nargs="+",
        required=True,
        help="Info pkl(s) from prepare_B2D_nautilus.py (assumed shared by metadrive/bench2drive).",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize BEV detections during extraction.")
    parser.add_argument("--viz-dir", type=Path, help="Directory to save BEV visualizations.")
    parser.add_argument("--viz-score-thresh", type=float, default=0.2, help="Score threshold for BEV visualization.")
    args = parser.parse_args()
    return ExtractConfig(
        pairs=args.pairs,
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        output_root=args.output_root,
        default_command=args.default_command,
        fps=args.fps,
        max_samples=args.max_samples,
        save_cpu=args.save_cpu,
        seed=args.seed,
        info_pkls=[Path(p) for p in args.info_pkls],
        visualize=bool(args.visualize),
        viz_dir=args.viz_dir,
        viz_score_thresh=args.viz_score_thresh,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FrameBundle:
    scenario: str
    frame: str
    bench2drive: Dict[str, str]
    metadrive: Dict[str, str]
    prev_frame: Optional[str] = None
    prev_bench2drive: Optional[Dict[str, str]] = None
    prev_metadrive: Optional[Dict[str, str]] = None


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
                        prev_frame=payload.get("prev_frame"),
                        prev_bench2drive=payload.get("prev_bench2drive"),
                        prev_metadrive=payload.get("prev_metadrive"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"Missing field {exc} on line {line_no}") from exc
    return bundles


def load_infos(info_paths: List[Path]) -> Dict[Tuple[str, int], Dict]:
    """Load info.pkl entries and index by (folder, frame_idx)."""
    index: Dict[Tuple[str, int], Dict] = {}
    import pickle

    for p in info_paths:
        if not p.is_file():
            raise FileNotFoundError(f"info.pkl not found: {p}")
        with p.open("rb") as f:
            data = pickle.load(f)
        for entry in data:
            key = (entry["folder"], int(entry["frame_idx"]))
            index[key] = entry
    return index


def _timestamp_from_frame(frame_id: str, fps: float) -> float:
    try:
        return int(frame_id) / fps
    except ValueError:
        return 0.0


def invert_pose(pose: np.ndarray) -> np.ndarray:
    inv_pose = np.eye(4, dtype=np.float32)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, -1] = -inv_pose[:3, :3] @ pose[:3, -1]
    return inv_pose.astype(np.float32)


def _load_multiview_images(cam_paths: Dict[str, str]) -> Optional[Tuple[List[np.ndarray], List[str]]]:
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


def _zero_can_bus() -> np.ndarray:
    # 18-dim vector expected by UniAD; placeholders until motion data is provided.
    can_bus = np.zeros(18, dtype=np.float32)
    return can_bus


def _match_info_entry(bundle: FrameBundle, info_index: Dict[Tuple[str, int], Dict]) -> Optional[Dict]:
    """Try to locate info entry with a few folder-name variants."""
    try:
        frame_idx = int(bundle.frame)
    except ValueError:
        return None
    candidates = [
        (bundle.scenario, frame_idx),
        (f"v1/{bundle.scenario}", frame_idx),
        (bundle.scenario.lstrip("v1/"), frame_idx),
    ]
    for key in candidates:
        if key in info_index:
            return info_index[key]
    return None


def _identity_lidar_to_global() -> Tuple[np.ndarray, np.ndarray]:
    ego2world = np.eye(4, dtype=np.float32)
    lidar2global = ego2world @ LIDAR2EGO
    return lidar2global[0:3, 0:3].astype(np.float32), lidar2global[0:3, 3].astype(np.float32)


def build_sample(bundle: FrameBundle, domain: str, cfg: ExtractConfig, info_index: Dict[Tuple[str, int], Dict]) -> Optional[Dict]:
    """
    Construct a UniAD-ready sample dict for one domain (metadrive/bench2drive).
    Returns None if any required image is missing or unreadable.
    """
    cam_paths = getattr(bundle, domain)
    imgs_and_names = _load_multiview_images(cam_paths)
    if imgs_and_names is None:
        return None
    imgs, filenames = imgs_and_names
    info_entry = _match_info_entry(bundle, info_index)
    if info_entry is None:
        print(f"[warn] Missing info for {bundle.scenario} frame {bundle.frame}")
        return None

    from pyquaternion import Quaternion  # local import to avoid global import if unused

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

    command = info_entry.get("command_near", 4)
    if command < 0:
        command = 4
    command = int(command) - 1  # UniAD expects 0-indexed
    timestamp = info_entry.get("timestamp", info_entry["frame_idx"] / 10.0)
    frame_idx = int(info_entry["frame_idx"])
    box_type_3d, _ = get_box_type("LiDAR")

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
        "box_type_3d": box_type_3d,
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


def reset_temporal_state(model) -> None:
    """Clear UniAD temporal caches so each frame is treated independently."""
    for attr in ["prev_frame_infos", "prev_frame_info", "prev_bev"]:
        if hasattr(model, attr):
            setattr(model, attr, [] if isinstance(getattr(model, attr), list) else None)


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, data in batch.items():
        if key == "img_metas":
            continue
        if torch.is_tensor(data[0]):
            data[0] = data[0].to(device, non_blocking=True)
    return batch


def extract_bev_embedding(sample: Dict, model, pipeline, device: torch.device) -> torch.Tensor:
    """
    Run only the BEV encoder (get_bevs) to obtain the BEV embedding, avoiding the
    full UniAD forward.
    """
    processed = pipeline(sample)
    batch = mm_collate_to_batch_form([processed], samples_per_gpu=1)
    batch = move_to_device(batch, device)
    reset_temporal_state(model)

    imgs = batch["img"][0] if isinstance(batch["img"], (list, tuple)) else batch["img"]
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)  # (B=1, N, C, H, W)

    img_metas = batch["img_metas"][0] if isinstance(batch["img_metas"], (list, tuple)) else batch["img_metas"]
    if hasattr(img_metas, "data"):
        img_metas = img_metas.data
    if not isinstance(img_metas, list):
        img_metas = [img_metas]

    if not hasattr(model, "get_bevs"):
        raise RuntimeError("Loaded UniAD model does not expose get_bevs; ensure config uses UniADTrack/BEV encoder.")

    with torch.no_grad():
        bev_embed, _ = model.get_bevs(imgs, img_metas, prev_bev=None)
    return bev_embed.detach()


def extract_bev_embedding_from_batch(batch: Dict, model, device: torch.device) -> torch.Tensor:
    """
    Variant of extract_bev_embedding that consumes an already-collated batch
    (output of mm_collate_to_batch_form) to avoid rerunning the pipeline.
    """
    batch = move_to_device(batch, device)
    reset_temporal_state(model)

    imgs = batch["img"][0] if isinstance(batch["img"], (list, tuple)) else batch["img"]
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)

    img_metas = batch["img_metas"]
    # unwrap DataContainer and extra samples_per_gpu nesting
    if isinstance(img_metas, (list, tuple)):
        if len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]
        img_metas = [m.data[0] if hasattr(m, "data") else m for m in img_metas]
    else:
        if hasattr(img_metas, "data"):
            img_metas = img_metas.data
        img_metas = img_metas if isinstance(img_metas, list) else [img_metas]

    with torch.no_grad():
        bev_embed, _ = model.get_bevs(imgs, img_metas, prev_bev=None)
    return bev_embed.detach()


def extract_pair_bevs(
    bundle: FrameBundle,
    cfg: ExtractConfig,
    model,
    pipeline,
    device: torch.device,
    info_index: Dict[Tuple[str, int], Dict],
) -> Optional[Dict[str, torch.Tensor]]:
    md_sample = build_sample(bundle, "metadrive", cfg, info_index)
    b2d_sample = build_sample(bundle, "bench2drive", cfg, info_index)
    if md_sample is None or b2d_sample is None:
        return None
    md_bev = extract_bev_embedding(md_sample, model, pipeline, device)
    b2d_bev = extract_bev_embedding(b2d_sample, model, pipeline, device)
    return {"metadrive": md_bev, "bench2drive": b2d_bev}


def maybe_save_bevs(bundle: FrameBundle, bevs: Dict[str, torch.Tensor], cfg: ExtractConfig) -> None:
    if cfg.output_root is None:
        return
    out_dir = cfg.output_root / bundle.scenario / str(bundle.frame)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {k: (v.cpu() if cfg.save_cpu else v) for k, v in bevs.items()}
    torch.save(payload, out_dir / "bev_pair.pt")


class FlowMatchingPlaceholder(torch.nn.Module):
    """
    Reserved for future flow-matching logic. Not used in the current extraction step.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Flow matching will be implemented after BEV extraction is validated.")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    model, pipeline = load_model_and_pipeline(str(cfg.config), str(cfg.checkpoint))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    bundles = load_frame_bundles(cfg.pairs)
    info_index = load_infos(cfg.info_pkls)
    if cfg.max_samples is not None:
        bundles = bundles[: cfg.max_samples]

    success = 0
    for idx, bundle in enumerate(bundles, start=1):
        bevs = extract_pair_bevs(bundle, cfg, model, pipeline, device, info_index)
        if bevs is None:
            print(f"[skip] Missing data for {bundle.scenario} frame {bundle.frame}")
            continue
        success += 1
        print(
            f"[{idx}/{len(bundles)}] {bundle.scenario} frame {bundle.frame}: "
            f"md_bev {tuple(bevs['metadrive'].shape)}, b2d_bev {tuple(bevs['bench2drive'].shape)}"
        )
        maybe_save_bevs(bundle, bevs, cfg)
        if cfg.visualize:
            out_dir = cfg.viz_dir or cfg.output_root or (REPO_ROOT / "bev_viz")
            out_dir = Path(out_dir)
            try:
                det_out, seg_out, occ_out = run_heads_from_bev(bevs["bench2drive"], model, score_thresh=cfg.viz_score_thresh)
            except Exception as e:
                print(f"[viz] Failed to run heads for bench2drive {bundle.scenario}/{bundle.frame}: {e}")
                det_out, seg_out, occ_out = None, None, None
            visualize_bev(
                bevs["bench2drive"],
                model,
                score_thresh=cfg.viz_score_thresh,
                output_png=out_dir / bundle.scenario / f"{bundle.frame}_b2d.png",
                seg_data=seg_out,
                occ_data=occ_out,
            )
            try:
                det_out_md, seg_out_md, occ_out_md = run_heads_from_bev(bevs["metadrive"], model, score_thresh=cfg.viz_score_thresh)
            except Exception as e:
                print(f"[viz] Failed to run heads for metadrive {bundle.scenario}/{bundle.frame}: {e}")
                det_out_md, seg_out_md, occ_out_md = None, None, None
            visualize_bev(
                bevs["metadrive"],
                model,
                score_thresh=cfg.viz_score_thresh,
                output_png=out_dir / bundle.scenario / f"{bundle.frame}_md.png",
                seg_data=seg_out_md,
                occ_data=occ_out_md,
            )

    print(f"Completed. Valid BEV pairs: {success}/{len(bundles)}")


# ---------------- Visualization helpers ----------------

BEV_EXTENT = [-51.2, 51.2, -51.2, 51.2]  # (y_min, y_max, x_min, x_max)
AGENT_CLASSES = [
    "car",
    "van",
    "truck",
    "bicycle",
    "traffic_sign",
    "traffic_cone",
    "traffic_light",
    "pedestrian",
    "others",
]
AGENT_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, len(AGENT_CLASSES)))


def _normalize_bev(bev: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    """
    Ensure BEV shape is (H*W, B, C). Accepts (H*W, B, C), (B, H*W, C), or (B, C, H, W).
    """
    if bev.dim() == 3:
        # Check if it's already (H*W, B, C)
        if bev.shape[0] == bev_h * bev_w:
            return bev
        # Check if it's (B, H*W, C) from precomputed cache
        if bev.shape[1] == bev_h * bev_w:
            # Permute: (B, H*W, C) -> (H*W, B, C)
            return bev.permute(1, 0, 2)
    if bev.dim() == 4:
        # Check if it's a malformed (1, H*W, B, C) or similar
        if bev.shape[1] == bev_h * bev_w:
            assert bev.shape[0] == 1
            return bev[0]
        # Standard (B, C, H, W) format
        b, c, h, w = bev.shape
        assert h == bev_h and w == bev_w, f"Unexpected BEV size {h}x{w}, expected {bev_h}x{bev_w}"
        bev = bev.permute(2, 3, 0, 1).reshape(h * w, b, c)
        return bev
    raise ValueError(f"Unsupported BEV shape {bev.shape}")


def _decode_tracks(model, det_output, track_instances, img_metas):
    track_scores = det_output["all_cls_scores"][-1, 0, :].sigmoid().max(dim=-1).values
    track_instances.scores = track_scores
    track_instances.pred_logits = det_output["all_cls_scores"][-1, 0]
    track_instances.pred_boxes = det_output["all_bbox_preds"][-1, 0]
    track_instances.output_embedding = det_output["query_feats"][-1][0]
    track_instances.ref_pts = det_output["last_ref_points"][0]
    result = model._track_instances2results(track_instances, img_metas, with_mask=False)  # type: ignore[attr-defined]
    return result


def run_heads_from_bev(
    bev_embed: torch.Tensor,
    model,
    score_thresh: float = 0.2,
) -> Tuple[Dict, Dict, Dict]:
    """
    Given a BEV embedding, run detection -> seg -> motion -> occ heads to
    produce outputs suitable for visualization.
    """
    bev_h = getattr(model.pts_bbox_head, "bev_h", 200)
    bev_w = getattr(model.pts_bbox_head, "bev_w", 200)
    bev_embed = _normalize_bev(bev_embed, bev_h, bev_w)

    box_type_3d, _ = get_box_type("LiDAR")
    img_metas = [dict(box_type_3d=box_type_3d)]

    # Detection
    track_instances = model._generate_empty_tracks()  # type: ignore[attr-defined]
    det_output = model.pts_bbox_head.get_detections(  # type: ignore[attr-defined]
        bev_embed,
        object_query_embeds=track_instances.query,
        ref_points=track_instances.ref_pts,
        img_metas=img_metas,
    )
    track_scores = det_output["all_cls_scores"][-1, 0, :].sigmoid().max(dim=-1).values
    track_instances.scores = track_scores
    track_instances.pred_logits = det_output["all_cls_scores"][-1, 0]
    track_instances.pred_boxes = det_output["all_bbox_preds"][-1, 0]
    track_instances.output_embedding = det_output["query_feats"][-1][0]
    track_instances.ref_pts = det_output["last_ref_points"][0]
    # mark sdc query
    track_instances.obj_idxes[900] = -2  # hard-coded ego query

    model.track_base.update(track_instances, None)  # type: ignore[attr-defined]
    active_index = (track_instances.obj_idxes >= 0) & (
        track_instances.scores >= model.track_base.filter_score_thresh  # type: ignore[attr-defined]
    )
    outs_track = {}
    outs_track.update(model.select_active_track_query(track_instances, active_index, img_metas))  # type: ignore[attr-defined]
    outs_track.update(model.select_sdc_track_query(track_instances[track_instances.obj_idxes == -2], img_metas))  # type: ignore[attr-defined]
    outs_track["bev_pos"] = det_output.get("bev_pos", None)

    # Segmentation head
    seg_results = model.seg_head.forward_test(bev_embed, img_metas=img_metas, rescale=False)  # type: ignore[attr-defined]
    seg_data = seg_results[0]

    # Motion head
    motion_results, outs_motion = model.motion_head.forward_test(  # type: ignore[attr-defined]
        bev_embed, outs_track=outs_track, outs_seg=seg_data
    )
    outs_motion["bev_pos"] = outs_track.get("bev_pos", None)

    # Occupancy head
    occ_no_query = outs_motion["track_query"].shape[1] == 0
    occ_data = model.occ_head.forward_test(  # type: ignore[attr-defined]
        bev_embed,
        outs_motion,
        no_query=occ_no_query,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
    )

    return det_output, seg_data, occ_data


def _plot_map(ax, seg_data) -> None:
    bev_h, bev_w = 200, 200
    bev_map_rgba = np.zeros((bev_h, bev_w, 4))
    if "pts_bbox" in seg_data and "drivable" in seg_data["pts_bbox"]:
        drivable_mask = seg_data["pts_bbox"]["drivable"].cpu().numpy()
        bev_map_rgba[drivable_mask] = np.array([0, 0.4, 0, 0.2])
    if "pts_bbox" in seg_data and "lane" in seg_data["pts_bbox"]:
        lane_masks = seg_data["pts_bbox"]["lane"].cpu().numpy()  # (6, H, W)
        lane_colors = [
            np.array([1, 1, 0, 1]),
            np.array([0.6, 0.3, 1, 1]),
            np.array([1, 0.5, 0, 1]),
            np.array([0, 0.8, 1, 1]),
            np.array([1, 0.2, 0.7, 1]),
            np.array([0.7, 0.7, 0.7, 1]),
        ]
        for idx, mask in enumerate(lane_masks):
            bev_map_rgba[mask > 0] = lane_colors[idx % len(lane_colors)]
    ax.imshow(bev_map_rgba, extent=BEV_EXTENT, origin="lower", interpolation="none", zorder=1)


def _plot_occupancy(ax, occ_data) -> None:
    if "seg_out" not in occ_data:
        return
    occ_grid = occ_data["seg_out"][0, 0, 0].float().cpu().numpy()
    occ_masked = np.ma.masked_where(occ_grid == 0, occ_grid)
    ax.imshow(
        occ_masked,
        cmap="gray_r",
        extent=BEV_EXTENT,
        origin="lower",
        alpha=0.4,
        zorder=2,
        vmin=0,
        vmax=1,
    )


def visualize_bev(
    bev_embed: torch.Tensor,
    model,
    *,
    score_thresh: float = 0.2,
    output_png: Optional[Path] = None,
    seg_data: Optional[Dict] = None,
    occ_data: Optional[Dict] = None,
) -> None:
    """
    Visualize detection results from a BEV embedding using the model's heads.
    Only detection outputs are rendered (map/occ heads are skipped).
    """
    bev_h = getattr(model.pts_bbox_head, "bev_h", 200)
    bev_w = getattr(model.pts_bbox_head, "bev_w", 200)
    bev_embed = _normalize_bev(bev_embed, bev_h, bev_w)

    box_type_3d, _ = get_box_type("LiDAR")
    img_metas = [dict(box_type_3d=box_type_3d)]

    track_instances = model._generate_empty_tracks()  # type: ignore[attr-defined]
    det_output = model.pts_bbox_head.get_detections(  # type: ignore[attr-defined]
        bev_embed,
        object_query_embeds=track_instances.query,
        ref_points=track_instances.ref_pts,
        img_metas=img_metas,
    )
    det_res = _decode_tracks(model, det_output, track_instances, img_metas)

    boxes = det_res["boxes_3d"]
    labels = det_res["labels_3d"]
    scores = det_res["scores_3d"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(BEV_EXTENT[0], BEV_EXTENT[1])
    ax.set_ylim(BEV_EXTENT[2], BEV_EXTENT[3])
    ax.set_xlabel("Y (m, left/right)")
    ax.set_ylabel("X (m, forward)")
    ax.set_title("BEV Detections")
    ax.grid(True, linestyle="--", alpha=0.3)

    if seg_data is not None:
        _plot_map(ax, seg_data)
    if occ_data is not None:
        _plot_occupancy(ax, occ_data)

    corners = boxes.corners  # (N, 8, 3)
    for i in range(corners.shape[0]):
        if scores[i] < score_thresh:
            continue
        color = AGENT_COLORS[int(labels[i]) % len(AGENT_COLORS)]
        box_corners = corners[i][:4, :2].cpu().numpy()
        poly = patches.Polygon(
            box_corners,
            closed=True,
            facecolor=color,
            edgecolor="k",
            alpha=0.5,
        )
        ax.add_patch(poly)

    if output_png:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()

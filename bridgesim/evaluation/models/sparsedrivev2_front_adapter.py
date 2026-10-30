"""
SparseDriveV2 (front-camera-only) adapter for the BridgeSim unified evaluator.

The front-only checkpoint was trained with `cams=["cam_f0"]` and v1 metric
heads (no_at_fault_collisions, drivable_area_compliance,
driving_direction_compliance, time_to_collision_within_bound, ego_progress,
comfort). At inference, only CAM_F0 imagery + ego status (driving command,
velocity, acceleration) are needed; the LiDAR/multi-cam paths are unused.

The model's feature builder expects a `camera_feature` dict containing
`imgs`, `projection_mat`, and `image_wh` (built from CAM_F0 intrinsics +
extrinsics, with the resize/crop transform folded into the projection
matrix). We mirror SD's test-mode `resize_crop_flip_img` math here so the
in-memory image from the BridgeSim renderer flows through the same
geometry that the trained model expects.
"""

import copy
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.models.model_configs import FRONT_ONLY_BASE
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from bridgesim.modelzoo.navsim.agents.sparsedrive.sparsedrive_config import (
    SparseDriveConfig,
)
from bridgesim.modelzoo.navsim.agents.sparsedrive.sparsedrive_model import (
    SparseDriveModel,
)
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS


FRONT_CAM_KEY = "CAM_F0"
FRONT_CAM_SD = "cam_f0"

# Default aux asset location (independent of the current working directory):
# <DREAMSTREAM_POLICY_ROOT>/aux/sparsedrive/
#   resnet34.bin  (timm/resnet34.a1_in1k pytorch_model.bin)
#   kmeans/{path_1024.npy, velocity_256.npy, trajectory_1024_256.npz}
_DEFAULT_AUX_DIR = Path(FRONT_ONLY_BASE) / "aux" / "sparsedrive"


def _v1_metric_list():
    return [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
    ]


class SparseDriveV2FrontAdapter(BaseModelAdapter):
    """Adapter for the front-only SparseDriveV2 checkpoint."""

    def __init__(self, checkpoint_path: str, aux_dir: str = None, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.aux_dir = aux_dir
        self.config = None
        self._cam_params = None
        self._current_frame_id = 0

    # ------------------------------------------------------------------ load

    def _resolve_aux(self, rel: str) -> str:
        """Resolve an aux asset path relative to either --sd-aux-dir or the
        default <DREAMSTREAM_POLICY_ROOT>/aux/sparsedrive/ layout."""
        base = Path(self.aux_dir) if self.aux_dir is not None else _DEFAULT_AUX_DIR
        return str(base / rel)

    def load_model(self):
        print("Loading SparseDriveV2 (front-only) model...")

        self.config = SparseDriveConfig()
        # Front-only config used to train the checkpoint:
        self.config.cams = [FRONT_CAM_SD]
        self.config.velocity_filter_num = [64, 20]
        self.config.dataset_version = "v1"
        self.config.metrics = _v1_metric_list()

        # Aux assets (resnet34 backbone, k-means vocab files)
        self.config.bkb_path = self._resolve_aux("resnet34.bin")
        self.config.path_anchor = self._resolve_aux("kmeans/path_1024.npy")
        self.config.velocity_anchor = self._resolve_aux("kmeans/velocity_256.npy")
        self.config.trajectory_anchor = self._resolve_aux("kmeans/trajectory_1024_256.npz")

        self.model = SparseDriveModel(self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Upstream trains as `agent._sparsedrive_model.<...>`; strip prefixes
        # so the keys line up with our SparseDriveModel instantiation.
        clean_sd = {}
        for k, v in state_dict.items():
            new_key = (
                k.replace("agent._sparsedrive_model.", "")
                .replace("_sparsedrive_model.", "")
                .replace("agent.", "")
            )
            clean_sd[new_key] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"[SD-Front LOAD] Missing keys: {len(missing_keys)} (sample: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"[SD-Front LOAD] Unexpected keys: {len(unexpected_keys)} (sample: {unexpected_keys[:5]})")

        self.model.to(self.device)
        self.model.eval()

        self._build_camera_params()

        print("SparseDriveV2 (front-only) model loaded successfully.")

    def _build_camera_params(self):
        """Precompute CAM_F0 intrinsics + extrinsics in SD's expected form.

        Mirrors the RAP adapter's calibration setup: uses the converter's
        camera_params dict (CAM_F0 entry) which provides sensor2lidar
        rotation/translation and intrinsics derived from the navsim sensor
        rig spec.
        """
        # Local import to avoid pulling renderer-related deps at module import.
        from bridgesim.evaluation.models.rap_adapter import camera_params as CP

        params = CP[FRONT_CAM_KEY]
        intrinsic_3x3 = params["intrinsics"].astype(np.float64)
        sensor2lidar_rotation = params["sensor2lidar_rotation"].astype(np.float64)
        sensor2lidar_translation = params["sensor2lidar_translation"].astype(np.float64)
        distortion = params.get("distortion", np.zeros(5, dtype=np.float64))

        self._cam_params = {
            "intrinsics_3x3": intrinsic_3x3,
            "sensor2lidar_rotation": sensor2lidar_rotation,
            "sensor2lidar_translation": sensor2lidar_translation,
            "distortion": distortion.astype(np.float64),
        }

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in (FRONT_CAM_KEY,)}

    # ------------------------------------------------------------- preprocess

    def _img_transform(self, img: np.ndarray, aug_config: dict):
        """Test-mode resize + crop matching SparseDriveFeatureBuilder._img_transform."""
        H, W = img.shape[:2]
        resize = aug_config["resize"]
        resize_dims = aug_config["resize_dims"]
        crop = aug_config["crop"]
        rotate = aug_config.get("rotate", 0)

        origin_dtype = img.dtype
        scale = None
        min_value = None
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_value = img.max()
            scale = 255 / (max_value - min_value) if max_value != min_value else 1.0
            img = (img - min_value) * scale
            img = np.uint8(img)
        pil = Image.fromarray(img)
        pil = pil.resize(resize_dims).crop(crop)
        if rotate:
            pil = pil.rotate(rotate)
        out = np.array(pil).astype(np.float32)
        if origin_dtype != np.uint8 and scale is not None:
            out = out / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        # rotate=0 for inference; skip rotation matrix application
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return out, extend_matrix

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Build the camera_feature dict the model expects."""
        cfg = self.config

        cam_f0 = images_dict.get(FRONT_CAM_KEY)
        if cam_f0 is None:
            cam_f0 = np.zeros((cfg.H, cfg.W, 3), dtype=np.uint8)

        # Optional native-resolution normalization (matches SD's load_images path)
        if cfg.image_resize_hw is not None and tuple(cam_f0.shape[:2]) != tuple(cfg.image_resize_hw):
            th, tw = cfg.image_resize_hw
            cam_f0 = cv2.resize(cam_f0, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        if cfg.to_bgr:
            cam_f0 = cv2.cvtColor(cam_f0, cv2.COLOR_RGB2BGR)
        cam_f0 = cam_f0.astype(np.float32)

        # Build lidar2img / cam_intrinsic for CAM_F0 in the pre-resize coordinate
        # frame (so the resize-crop matrix can be folded into it below).
        p = self._cam_params
        cam2lidar_rt = np.eye(4)
        cam2lidar_rt[:3, :3] = p["sensor2lidar_rotation"]
        cam2lidar_rt[:3, 3] = p["sensor2lidar_translation"]

        lidar2cam_r = np.linalg.inv(p["sensor2lidar_rotation"])
        lidar2cam_t = p["sensor2lidar_translation"] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = copy.deepcopy(p["intrinsics_3x3"])
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img_rt = viewpad @ lidar2cam_rt.T

        # SparseDrive's test-mode resize/crop math (resize covers final_dim, crop centers it)
        H, W = cfg.H, cfg.W
        fH, fW = cfg.final_dim
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(cfg.bot_pct_lim)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": False,
            "rotate": 0,
        }

        img_t, mat = self._img_transform(cam_f0, aug_config)
        lidar2img_rt = mat @ lidar2img_rt
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = mat[:3, :3] @ p["intrinsics_3x3"]

        # Normalize (same RGB stats SD's config defaults to)
        mean = np.array(cfg.img_mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.array(cfg.img_std, dtype=np.float32).reshape(1, 1, -1)
        img_norm = (img_t - mean) / std

        # Stack into (num_cams=1, C, H, W) then add batch
        img_chw = np.ascontiguousarray(img_norm.transpose(2, 0, 1))[None]  # (1, 3, H, W)
        imgs_tensor = torch.from_numpy(img_chw).float().unsqueeze(0).to(self.device)  # (B=1, num_cams=1, 3, H, W)

        # Build the per-batch metadata tensors (B, num_cams, ...)
        proj_mat = torch.from_numpy(np.stack([lidar2img_rt])).float().unsqueeze(0).to(self.device)
        image_wh = torch.from_numpy(
            np.array([[fW, fH]], dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        lidar2img_t = proj_mat
        lidar2cam_t = torch.from_numpy(np.stack([lidar2cam_rt.T])).float().unsqueeze(0).to(self.device)
        cam2lidar_t = torch.from_numpy(np.stack([cam2lidar_rt])).float().unsqueeze(0).to(self.device)
        cam_intr_t = torch.from_numpy(np.stack([intrinsic_4x4])).float().unsqueeze(0).to(self.device)
        distortion_t = torch.from_numpy(np.stack([p["distortion"]])).float().unsqueeze(0).to(self.device)

        return {
            "imgs": imgs_tensor,
            "projection_mat": proj_mat,
            "image_wh": image_wh,
            "lidar2img": lidar2img_t,
            "lidar2cam": lidar2cam_t,
            "cam2lidar": cam2lidar_t,
            "cam_intrinsic": cam_intr_t,
            "distortions": distortion_t,
        }

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        velocity = ego_state["velocity"][:2]
        heading = ego_state["heading"]
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Apply kick-off speed when stationary, as the other NAVSIM adapters do.
        if np.linalg.norm(vel_local) < 0.5:
            vel_local = np.array([2.0, 0.0])

        if "acceleration" in ego_state:
            acc_local = R @ ego_state["acceleration"][:2]
        else:
            acc_local = np.array([0.0, 0.0])

        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        self._current_frame_id = frame_id

        camera_feature = self._preprocess_images(images)

        command = ego_state.get("command", 3)
        status_feature = (
            self._get_status_feature(ego_state, command).unsqueeze(0).to(self.device)
        )

        return {"camera_feature": camera_feature, "status_feature": status_feature}

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            out = self.model(model_input, targets=None)
        # SparseDriveModel.forward returns (output_dict, loss_dict); unwrap.
        if isinstance(out, tuple):
            out = out[0]
        return out

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        trajectory = model_output["trajectory"][0].cpu().numpy()  # (T, 3) = (forward, lateral, heading)
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
        result = {"trajectory": traj_swapped}
        if trajectory.shape[1] > 2:
            result["heading"] = trajectory[:, 2]
        # Candidate proposals + scores, if the model exposes them. base_evaluator
        # expects `proposals` shaped (K, T, 3) in the prediction-time ego frame
        # with axis order (forward, lateral, heading) and `pdm_score` shaped (K,);
        # leave the proposal axes unswapped to match that convention.
        if "proposals" in model_output and "proposal_scores" in model_output:
            result["proposals"] = model_output["proposals"][0].cpu().numpy()  # (K, T, 3)
            result["pdm_score"] = model_output["proposal_scores"][0].cpu().numpy()  # (K,)
        return result

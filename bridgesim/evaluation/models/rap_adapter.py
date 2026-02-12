"""
Model adapter for RAP (Rasterized Autonomous Planner).
Supports both rasterized and MetaDrive camera modes.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

# Monkey patch for PyTorch-transformers compatibility
# Fix: transformers expects register_pytree_node but PyTorch 2.0.x/2.1.0 has _register_pytree_node
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    _original_register = torch.utils._pytree._register_pytree_node

    def _register_pytree_node_wrapper(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return _original_register(cls, flatten_fn, unflatten_fn)

    torch.utils._pytree.register_pytree_node = _register_pytree_node_wrapper

# RAP Imports from modelzoo
from bridgesim.modelzoo.navsim.agents.rap_dino.rap_model import RAPModel
from bridgesim.modelzoo.navsim.agents.rap_dino.navsim_config import RAPConfig

# Import camera params from converters
metabench_root = Path(__file__).resolve().parent.parent.parent.parent
converters_bench2drive_path = metabench_root / "converters" / "bench2drive"
sys.path.insert(0, str(converters_bench2drive_path))
from renderer import camera_params, convert_camera_params_to_simple_format

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD


# RAP Model expects images in this order
CAM_ORDER = ['CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_R0']

# Image Normalization Constants (RGB)
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)
IMG_SCALE = 0.4

# Camera names for MetaDrive
CAM_NAMES = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0']


class RAPAdapter(BaseModelAdapter):
    """
    Adapter for RAP model.

    RAP supports two image sources:
    - 'rasterized': Uses pre-rendered BEV images
    - 'metadrive': Uses MetaDrive camera sensors (4 cameras)
    """

    def __init__(self, checkpoint_path: str, image_source: str = "metadrive", **kwargs):
        """
        Initialize RAP adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            image_source: 'rasterized' or 'metadrive'
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.image_source = image_source
        self.config = None
        self.lidar2img_tensor = None
        self.img_shape_tensor = None

    def load_model(self):
        """Load RAP model."""
        print(f"Loading RAP model (image_source={self.image_source})...")

        # Initialize RAP config
        self.config = RAPConfig()
        self.config.b2d = False
        self.config.num_poses = 10
        self.config.trajectory_sampling.num_poses = 10
        self.config.trajectory_sampling.time_horizon = 5.0

        # Initialize model
        self.model = RAPModel(self.config).to(self.device)
        self.model.progress = 0.0
        self.model.batch_size = 1

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefixes if trained with Lightning/DDP
        clean_sd = {k.replace('agent._rap_model.', '').replace('_rap_model.', ''): v
                   for k, v in state_dict.items()}
        self.model.load_state_dict(clean_sd, strict=False)
        self.model.eval()

        # Precompute calibration features
        self._setup_calibration_features()

        print("RAP model loaded successfully.")

    def _setup_calibration_features(self):
        """Precompute lidar2img matrices and image shapes."""
        lidar2imgs = []
        img_shapes = []

        # Scale Matrix (0.4x resizing)
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = IMG_SCALE
        S[1, 1] = IMG_SCALE

        for cam_name in CAM_ORDER:
            params = camera_params[cam_name]

            # 1. Construct Sensor2Lidar (Extrinsics)
            s2l = np.eye(4, dtype=np.float32)
            s2l[:3, :3] = params['sensor2lidar_rotation']
            s2l[:3, 3] = params['sensor2lidar_translation']

            # 2. Get Lidar2Sensor (Inverse)
            l2s = np.linalg.inv(s2l)

            # 3. Construct Intrinsics (4x4)
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = params['intrinsics']

            # 4. Compute Lidar2Img: S * K * L2S
            l2i = S @ K @ l2s
            lidar2imgs.append(l2i)

            # 5. Image Shape (H, W, 3) after scaling
            h = int(1120 * IMG_SCALE)  # 448
            w = int(1920 * IMG_SCALE)  # 768
            img_shapes.append((h, w, 3))

        # Convert to tensors
        self.lidar2img_tensor = torch.from_numpy(np.stack(lidar2imgs)).float().unsqueeze(0).to(self.device)
        self.img_shape_tensor = torch.from_numpy(np.stack(img_shapes)).float().unsqueeze(0).to(self.device)

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """RAP uses 4 cameras when using MetaDrive mode."""
        if self.image_source == "metadrive":
            # Convert camera params to MetaDrive format
            cam_configs = convert_camera_params_to_simple_format(
                camera_params,
                image_width=1920,
                image_height=1120,
                to_metadrive=True
            )
            return cam_configs
        else:
            # Rasterized mode doesn't use real cameras
            return {}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for RAP model.
        Scale -> Normalize -> Pad
        """
        processed_imgs = []

        for cam_name in CAM_ORDER:
            # Map from CAM_NAMES to CAM_ORDER naming
            md_name = cam_name.replace('_B0', '_BACK').replace('_F0', '_FRONT').replace('_L0', '_FRONT_LEFT').replace('_R0', '_FRONT_RIGHT')
            if 'BACK' in md_name:
                md_name = 'CAM_B0'
            elif md_name == 'CAM_FRONT':
                md_name = 'CAM_F0'
            elif 'LEFT' in md_name:
                md_name = 'CAM_L0'
            elif 'RIGHT' in md_name:
                md_name = 'CAM_R0'

            img = images_dict[md_name]

            # 1. RandomScale(0.4)
            h, w = img.shape[:2]
            new_h, new_w = int(h * IMG_SCALE), int(w * IMG_SCALE)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 2. Transpose (H, W, C) -> (C, H, W)
            img = img.transpose(2, 0, 1)
            processed_imgs.append(img)

        # Stack cameras: (4, 3, H, W)
        img_tensor = torch.from_numpy(np.stack(processed_imgs)).float()

        # 3. Normalize (using RGB means)
        img_tensor = (img_tensor - IMG_MEAN.to(img_tensor.device)) / IMG_STD.to(img_tensor.device)

        # 4. PadMultiViewImage (Divisor 32)
        _, _, h, w = img_tensor.shape
        pad_h = int(np.ceil(h / 32)) * 32
        pad_w = int(np.ceil(w / 32)) * 32

        if pad_h != h or pad_w != w:
            padding = torch.nn.ZeroPad2d((0, pad_w - w, 0, pad_h - h))
            img_tensor = padding(img_tensor)

        return img_tensor

    def _get_ego_status(self, velocity: np.ndarray, acceleration: np.ndarray, command: np.ndarray) -> torch.Tensor:
        """
        Constructs ego_status tensor [11].
        Format: [pose(3), vel(2), acc(2), command(4)]

        RAP model uses local frame where current pose is [0, 0, 0].
        """
        pose = torch.zeros(3, dtype=torch.float32)
        vel = torch.tensor(velocity, dtype=torch.float32)
        acc = torch.tensor(acceleration, dtype=torch.float32)
        cmd = torch.tensor(command, dtype=torch.float32)

        return torch.cat([pose, vel, acc, cmd], dim=0)

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for RAP model."""
        # 1. Preprocess images
        img_tensor = self._preprocess_images(images).unsqueeze(0).to(self.device)

        # 2. Compute velocity and acceleration in local frame
        velocity = ego_state['velocity']
        heading = ego_state['heading']

        # Rotation Matrix Global -> Ego
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])

        vel_local = R @ velocity[:2]

        # Compute acceleration (if previous velocity is available, use finite difference)
        if hasattr(self, '_prev_velocity'):
            acc_global = (velocity[:2] - self._prev_velocity[:2]) / 0.1  # 10Hz
            acc_local = R @ acc_global
        else:
            acc_local = np.array([0.0, 0.0])
        self._prev_velocity = velocity

        # 3. Get command
        command = ego_state['command']
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        # 4. Prepare ego status
        curr_status = self._get_ego_status(vel_local, acc_local, cmd_vec)
        # Stack 4 times to mimic history
        ego_status_tensor = torch.stack([curr_status] * 4).unsqueeze(0).to(self.device)

        # 5. Build inputs
        inputs = {
            "camera_feature": img_tensor,
            "ego_status": ego_status_tensor,
            "camera_valid": torch.tensor([True]).to(self.device),
            "lidar2img": self.lidar2img_tensor,
            "img_shape": self.img_shape_tensor
        }

        return inputs

    def run_inference(self, model_input: Any) -> Any:
        """Run RAP model inference."""
        with torch.no_grad():
            output = self.model(model_input, targets=None)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse RAP output."""
        # Trajectory shape: (B, M, T, 2) -> Select best mode
        # Usually 'trajectory' key in output contains the selected best proposal
        pred_traj_full = model_output["trajectory"][0].cpu().numpy()  # Shape might be (T, 2) or (T, 3)

        print(f"[RAP DEBUG] Raw trajectory: {pred_traj_full[:3]}...")  # First 3 points

        # RAP outputs [forward, lateral, heading]
        # Swap columns: [forward, lateral] -> [lateral, forward]
        if pred_traj_full.ndim == 2 and pred_traj_full.shape[1] >= 2:
            pred_traj_local = np.column_stack([pred_traj_full[:, 1], pred_traj_full[:, 0]])
        else:
            pred_traj_local = pred_traj_full

        print(f"[RAP DEBUG] Swapped trajectory: {pred_traj_local[:3]}...")  # First 3 points

        return {'trajectory': pred_traj_local}

    def get_trajectory_time_horizon(self) -> float:
        """RAP predicts trajectory with 5.0s horizon (10 poses at 0.5s intervals)."""
        return 5.0

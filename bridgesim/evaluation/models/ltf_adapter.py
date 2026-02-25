"""
Model adapter for LTF (Latent TransFuser) model from bridgesim.modelzoo.navsim.

LTF is a variant of TransFuser that uses positional encoding instead of LiDAR,
making it an image-only model. Enable by setting config.latent=True.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_model import TransfuserModel
from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_config import TransfuserConfig

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD


class LTFAdapter(BaseModelAdapter):
    """
    Adapter for LTF (Latent TransFuser) model from bridgesim.modelzoo.navsim.

    LTF uses multi-camera images with positional encoding instead of LiDAR,
    making it an image-only autonomous driving model.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize LTF adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        self.trajectory_sampling = None

    def load_model(self):
        """Load LTF model from checkpoint."""
        print("Loading LTF (Latent TransFuser) model...")

        # Initialize config with latent=True for LTF
        self.config = TransfuserConfig()
        self.config.latent = True  # Key difference from TransFuser
        self.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

        # Initialize model
        self.model = TransfuserModel(self.trajectory_sampling, self.config)

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefixes if trained with Lightning/DDP
        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        self.model.load_state_dict(clean_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print("LTF model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """LTF uses 3 cameras (left, front, right) stitched together."""
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0')}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for LTF model.
        LTF expects a stitched panoramic image (left + front + right) resized to 1024x256.
        """
        # Get individual camera images
        cam_l0 = images_dict.get('CAM_L0')
        cam_f0 = images_dict.get('CAM_F0')
        cam_r0 = images_dict.get('CAM_R0')

        # Create dummy images if missing
        dummy_h, dummy_w = 1080, 1920
        if cam_l0 is None:
            cam_l0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_f0 is None:
            cam_f0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_r0 is None:
            cam_r0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)

        # Crop images to ensure proper stitching (matching navsim transfuser_features.py)
        h, w = cam_f0.shape[:2]
        crop_tb = int(28 * h / 1080)  # Scale crop based on actual image height
        crop_lr = int(416 * w / 1920)  # Scale crop based on actual image width

        # Crop: [top:bottom, left:right]
        l0_cropped = cam_l0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_l0[:, crop_lr:-crop_lr]
        f0_cropped = cam_f0[crop_tb:-crop_tb] if crop_tb > 0 else cam_f0
        r0_cropped = cam_r0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_r0[:, crop_lr:-crop_lr]

        # Stitch images horizontally: left + front + right
        stitched_image = np.concatenate([l0_cropped, f0_cropped, r0_cropped], axis=1)

        # Resize to expected size (1024, 256)
        resized_image = cv2.resize(stitched_image, (1024, 256))

        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()

        # Normalize (RGB values 0-255 to 0-1)
        tensor_image = tensor_image / 255.0

        return tensor_image

    def _create_lidar_bev(self) -> torch.Tensor:
        """
        Create dummy LiDAR BEV representation.

        Note: LTF uses positional encoding instead of real LiDAR,
        but still expects the lidar_feature tensor in the input dict.
        The model internally replaces this with learned positional encoding.
        """
        lidar_bev = torch.zeros(
            self.config.lidar_seq_len,
            self.config.lidar_resolution_height,
            self.config.lidar_resolution_width,
            dtype=torch.float32
        )
        return lidar_bev

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        """
        Create status feature tensor.
        Format: [command(4), velocity(2), acceleration(2)]
        """
        # Command one-hot
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        # Velocity in local frame
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Acceleration (if available)
        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0])

        # Combine: [cmd(4), vel(2), acc(2)] = 8
        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for LTF model."""
        # 1. Preprocess images
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)

        # 2. Create dummy LiDAR BEV (LTF uses positional encoding instead)
        lidar_feature = self._create_lidar_bev().unsqueeze(0).to(self.device)

        # 3. Get status feature
        command = ego_state.get('command', 3)
        status_feature = self._get_status_feature(ego_state, command).unsqueeze(0).to(self.device)

        inputs = {
            "camera_feature": camera_feature,
            "lidar_feature": lidar_feature,
            "status_feature": status_feature,
        }

        return inputs

    def run_inference(self, model_input: Any) -> Any:
        """Run LTF model inference."""
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse LTF output."""
        # Extract trajectory from output
        trajectory = model_output["trajectory"][0].cpu().numpy()  # (T, 3) -> (x, y, heading)

        # Swap columns: LTF outputs [forward, lateral] but evaluator expects [lateral, forward]
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])

        return {'trajectory': traj_swapped}

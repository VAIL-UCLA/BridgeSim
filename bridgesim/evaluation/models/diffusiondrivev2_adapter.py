"""
Model adapter for DiffusionDriveV2 model from bridgesim.modelzoo.navsim.
DiffusionDriveV2 uses diffusion-based trajectory prediction with scoring module.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict


from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel
from bridgesim.modelzoo.navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


# Image Normalization Constants (RGB)
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)

# Command Mapping
CMD_MAPPING = {
    0: np.array([1, 0, 0, 0], dtype=np.float32),  # Left
    1: np.array([0, 1, 0, 0], dtype=np.float32),  # Forward
    2: np.array([0, 0, 1, 0], dtype=np.float32),  # Right
}
DEFAULT_CMD = np.array([0, 1, 0, 0], dtype=np.float32)


class DiffusionDriveV2Adapter(BaseModelAdapter):
    """
    Adapter for DiffusionDriveV2 model from bridgesim.modelzoo.navsim.

    DiffusionDriveV2 uses diffusion-based trajectory generation with a scoring module
    for trajectory selection.
    """

    def __init__(
        self,
        checkpoint_path: str,
        plan_anchor_path: str = None,
        bev_calibrator=None,
        enable_temporal_consistency: bool = False,
        temporal_alpha: float = 1.5,
        temporal_lambda: float = 0.3,
        temporal_max_history: int = 8,
        temporal_sigma: float = 5.0,
        consensus_temperature: float = 1.0,
        **kwargs
    ):
        """
        Initialize DiffusionDriveV2 adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            plan_anchor_path: Path to plan anchor numpy file
            bev_calibrator: Optional TransfuserBEVCalibrator for domain adaptation
            enable_temporal_consistency: Whether to use temporal consistency scoring
            temporal_alpha: Decay base for temporal weighting (older = more weight)
            temporal_lambda: Weight for temporal consistency in combined score
            temporal_max_history: Max number of past trajectories to store
            temporal_sigma: Position normalization factor (meters)
            consensus_temperature: Softmax temperature for consensus trajectory weighting.
                Lower values make consensus dominated by highest-PDM trajectory.
                Higher values make consensus closer to uniform mean. (default: 1.0)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        self.plan_anchor_path = plan_anchor_path
        self.bev_calibrator = bev_calibrator

        # Temporal consistency parameters
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_alpha = temporal_alpha
        self.temporal_lambda = temporal_lambda
        self.temporal_max_history = temporal_max_history
        self.temporal_sigma = temporal_sigma
        self.consensus_temperature = consensus_temperature
        self.current_sim_time = 0.0  # Track current simulation time

    def load_model(self):
        """Load DiffusionDriveV2 model from checkpoint."""
        print("Loading DiffusionDriveV2 model...")

        # Initialize config
        self.config = TransfuserConfig()

        # Override plan_anchor_path if provided
        if self.plan_anchor_path:
            self.config.plan_anchor_path = self.plan_anchor_path

        # Initialize model
        self.model = V2TransfuserModel(self.config)

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

        # Set temporal consistency parameters if enabled
        if self.enable_temporal_consistency:
            self.model.set_temporal_params(
                alpha=self.temporal_alpha,
                lambda_consist=self.temporal_lambda,
                max_history=self.temporal_max_history,
                sigma=self.temporal_sigma,
                consensus_temperature=self.consensus_temperature,
            )
            print(f"Temporal consistency enabled: alpha={self.temporal_alpha}, "
                  f"lambda={self.temporal_lambda}, max_history={self.temporal_max_history}, "
                  f"sigma={self.temporal_sigma}, consensus_temp={self.consensus_temperature}")

        # Load and inject BEV calibrator if provided
        if self.bev_calibrator is not None:
            print("Loading BEV calibrator for domain adaptation...")
            self.bev_calibrator.load_model()
            self.model.bev_calibrator = self.bev_calibrator
            print("BEV calibrator injected into model.")

        print("DiffusionDriveV2 model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """DiffusionDriveV2 uses 3 cameras (left, front, right) stitched together."""
        return {
            'CAM_F0': {'x': 1.3, 'y': 0.0, 'z': 2.3, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
            'CAM_L0': {'x': 1.3, 'y': -0.5, 'z': 2.3, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
            'CAM_R0': {'x': 1.3, 'y': 0.5, 'z': 2.3, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
        }

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for DiffusionDriveV2 model.
        DiffusionDriveV2 expects a stitched panoramic image (left + front + right) resized to 1024x256.
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
        resized_image = cv2.resize(stitched_image, (self.config.camera_width, self.config.camera_height))

        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()

        # Normalize (RGB values 0-255 to 0-1)
        tensor_image = tensor_image / 255.0

        return tensor_image

    def _create_lidar_bev(self) -> torch.Tensor:
        """Create dummy LiDAR BEV representation."""
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
        cmd_vec = CMD_MAPPING.get(command, DEFAULT_CMD)

        # Velocity in local frame
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if np.linalg.norm(vel_local) < 0.5:
            vel_local[0] = KICKOFF_SPEED

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
        """Prepare input for DiffusionDriveV2 model."""
        # 1. Preprocess images
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)

        # 2. Create LiDAR BEV
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
        """Run DiffusionDriveV2 model inference."""
        use_bev_calibrator = self.bev_calibrator is not None
        with torch.no_grad():
            if self.enable_temporal_consistency:
                output = self.model.forward_temporal(
                    model_input,
                    current_time=self.current_sim_time,
                    targets=None,
                    cal_pdm=False,
                    use_bev_calibrator=use_bev_calibrator,
                )
            else:
                output = self.model(model_input, targets=None, cal_pdm=False, use_bev_calibrator=use_bev_calibrator)
        return output

    def set_simulation_time(self, sim_time: float):
        """Set current simulation time for temporal consistency scoring."""
        self.current_sim_time = sim_time

    def reset_temporal_history(self):
        """Reset temporal consistency history. Call at start of new scenario."""
        if self.enable_temporal_consistency and self.model is not None:
            self.model.reset_temporal_history()
            self.current_sim_time = 0.0
            print("Temporal consistency history reset.")

    def get_temporal_history_length(self) -> int:
        """Get current length of temporal history buffer."""
        if self.enable_temporal_consistency and self.model is not None:
            return self.model.get_temporal_history_length()
        return 0

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse DiffusionDriveV2 output."""
        result = {}

        # Extract best trajectory from output
        if "trajectory" in model_output:
            trajectory = model_output["trajectory"][0].cpu().numpy()  # (T, 3) -> (x, y, heading)
            # Swap columns: model outputs [forward, lateral] but evaluator expects [lateral, forward]
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            result['trajectory'] = traj_swapped
        else:
            # Fallback: return empty trajectory
            result['trajectory'] = np.zeros((8, 2), dtype=np.float32)

        # Extract all candidate trajectories for visualization
        # trajectory_candidates shape: (B, 4, 8, 3) -> 4 candidates: 1 coarse best + 3 fine bests
        if "trajectory_candidates" in model_output:
            candidates = model_output["trajectory_candidates"][0].cpu().numpy()  # (4, 8, 3)
            # Swap columns for each candidate
            candidates_swapped = []
            for i in range(candidates.shape[0]):
                cand = candidates[i]  # (8, 3) -> (x, y, heading)
                cand_swapped = np.column_stack([cand[:, 1], cand[:, 0]])  # (8, 2)
                candidates_swapped.append(cand_swapped)
            result['trajectory_candidates'] = np.array(candidates_swapped)  # (4, 8, 2)

        # Extract top-32 candidate trajectories for visualization
        # trajectory_topk shape: (B, 32, 8, 3) -> 32 top candidates from coarse scoring
        if "trajectory_topk" in model_output:
            topk = model_output["trajectory_topk"][0].cpu().numpy()  # (32, 8, 3)
            # Swap columns for each candidate
            topk_swapped = []
            for i in range(topk.shape[0]):
                cand = topk[i]  # (8, 3) -> (x, y, heading)
                cand_swapped = np.column_stack([cand[:, 1], cand[:, 0]])  # (8, 2)
                topk_swapped.append(cand_swapped)
            result['trajectory_topk'] = np.array(topk_swapped)  # (32, 8, 2)

        # Extract top-32 scores for visualization
        if "topk_scores" in model_output:
            result['topk_scores'] = model_output["topk_scores"][0].cpu().numpy()  # (32,)

        # Extract all coarse candidate trajectories for visualization
        # trajectory_coarse shape: (B, 200, 8, 3) -> all 200 coarse candidates
        if "trajectory_coarse" in model_output:
            coarse = model_output["trajectory_coarse"][0].cpu().numpy()  # (200, 8, 3)
            # Swap columns for each candidate
            coarse_swapped = []
            for i in range(coarse.shape[0]):
                cand = coarse[i]  # (8, 3) -> (x, y, heading)
                cand_swapped = np.column_stack([cand[:, 1], cand[:, 0]])  # (8, 2)
                coarse_swapped.append(cand_swapped)
            result['trajectory_coarse'] = np.array(coarse_swapped)  # (200, 8, 2)

        # Extract coarse scores for visualization
        if "coarse_scores" in model_output:
            result['coarse_scores'] = model_output["coarse_scores"][0].cpu().numpy()  # (200,)

        # Extract temporal consistency scores (if available)
        if "pdm_coarse_scores" in model_output:
            result['pdm_coarse_scores'] = model_output["pdm_coarse_scores"][0].cpu().numpy()

        if "temporal_coarse_scores" in model_output:
            result['temporal_coarse_scores'] = model_output["temporal_coarse_scores"][0].cpu().numpy()

        if "temporal_history_length" in model_output:
            result['temporal_history_length'] = model_output["temporal_history_length"]

        return result

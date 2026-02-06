"""
Model adapter for DrivoR (Driving with Routing) model from bridgesim.modelzoo.navsim.
DrivoR uses DINOv2 with LoRA for image encoding and transformer-based trajectory prediction.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf


from bridgesim.modelzoo.navsim.agents.drivoR.drivor_model import DrivoRModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


# Image Normalization Constants (ImageNet RGB)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Command Mapping (MetaDrive to DrivoR one-hot)
CMD_MAPPING = {
    0: np.array([0, 1, 0, 0], dtype=np.float32),  # 0 -> Forward
    1: np.array([1, 0, 0, 0], dtype=np.float32),  # 1 -> Left
    2: np.array([0, 0, 1, 0], dtype=np.float32),  # 2 -> Right
}
DEFAULT_CMD = np.array([0, 1, 0, 0], dtype=np.float32)


def create_drivor_config(
    num_cameras: int = 4,
    image_size: tuple = (1148, 672),  # Match checkpoint training size
    num_poses: int = 8,
    proposal_num: int = 64,
    use_lidar: bool = False,
) -> OmegaConf:
    """
    Create a default DrivoR configuration.

    Args:
        num_cameras: Number of cameras to use (4 or 8)
        image_size: (width, height) for input images
        num_poses: Number of trajectory poses to predict
        proposal_num: Number of trajectory proposals
        use_lidar: Whether to use LiDAR input

    Returns:
        OmegaConf configuration object
    """
    # Default camera configuration based on num_cameras
    if num_cameras == 4:
        cam_config = {
            "cam_f0": [3],  # front
            "cam_l0": [3],  # left
            "cam_l1": [],
            "cam_l2": [],
            "cam_r0": [3],  # right
            "cam_r1": [],
            "cam_r2": [],
            "cam_b0": [3],  # back
        }
    else:  # 8 cameras
        cam_config = {
            "cam_f0": [3],
            "cam_l0": [3],
            "cam_l1": [3],
            "cam_l2": [3],
            "cam_r0": [3],
            "cam_r1": [3],
            "cam_r2": [3],
            "cam_b0": [3],
        }

    config_dict = {
        # Camera configuration
        **cam_config,
        "lidar_pc": [3] if use_lidar else [],

        # Image settings
        "image_size": list(image_size),  # (width, height)
        "lidar_image_size": [256, 256],

        # Model architecture
        "num_scene_tokens": 16,
        "tf_d_model": 256,  # Match checkpoint
        "tf_d_ffn": 1024,
        "num_poses": num_poses,
        "proposal_num": proposal_num,
        "ref_num": 4,
        "scorer_ref_num": 2,

        # Training settings
        "full_history_status": False,
        "one_token_per_traj": True,
        "b2d": True,  # Bench2Drive mode

        # Trajectory sampling
        "trajectory_sampling": {
            "num_poses": num_poses,
            "time_horizon": 4.0,
            "interval_length": 0.5,
        },

        # Image backbone (DINOv2 ViT-Small)
        "image_backbone": {
            "model_name": "timm/vit_small_patch14_dinov2.lvd142m",
            "model_weights": None,  # Let timm download automatically
            "use_lora": True,
            "lora_rank": 32,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },

        # LiDAR backbone (uses same ImgEncoder class)
        "lidar_backbone": {
            "model_name": "timm/vit_small_patch14_dinov2.lvd142m",
            "model_weights": None,
            "use_lora": False,
            "lora_rank": 0,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },

        # LiDAR settings (for feature builder)
        "lidar_min_x": -32,
        "lidar_max_x": 32,
        "lidar_min_y": -32,
        "lidar_max_y": 32,
        "lidar_max_height": 2.0,
        "lidar_split_height": 0.2,
        "lidar_use_ground_plane": True,
        "lidar_hist_max_per_pixel": 5,

        # Scorer weights
        "noc": 5.0,
        "dac": 5.0,
        "ddc": 5.0,
        "ttc": 2.0,
        "ep": 2.0,
        "comfort": 2.0,

        # Scorer settings
        "double_score": False,
        "agent_pred": False,
        "area_pred": False,
        "bev_map": False,
        "bev_agent": False,

        # Additional settings
        "long_trajectory_additional_poses": 0,
    }

    return OmegaConf.create(config_dict)


class DrivoRAdapter(BaseModelAdapter):
    """
    Adapter for DrivoR model from bridgesim.modelzoo.navsim.

    DrivoR uses DINOv2 with LoRA for image encoding and a transformer decoder
    for multi-proposal trajectory prediction with scoring.
    """

    # Model training image size (must match checkpoint)
    MODEL_IMAGE_SIZE = (1148, 672)  # (width, height) - fixed to match checkpoint

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        num_cameras: int = 4,
        image_size: tuple = None,  # Ignored - model uses fixed size
        num_poses: int = 8,
        use_lidar: bool = False,
        **kwargs
    ):
        """
        Initialize DrivoR adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            config_path: Path to config yaml (optional, uses default if not provided)
            num_cameras: Number of cameras (4 or 8)
            image_size: Ignored - model uses fixed size (1148, 672) to match checkpoint
            num_poses: Number of trajectory poses
            use_lidar: Whether to use LiDAR input
        """
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.num_cameras = num_cameras
        # Always use fixed image size that matches checkpoint training
        self.image_size = self.MODEL_IMAGE_SIZE
        self.num_poses = num_poses
        self.use_lidar = use_lidar
        self.config = None

        # Camera order for processing (front first, then others)
        if num_cameras == 4:
            self.cam_order = ['CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_R0']
        else:
            self.cam_order = ['CAM_F0', 'CAM_B0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2']

    def load_model(self):
        """Load DrivoR model from checkpoint."""
        print("Loading DrivoR model...")

        # Load or create config
        if self.config_path and Path(self.config_path).exists():
            self.config = OmegaConf.load(self.config_path)
        else:
            self.config = create_drivor_config(
                num_cameras=self.num_cameras,
                image_size=self.image_size,
                num_poses=self.num_poses,
                use_lidar=self.use_lidar,
            )

        # Initialize model
        self.model = DrivoRModel(self.config)

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefixes if trained with Lightning/DDP
        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._drivor_model.', '').replace('_drivor_model.', '')
            clean_sd[new_key] = v

        # Load with strict=False to handle missing/extra keys gracefully
        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)}")

        self.model.to(self.device)
        self.model.eval()

        print("DrivoR model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Return camera configuration for DrivoR."""
        if self.num_cameras == 4:
            return {
                'CAM_F0': {'x': 1.3, 'y': 0.0, 'z': 2.3, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_L0': {'x': 1.3, 'y': -0.5, 'z': 2.3, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_R0': {'x': 1.3, 'y': 0.5, 'z': 2.3, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_B0': {'x': -1.3, 'y': 0.0, 'z': 2.3, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1920, 'height': 1080},
            }
        else:
            # 8 camera configuration
            return {
                'CAM_F0': {'x': 1.3, 'y': 0.0, 'z': 2.3, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_L0': {'x': 1.3, 'y': -0.5, 'z': 2.3, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_L1': {'x': 0.0, 'y': -0.8, 'z': 2.3, 'yaw': -90.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_L2': {'x': -1.0, 'y': -0.5, 'z': 2.3, 'yaw': -135.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_R0': {'x': 1.3, 'y': 0.5, 'z': 2.3, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_R1': {'x': 0.0, 'y': 0.8, 'z': 2.3, 'yaw': 90.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_R2': {'x': -1.0, 'y': 0.5, 'z': 2.3, 'yaw': 135.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
                'CAM_B0': {'x': -1.3, 'y': 0.0, 'z': 2.3, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1920, 'height': 1080},
            }

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for DrivoR model.

        Args:
            images_dict: Dictionary mapping camera names to RGB images

        Returns:
            Tensor of shape (num_cams, 3, H, W)
        """
        processed_imgs = []
        target_width, target_height = self.image_size

        for cam_name in self.cam_order:
            if cam_name in images_dict:
                img = images_dict[cam_name]
            else:
                # Create dummy image if camera missing
                img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            # Resize to target dimensions
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Convert to float and normalize (ImageNet stats)
            img = img.astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD

            # Transpose to (C, H, W)
            img = img.transpose(2, 0, 1)
            processed_imgs.append(img)

        # Stack cameras: (num_cams, 3, H, W)
        img_tensor = torch.from_numpy(np.stack(processed_imgs)).float()

        return img_tensor

    def _get_ego_status(
        self,
        ego_state: Dict[str, Any],
        command: int
    ) -> torch.Tensor:
        """
        Create ego status feature tensor.

        Format: [pose(3), velocity(2), acceleration(2), command(4)] = 11 dims

        Args:
            ego_state: Dictionary with ego vehicle state
            command: Navigation command index

        Returns:
            Tensor of shape (11,) or (4, 11) if full_history_status
        """
        # Pose in local frame (current frame is origin)
        pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Get velocity in local frame
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if np.linalg.norm(vel_local) < 0.5:
            vel_local[0] = KICKOFF_SPEED

        # Get acceleration in local frame
        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0], dtype=np.float32)

        # Command one-hot
        cmd_vec = CMD_MAPPING.get(command, DEFAULT_CMD)

        # Combine: [pose(3), vel(2), acc(2), cmd(4)] = 11
        ego_status = np.concatenate([pose, vel_local, acc_local, cmd_vec]).astype(np.float32)

        # If full_history_status, stack 4 times
        if self.config.get('full_history_status', False):
            ego_status = np.tile(ego_status, (4, 1))  # (4, 11)

        return torch.from_numpy(ego_status)

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int
    ) -> Any:
        """
        Prepare input for DrivoR model.

        Args:
            images: Dictionary mapping camera names to RGB images
            ego_state: Current ego vehicle state
            scenario_data: Full scenario metadata
            frame_id: Current frame index

        Returns:
            Dictionary with model inputs
        """
        # 1. Preprocess images
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)

        # 2. Get command
        command = ego_state.get('command', 3)  # Default to forward

        # 3. Get ego status - shape: (batch, time, 11)
        # Model expects [:, -1] to get last timestep, so we need (batch, time, features)
        ego_status = self._get_ego_status(ego_state, command).unsqueeze(0).unsqueeze(0).to(self.device)
        # Shape: (1, 1, 11) - batch=1, time=1, features=11

        # 4. Build input dictionary
        inputs = {
            "image": camera_feature,
            "ego_status": ego_status,
        }

        # 5. Add LiDAR features if needed (placeholder - zeros)
        if self.use_lidar:
            lidar_size = self.config.lidar_image_size
            lidar_feature = torch.zeros(
                1, 1, 2, lidar_size[0], lidar_size[1],
                dtype=torch.float32, device=self.device
            )
            inputs["lidar_feature"] = lidar_feature

        return inputs

    def run_inference(self, model_input: Any) -> Any:
        """Run DrivoR model inference."""
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def parse_output(
        self,
        model_output: Any,
        ego_state: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Parse DrivoR model output.

        Args:
            model_output: Raw model output dictionary
            ego_state: Current ego state

        Returns:
            Dictionary with trajectory and optional scores
        """
        # Extract best trajectory (already selected by model based on PDM scores)
        trajectory = model_output["trajectory"][0].cpu().numpy()  # (num_poses, 3)

        # Extract x, y coordinates and swap columns (model outputs [forward, lateral] but evaluator expects [lateral, forward])
        trajectory_xy = np.column_stack([trajectory[:, 1], trajectory[:, 0]])

        # Optionally include heading
        heading = trajectory[:, 2] if trajectory.shape[1] > 2 else None

        result = {'trajectory': trajectory_xy}

        if heading is not None:
            result['heading'] = heading

        # Include proposals and scores if available (for debugging/visualization)
        if "proposals" in model_output:
            result['proposals'] = model_output["proposals"][0].cpu().numpy()

        if "pdm_score" in model_output:
            result['pdm_score'] = model_output["pdm_score"][0].cpu().numpy()

        return result

    def save_intermediate_outputs(self, model_output: Any, output_path: str):
        """
        Save intermediate DrivoR outputs for debugging/visualization.

        Args:
            model_output: Raw model output
            output_path: Directory to save outputs
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save trajectory
        if "trajectory" in model_output:
            np.save(output_path / "trajectory.npy", model_output["trajectory"][0].cpu().numpy())

        # Save all proposals
        if "proposals" in model_output:
            np.save(output_path / "proposals.npy", model_output["proposals"][0].cpu().numpy())

        # Save PDM scores
        if "pdm_score" in model_output:
            np.save(output_path / "pdm_scores.npy", model_output["pdm_score"][0].cpu().numpy())

        # Save individual score components
        if "pred_logit" in model_output:
            pred_logit = model_output["pred_logit"]
            for key, value in pred_logit.items():
                np.save(output_path / f"score_{key}.npy", value[0].cpu().numpy())

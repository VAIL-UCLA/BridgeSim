"""
Model adapter for TCP (Trajectory-guided Control Prediction).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict
from torchvision import transforms as T

# TCP model imports from modelzoo
from bridgesim.modelzoo.bench2drive.adzoo.TCP.model import TCP
from bridgesim.modelzoo.bench2drive.adzoo.TCP.config import GlobalConfig

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


class TCPAdapter(BaseModelAdapter):
    """
    Adapter for TCP model.

    TCP uses only the front camera (CAM_FRONT) and predicts waypoints + control signals.
    Note: Original ResNet34 backbone expects 3-channel input (single RGB image).
    """

    def __init__(self, checkpoint_path: str, planner_type: str = "only_traj", **kwargs):
        """
        Initialize TCP adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            planner_type: One of 'only_ctrl', 'only_traj', 'merge_ctrl_traj'
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.planner_type = planner_type
        self.config = None
        self.img_normalize = None

    def load_model(self):
        """Load TCP model from PyTorch Lightning checkpoint."""
        print(f"Loading TCP model with planner_type={self.planner_type}...")

        # Initialize TCP model
        self.config = GlobalConfig()
        self.model = TCP(self.config)

        # Load PyTorch Lightning checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cuda")
        state_dict = ckpt["state_dict"]

        # Remove "model." prefix from state dict keys
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value

        # Load state dict
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.cuda()
        self.model.eval()

        # Image normalization
        self.img_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        print(f"TCP model loaded successfully (pred_len={self.config.pred_len})")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """TCP uses only 3 front cameras."""
        return {
            'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
        }

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for TCP model."""
        # Process images
        # TCP uses only front camera (ResNet expects 3 channels)
        img = images['CAM_FRONT']
        # Convert BGR to RGB
        img_rgb = img[:, :, ::-1].copy()
        # Resize to 256x256
        img_resized = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_resized = torch.nn.functional.interpolate(
            img_resized.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        # Normalize
        img_normalized = self.img_normalize(img_resized)

        # Add batch dimension: [1, 3, 256, 256]
        img_tensor = img_normalized.unsqueeze(0).cuda()

        # Compute target point in ego frame
        delta_world = ego_state['waypoint'] - ego_state['position'][:2]
        cos_h = np.cos(-ego_state['heading'])
        sin_h = np.sin(-ego_state['heading'])
        target_ego_x = cos_h * delta_world[0] - sin_h * delta_world[1]
        target_ego_y = sin_h * delta_world[0] + cos_h * delta_world[1]
        target_point = torch.FloatTensor([[target_ego_x, target_ego_y]]).cuda()

        # Speed normalized by 12
        speed = np.linalg.norm(ego_state['velocity'])
        speed_normalized = torch.FloatTensor([[speed / 12.0]]).cuda()

        # Command one-hot (6 classes: 0-5)
        command = ego_state['command']
        if command < 0:
            command = 3  # Default to LANEFOLLOW
        cmd_one_hot = torch.zeros(1, 6).cuda()
        cmd_one_hot[0, command] = 1.0

        # Concatenate state: [speed, target_x, target_y, cmd_one_hot] = [1, 9]
        state = torch.cat([speed_normalized, target_point, cmd_one_hot], 1)

        return {
            'img': img_tensor,
            'state': state,
            'target_point': target_point
        }

    def run_inference(self, model_input: Any) -> Any:
        """Run TCP model inference."""
        with torch.no_grad():
            pred = self.model(
                model_input['img'],
                model_input['state'],
                model_input['target_point']
            )

        # TCP outputs waypoints in pixel coordinates - convert to meters
        PIXELS_PER_METER = 5.5
        pred_wp_pixels = pred['pred_wp']  # [1, pred_len, 2]
        pred_wp_meters = pred_wp_pixels / PIXELS_PER_METER

        return {
            'pred_wp': pred_wp_meters,
            'pred_speed': pred['pred_speed'],
            'action_index': pred['action_index']
        }

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse TCP output."""
        # Extract waypoints
        pred_wp = model_output['pred_wp']  # [1, pred_len, 2]

        if self.planner_type == "only_traj":
            # Use predicted waypoints directly
            plan_traj = pred_wp[0].cpu().numpy()  # [pred_len, 2]

        elif self.planner_type == "only_ctrl":
            # Use control signals (not commonly used)
            # For now, fall back to trajectory
            plan_traj = pred_wp[0].cpu().numpy()

        elif self.planner_type == "merge_ctrl_traj":
            # Merge control and trajectory (not commonly used)
            # For now, fall back to trajectory
            plan_traj = pred_wp[0].cpu().numpy()

        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")

        return {'trajectory': plan_traj}

    def get_trajectory_time_horizon(self) -> float:
        """TCP predicts 4 waypoints at 0.5s intervals = 2.0s horizon."""
        return 2.0

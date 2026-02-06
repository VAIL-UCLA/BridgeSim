"""
Model adapter for LEAD NavSim (LTFv6) model.

This adapter uses the self-contained ltfv6.py model definition bundled with the checkpoint.
LEAD NavSim uses 4 cameras with 1920x270 resolution and 4 discrete commands.
"""

import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


class LEADNavsimAdapter(BaseModelAdapter):
    """
    Adapter for LEAD NavSim (LTFv6) model.

    Uses 4 cameras (front, front-left, front-right, back) with 1920x270 resolution.
    This is the NavSim-trained version of LEAD with Latent TransFuser backbone.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize LEAD NavSim adapter.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        # Expected image dimensions from config
        self.image_width = 1920
        self.image_height = 270

    def load_model(self):
        """Load LEAD NavSim model from checkpoint."""
        print("Loading LEAD NavSim (LTFv6) model...")

        # Get the directory containing the checkpoint (should have ltfv6.py and config.json)
        checkpoint_dir = Path(self.checkpoint_path).parent

        # Add checkpoint directory to path to import ltfv6
        sys.path.insert(0, str(checkpoint_dir))

        # Import the model loader from the bundled ltfv6.py
        from ltfv6 import load_tf

        # Load model using the bundled loader
        self.model = load_tf(self.checkpoint_path, torch.device(self.device))

        # Store config reference
        self.config = self.model.config

        # Force float32 mode (override auto-detected bfloat16 for A100/L40S)
        # This ensures consistent dtype between model weights and inputs
        type(self.config).torch_float_type = property(lambda self: torch.float32)
        self.model = self.model.float()
        print("Forced float32 mode for inference")

        # Update image dimensions from config if available
        if hasattr(self.config, 'final_image_width'):
            self.image_width = self.config.final_image_width
        if hasattr(self.config, 'final_image_height'):
            self.image_height = self.config.final_image_height

        print(f"Image dimensions: {self.image_width}x{self.image_height}")
        print("LEAD NavSim model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """LEAD NavSim uses 4 cameras."""
        return {
            'CAM_F0': {'x': 1.3, 'y': 0.0, 'z': 2.3, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
            'CAM_L0': {'x': 1.3, 'y': -0.5, 'z': 2.3, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
            'CAM_R0': {'x': 1.3, 'y': 0.5, 'z': 2.3, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1920, 'height': 1080},
            'CAM_B0': {'x': -1.3, 'y': 0.0, 'z': 2.3, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1920, 'height': 1080},
        }

    def _preprocess_rgb(self, camera_images: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess multi-camera RGB images for LEAD NavSim model (4 cameras).

        The model expects a stitched panoramic image of shape (3, H, W) where
        W = 1920 (4 cameras * 480 each) and H = 270.
        """
        # Camera order for NavSim 4-camera setup
        cam_order = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_B0']

        processed_cams = []
        cam_width = self.image_width // 4  # 480 per camera

        for cam_name in cam_order:
            if cam_name not in camera_images:
                # If camera is missing, use zeros
                processed_cams.append(np.zeros((self.image_height, cam_width, 3), dtype=np.uint8))
                continue

            img = camera_images[cam_name]

            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to target dimensions per camera
            img = cv2.resize(img, (cam_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            processed_cams.append(img)

        # Concatenate all 4 cameras horizontally: 4 * 480 = 1920 width
        rgb = np.concatenate(processed_cams, axis=1)

        # Convert to tensor: (H, W, 3) -> (3, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return rgb_tensor

    def _get_command_onehot(self, command: int) -> np.ndarray:
        """
        Convert driving command to one-hot encoding.

        NavSim commands: [left, straight, right, unknown]
        """
        cmd_onehot = np.zeros(4, dtype=np.float32)
        if command == 0:  # Left
            cmd_onehot[0] = 1.0
        elif command == 1:  # Straight/Forward
            cmd_onehot[1] = 1.0
        elif command == 2:  # Right
            cmd_onehot[2] = 1.0
        else:  # Unknown/LaneFollow -> straight
            cmd_onehot[1] = 1.0
        return cmd_onehot

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for LEAD NavSim model."""
        # 1. Preprocess RGB images
        rgb = self._preprocess_rgb(images).unsqueeze(0).to(self.device)

        # 2. Get speed (m/s)
        speed = np.linalg.norm(ego_state['velocity'][:2])

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if speed < 0.5:
            speed = KICKOFF_SPEED

        # 3. Get acceleration (m/s²)
        if 'acceleration' in ego_state:
            acceleration = np.linalg.norm(ego_state['acceleration'][:2])
        else:
            acceleration = 0.0

        # 4. Get command one-hot
        command = ego_state.get('command', 1)  # Default to straight
        cmd_onehot = self._get_command_onehot(command)

        # 5. Build input tensors
        input_data = {
            'rgb': rgb,
            'speed': torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            'acceleration': torch.tensor([[acceleration]], dtype=torch.float32, device=self.device),
            'command': torch.from_numpy(cmd_onehot).unsqueeze(0).to(self.device),
        }

        return input_data

    def run_inference(self, model_input: Any) -> Any:
        """Run LEAD NavSim model inference."""
        with torch.no_grad():
            prediction = self.model(model_input)
        return prediction

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse LEAD NavSim output."""
        # Extract waypoints from prediction
        if model_output.pred_future_waypoints is not None:
            # Convert to float32 for numpy compatibility (model may output bfloat16)
            waypoints = model_output.pred_future_waypoints[0].float().cpu().numpy()  # (n_waypoints, 2)

            # Swap columns: model outputs [forward, lateral] but evaluator expects [lateral, forward]
            waypoints_swapped = np.column_stack([waypoints[:, 1], waypoints[:, 0]])
        else:
            # Fallback: use empty trajectory
            waypoints_swapped = np.zeros((8, 2), dtype=np.float32)

        return {'trajectory': waypoints_swapped}

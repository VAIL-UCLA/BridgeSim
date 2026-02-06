"""
Model adapter for LEAD (Learning End-to-end Autonomous Driving).
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict

# Add modelzoo navsim agents to path for LEAD
# Note: Requires the 'lead' package to be present in modelzoo/navsim/agents/
modelzoo_agents_path = Path(__file__).resolve().parent.parent.parent / "modelzoo" / "navsim" / "agents"
sys.path.insert(0, str(modelzoo_agents_path))

# Disable beartype for Python 3.8 compatibility
import beartype
beartype.beartype = lambda func: func

# Patch jaxtyping for Python 3.8 compatibility
import jaxtyping as jt
class MockArrayType:
    def __class_getitem__(cls, params):
        return object

jt.Float = MockArrayType
jt.Int = MockArrayType
jt.Bool = MockArrayType
jt.UInt = MockArrayType
jt.Complex = MockArrayType

from lead.tfv6.tfv6 import TFv6
from lead.training.config_training import TrainingConfig

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


class LEADAdapter(BaseModelAdapter):
    """
    Adapter for LEAD model.

    LEAD uses 6 cameras (full panoramic view) and processes images with LTF (Latent TransFuser).
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize LEAD adapter.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config_training = None

    def load_model(self):
        """Load LEAD model from checkpoint."""
        print("Loading LEAD model...")

        # Create config for LEAD
        config_dict = {
            'carla_root': 'data/carla_leaderboard2',
            'carla_leaderboard_mode': True,
            'use_planning_decoder': True,
            'predict_target_speed': True,
            'predict_temporal_spatial_waypoints': True,
            'predict_spatial_path': True,
            'detect_boxes': True,
            'use_bev_semantic': True,
            'use_semantic': True,
            'use_depth': True,
            'use_lidar_data': True,
            'use_radar_detection': False,
            'LTF': True,  # Latent TransFuser mode (2-channel LiDAR)
            'num_used_cameras': 6,
            'used_cameras': [True, True, True, True, True, True],
            'carla_fps': 20.0,
            'data_save_freq': 10.0,
            'target_speed_classes': 41,
            'iou_treshold_nms': 0.2,
            # Image configuration
            'final_image_width': 2304,
            'final_image_height': 384,
            'image_resolution': 1.0,
            # LiDAR BEV configuration
            'pixels_per_meter': 4.0,
            'lidar_resolution_width': 384,
            'lidar_resolution_height': 320,
            # Architecture
            'image_architecture': 'resnet34',
            'lidar_architecture': 'resnet34',
            # Transformer settings
            'block_exp': 4,
            'n_layer': 2,
            'n_head': 4,
            'embd_pdrop': 0.1,
            'resid_pdrop': 0.1,
            'attn_pdrop': 0.1,
        }

        self.config_training = TrainingConfig(config_dict)

        # Force float32 mode
        type(self.config_training).torch_float_type = property(lambda self: torch.float32)

        # Initialize LEAD model
        self.model = TFv6(torch.device(self.device), self.config_training)

        # Load checkpoint with manual handling of mismatched keys
        checkpoint_state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

        # Filter out keys that have size mismatches
        model_state = self.model.state_dict()
        filtered_checkpoint = OrderedDict()
        mismatched_keys = []

        for key, value in checkpoint_state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_checkpoint[key] = value
                else:
                    mismatched_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                filtered_checkpoint[key] = value

        if mismatched_keys:
            print(f"Warning: Skipping {len(mismatched_keys)} mismatched keys (will use random init):")
            for msg in mismatched_keys[:5]:
                print(f"  {msg}")
            if len(mismatched_keys) > 5:
                print(f"  ... and {len(mismatched_keys) - 5} more")

        result = self.model.load_state_dict(filtered_checkpoint, strict=False)
        if result.missing_keys:
            print(f"Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"Unexpected keys: {len(result.unexpected_keys)}")

        self.model.cuda(device=self.device).eval()
        self.model = self.model.float()

        print("LEAD model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """LEAD uses all 6 cameras."""
        return {
            'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
            'CAM_BACK_LEFT': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK_RIGHT': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
        }

    def _preprocess_rgb(self, camera_images: Dict[str, np.ndarray]) -> np.ndarray:
        """Preprocess multi-camera RGB images for LEAD model (6 cameras)."""
        cam_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        processed_cams = []

        for cam_name in cam_order:
            if cam_name not in camera_images:
                # If camera is missing, use zeros
                processed_cams.append(np.zeros((900, 384, 3), dtype=np.uint8))
                continue

            img = camera_images[cam_name]

            # Apply JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Crop to 384 width per camera
            if 'FRONT' in cam_name and 'LEFT' not in cam_name and 'RIGHT' not in cam_name:
                img = img[:, 200:1400, :]  # Front center
            elif 'LEFT' in cam_name:
                img = img[:, :1400, :]  # Left cameras
            elif 'RIGHT' in cam_name:
                img = img[:, 200:, :]  # Right cameras
            else:
                img = img[:, 200:1400, :]  # Back cameras

            # Resize to 384 per camera
            img = cv2.resize(img, (384, 900), interpolation=cv2.INTER_LINEAR)
            processed_cams.append(img)

        # Concatenate all 6 cameras: 6 * 384 = 2304 width
        rgb = np.concatenate(processed_cams, axis=1)

        # Resize to model input (2304x384)
        rgb = cv2.resize(rgb, (2304, 384), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0

        return rgb

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for LEAD model."""
        # 1. Preprocess RGB
        rgb = self._preprocess_rgb(images)

        # 2. Get waypoint in ego frame
        waypoint_world = ego_state['waypoint']
        ego_pos = ego_state['position'][:2]
        ego_heading = ego_state['heading']

        # Transform to ego frame
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        delta = waypoint_world - ego_pos
        target_point = np.array([
            sin_h * delta[0] + cos_h * delta[1],
            cos_h * delta[0] - sin_h * delta[1]
        ])

        # Use same target for previous and next (simplified)
        target_point_prev = target_point
        target_point_next = target_point

        # 3. Get speed
        speed = np.linalg.norm(ego_state['velocity'])

        # 4. Prepare command one-hot
        command = ego_state['command']
        cmd_one_hot = np.zeros(6, dtype=np.float32)
        if 0 <= command < 6:
            cmd_one_hot[command] = 1

        # 5. Build input tensors
        input_tensors = {
            'rgb': torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32),
            'rasterized_lidar': torch.zeros(1, 1, self.config_training.lidar_height_pixel, self.config_training.lidar_width_pixel, dtype=torch.float32, device=self.device),
            'radar': torch.zeros(1, 300, 5, dtype=torch.float32, device=self.device),
            'target_point': torch.from_numpy(target_point).unsqueeze(0).to(self.device, dtype=torch.float32),
            'target_point_previous': torch.from_numpy(target_point_prev).unsqueeze(0).to(self.device, dtype=torch.float32),
            'target_point_next': torch.from_numpy(target_point_next).unsqueeze(0).to(self.device, dtype=torch.float32),
            'speed': torch.tensor([speed]).unsqueeze(0).to(self.device, dtype=torch.float32),
            'command': torch.from_numpy(cmd_one_hot).unsqueeze(0).to(self.device, dtype=torch.float32),
        }

        return input_tensors

    def run_inference(self, model_input: Any) -> Any:
        """Run LEAD model inference."""
        with torch.no_grad():
            prediction = self.model(model_input)
        return prediction

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse LEAD output."""
        # Extract waypoints from prediction
        if hasattr(model_output, 'pred_future_waypoints') and model_output.pred_future_waypoints is not None:
            # pred_future_waypoints: (B, T, 2)
            waypoints = model_output.pred_future_waypoints[0].cpu().numpy()  # (T, 2)
        else:
            # Fallback: use empty trajectory
            waypoints = np.zeros((4, 2), dtype=np.float32)

        return {'trajectory': waypoints}

    def save_intermediate_outputs(self, model_output: Any, output_path: str):
        """Save intermediate LEAD outputs."""
        output_path = Path(output_path)

        # Save BEV embedding
        if hasattr(self.model, '_last_bev_embed_test') and self.model._last_bev_embed_test is not None:
            torch.save(self.model._last_bev_embed_test.cpu(), output_path / "bev_embed.pth")

        # Save semantic segmentation
        if hasattr(model_output, 'pred_semantic') and model_output.pred_semantic is not None:
            torch.save(model_output.pred_semantic.cpu(), output_path / "seg_output.pth")

        # Save motion prediction
        if hasattr(model_output, 'pred_future_waypoints') and model_output.pred_future_waypoints is not None:
            torch.save(model_output.pred_future_waypoints.cpu(), output_path / "motion_output.pth")

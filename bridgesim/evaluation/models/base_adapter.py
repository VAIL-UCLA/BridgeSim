"""
Base adapter interface for autonomous driving models.
All model adapters must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    Each model adapter handles:
    1. Model loading and initialization
    2. Input preprocessing (camera images, ego state, etc.)
    3. Model inference
    4. Output parsing and trajectory extraction
    """

    def __init__(self, checkpoint_path: str, config_path: str = None, **kwargs):
        """
        Initialize the model adapter.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config (optional, not used by all models)
            **kwargs: Additional model-specific arguments
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None
        self.device = "cuda"

    @abstractmethod
    def load_model(self):
        """
        Load the model from checkpoint.
        Must set self.model to the loaded model.
        """
        pass

    @abstractmethod
    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """
        Prepare model input from raw sensor data and ego state.

        Args:
            images: Dictionary mapping camera names to RGB images (H, W, 3)
            ego_state: Dictionary containing:
                - 'position': (x, y, z) ego position in world frame
                - 'heading': ego heading in radians
                - 'velocity': (vx, vy, vz) velocity vector
                - 'acceleration': (ax, ay, az) acceleration vector
                - 'angular_velocity': (wx, wy, wz) angular velocity
            scenario_data: Full scenario metadata
            frame_id: Current frame index

        Returns:
            Model-specific input format (e.g., batched tensor, dict, etc.)
        """
        pass

    @abstractmethod
    def run_inference(self, model_input: Any) -> Any:
        """
        Run model inference.

        Args:
            model_input: Preprocessed input from prepare_input()

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Parse model output and extract planning trajectory.

        Args:
            model_output: Raw output from run_inference()
            ego_state: Current ego state (needed for coordinate transforms)

        Returns:
            Dictionary containing:
                - 'trajectory': (N, 2) array of (x, y) waypoints in ego frame
                - 'heading' (optional): (N,) array of heading angles
                - Additional model-specific outputs (optional)
        """
        pass

    def get_waypoint_dt(self) -> float:
        """
        Return time interval between predicted waypoints in seconds.

        This is used to interpolate the trajectory to simulation timestep.
        Override in subclasses if the model uses a different interval.

        Returns:
            Time interval between waypoints (default: 0.5s)
        """
        return 0.5  # Default for most models (TransFuser, RAP, DrivoR, etc.)

    def get_trajectory_time_horizon(self) -> float:
        """
        Return the total time horizon of predicted trajectory in seconds.

        This is used to validate that the model predicts enough waypoints
        for the configured replan_rate.

        Returns:
            Time horizon in seconds (default: 4.0s)
        """
        return 4.0  # Default for most models (TransFuser, DiffusionDrive, etc.)

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """
        Return camera configuration for this model.

        Returns:
            Dictionary mapping camera names to config dicts with keys:
            {x, y, z, yaw, pitch, roll, fov, width, height}
        """
        # Default 6-camera setup (used by UniAD/VAD) + third-person view
        return {
            'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
            'CAM_BACK_LEFT': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_BACK_RIGHT': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            # # Third-person view camera: 3m behind ego, elevated, facing forward
            # 'CAM_THIRD_PERSON': {'x': -8.0, 'y': 0.0, 'z': 3.0, 'yaw': 0.0, 'pitch': -15.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
        }

    def save_intermediate_outputs(self, model_output: Any, output_path: str):
        """
        Save intermediate model outputs (optional, for debugging/visualization).

        Args:
            model_output: Raw model output
            output_path: Directory to save outputs
        """
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(checkpoint={self.checkpoint_path})"

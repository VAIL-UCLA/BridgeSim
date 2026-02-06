"""
BridgeSim - Core shared utilities for the BridgeSim autonomous driving evaluation framework.

This package provides consolidated camera parameters and training utilities used across
calibration, evaluation, and data conversion modules.
"""

from .camera_utils import (
    # Bench2Drive 6-camera setup
    BENCH2DRIVE_CAM_NAMES,
    BENCH2DRIVE_LIDAR2IMG,
    BENCH2DRIVE_LIDAR2CAM,
    BENCH2DRIVE_LIDAR2EGO,
    BENCH2DRIVE_CAM_DIR_TO_NAME,
    BENCH2DRIVE_CAM_NAME_TO_DIR,
    # OpenScene 9-camera setup
    OPENSCENE_CAMERA_PARAMS,
    # Utility functions
    convert_camera_params_to_simple_format,
    rotation_matrix_to_euler_angles,
    calculate_fov_from_intrinsics,
)

from .training_utils import (
    set_seed,
    BEV_MEAN,
    BEV_STD,
)

__version__ = "0.1.0"

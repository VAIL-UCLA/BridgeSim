"""
Camera parameters and utilities for BridgeSim.

This module consolidates camera calibration parameters from:
- Bench2Drive 6-camera setup (from calibration/train_uniad_bev.py)
- OpenScene 9-camera setup (from converters/bench2drive/renderer.py at project root)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# =============================================================================
# Bench2Drive 6-Camera Setup
# =============================================================================

BENCH2DRIVE_CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

BENCH2DRIVE_LIDAR2IMG = {
    "CAM_FRONT": np.array(
        [
            [1.14251841e03, 8.00000000e02, 0.00000000e00, -9.52000000e02],
            [0.00000000e00, 4.50000000e02, -1.14251841e03, -8.09704417e02],
            [0.00000000e00, 1.00000000e00, 0.00000000e00, -1.19000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_FRONT_LEFT": np.array(
        [
            [6.03961325e-14, 1.39475744e03, 0.00000000e00, -9.20539908e02],
            [-3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
            [-8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_FRONT_RIGHT": np.array(
        [
            [1.31064327e03, -4.77035138e02, 0.00000000e00, -4.06010608e02],
            [3.68618420e02, 2.58109396e02, -1.14251841e03, -6.47296750e02],
            [8.19152044e-01, 5.73576436e-01, 0.00000000e00, -8.29094072e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK": np.array(
        [
            [-5.60166031e02, -8.00000000e02, 0.00000000e00, -1.28800000e03],
            [5.51091060e-14, -4.50000000e02, -5.60166031e02, -8.58939847e02],
            [1.22464680e-16, -1.00000000e00, 0.00000000e00, -1.61000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK_LEFT": np.array(
        [
            [-1.14251841e03, 8.00000000e02, 0.00000000e00, -6.84385123e02],
            [-4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
            [-9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    "CAM_BACK_RIGHT": np.array(
        [
            [3.60989788e02, -1.34723223e03, 0.00000000e00, -1.04238127e02],
            [4.22861679e02, -1.53909064e02, -1.14251841e03, -4.96004706e02],
            [9.39692621e-01, -3.42020143e-01, 0.00000000e00, -4.92889531e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
}

BENCH2DRIVE_LIDAR2CAM = {
    "CAM_FRONT": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, 1.0, 0.0, -1.19], [0.0, 0.0, 0.0, 1.0]]
    ),
    "CAM_FRONT_LEFT": np.array(
        [
            [0.57357644, 0.81915204, 0.0, -0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [-0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_FRONT_RIGHT": np.array(
        [
            [0.57357644, -0.81915204, 0.0, 0.22517331],
            [0.0, 0.0, -1.0, -0.24],
            [0.81915204, 0.57357644, 0.0, -0.82909407],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_BACK": np.array(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.24], [0.0, -1.0, 0.0, -1.61], [0.0, 0.0, 0.0, 1.0]]
    ),
    "CAM_BACK_LEFT": np.array(
        [
            [-0.34202014, 0.93969262, 0.0, -0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [-0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CAM_BACK_RIGHT": np.array(
        [
            [-0.34202014, -0.93969262, 0.0, 0.25388956],
            [0.0, 0.0, -1.0, -0.24],
            [0.93969262, -0.34202014, 0.0, -0.49288953],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

BENCH2DRIVE_LIDAR2EGO = np.array(
    [[0.0, 1.0, 0.0, -0.39], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.84], [0.0, 0.0, 0.0, 1.0]]
)

# Mapping between camera directory names and canonical names
BENCH2DRIVE_CAM_DIR_TO_NAME = {
    "rgb_front": "CAM_FRONT",
    "rgb_front_left": "CAM_FRONT_LEFT",
    "rgb_front_right": "CAM_FRONT_RIGHT",
    "rgb_back": "CAM_BACK",
    "rgb_back_left": "CAM_BACK_LEFT",
    "rgb_back_right": "CAM_BACK_RIGHT",
}
BENCH2DRIVE_CAM_NAME_TO_DIR = {v: k for k, v in BENCH2DRIVE_CAM_DIR_TO_NAME.items()}

# =============================================================================
# Utility Functions
# =============================================================================


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Assumes the rotation order is Z-Y-X (yaw-pitch-roll).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    # Convert to degrees
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def convert_camera_params_to_simple_format(
    camera_params: Dict,
    image_width: int = 1920,
    image_height: int = 1120,
    to_metadrive: bool = False,
) -> Dict:
    """
    Convert camera_params dictionary to MetaDrive format with x, y, z, yaw, pitch, roll, fov.

    Args:
        camera_params: Dictionary containing camera calibration parameters
        image_width: Image width in pixels (default: 1920)
        image_height: Image height in pixels (default: 1120)
        to_metadrive: If True, converts from RAP/nuScenes to MetaDrive coordinate system

    Returns:
        Dictionary in the format:
        {
            'CAM_F0': {'x': ..., 'y': ..., 'z': ..., 'yaw': ..., 'pitch': ...,
                       'roll': ..., 'fov': ..., 'width': ..., 'height': ...},
            ...
        }

    Coordinate System Notes:
        RAP/nuScenes: X=Forward, Y=Left, Z=Up, Yaw around Z (0 deg = +X)
        MetaDrive: Y=Forward, X=Right, Z=Up, Heading around Z (0 deg = +Y)
    """
    simple_format = {}

    for cam_name, params in camera_params.items():
        # Extract position from translation vector
        translation = params["sensor2lidar_translation"]

        # Extract orientation from rotation matrix
        rotation = params["sensor2lidar_rotation"]
        roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation)

        # Calculate FOV from intrinsics
        intrinsics = params["intrinsics"]
        fov_h, fov_v = calculate_fov_from_intrinsics(intrinsics, image_width, image_height)

        if to_metadrive:
            # Transform from RAP (X=Fwd, Y=Left) to MetaDrive (Y=Fwd, X=Right)
            # Position: RAP(x,y,z) -> MetaDrive(x,y,z)
            #   RAP_X (forward) -> MetaDrive_Y (forward)
            #   RAP_Y (left) -> MetaDrive_-X (left = -right)
            #   RAP_Z (up) -> MetaDrive_Z (up)
            x_md = -translation[1]  # RAP Y (left) -> MetaDrive -X (right inverted)
            y_md = translation[0]  # RAP X (forward) -> MetaDrive Y (forward)
            z_md = translation[2]  # RAP Z (up) -> MetaDrive Z (up)

            # Orientation: Add 90 deg to account for coordinate system rotation
            # RAP yaw 0 deg points to +X (forward in RAP), MetaDrive heading 0 deg points to +Y (forward in MD)
            # Since we're also flipping the Y-axis (left->right), we need +90 deg not -90 deg
            # MetaDrive_heading = RAP_yaw + 90 deg
            yaw_md = yaw + 90.0

            # Pitch and roll need adjustment due to axis swap
            # This is more complex - for cameras typically mounted upright,
            # we can often zero out roll and keep pitch
            pitch_md = pitch
            roll_md = 0.0  # Reset roll as it's being used incorrectly

            simple_format[cam_name] = {
                "x": float(x_md),
                "y": float(y_md),
                "z": float(z_md),
                "yaw": float(yaw_md),
                "pitch": float(pitch_md),
                "roll": float(roll_md),
                "fov": float(fov_h),
                "fov_h": float(fov_h),
                "fov_v": float(fov_v),
                "width": image_width,
                "height": image_height,
            }
        else:
            # Keep original RAP/nuScenes coordinates
            simple_format[cam_name] = {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "fov": float(fov_h),
                "fov_h": float(fov_h),
                "fov_v": float(fov_v),
                "width": image_width,
                "height": image_height,
            }

    return simple_format

def calculate_fov_from_intrinsics(
    K: np.ndarray, image_width: int, image_height: int
) -> Tuple[float, float]:
    """
    Calculate horizontal and vertical FOV from intrinsics matrix.

    Args:
        K: 3x3 intrinsics matrix
        image_width: image width in pixels
        image_height: image height in pixels

    Returns:
        Tuple of (fov_horizontal, fov_vertical) in degrees
    """
    fx = K[0, 0]
    fy = K[1, 1]

    # FOV = 2 * arctan(image_dimension / (2 * focal_length))
    fov_horizontal = 2 * np.arctan(image_width / (2 * fx))
    fov_vertical = 2 * np.arctan(image_height / (2 * fy))

    # Convert to degrees
    return np.degrees(fov_horizontal), np.degrees(fov_vertical)


#NAVSIM camera Param 

NAVSIM_CAMERA_PARAMS = {
    'CAM_F0': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[-0.00785972, -0.02271912, 0.99971099],
                                                  [-0.99994262, 0.00745516, -0.00769211],
                                                  [-0.00727825, -0.99971409, -0.02277642]]),
               'sensor2lidar_translation': np.array([1.65506747, -0.01168732, 1.49112208])},
    'CAM_L0': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[0.81776776, -0.0057693, 0.57551942],
                                                  [-0.57553938, -0.01377628, 0.81765802],
                                                  [0.0032112, -0.99988846, -0.01458626]]),
               'sensor2lidar_translation': np.array([1.63069485, 0.11956747, 1.48117884])},
    'CAM_L1': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[0.93120104, 0.00261563, -0.36449662],
                                                  [0.36447127, -0.02048653, 0.93098926],
                                                  [-0.00503215, -0.99978671, -0.0200304]]),
               'sensor2lidar_translation': np.array([1.29939471, 0.63819702, 1.36736822])},
    'CAM_L2': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[0.63520782, 0.01497516, -0.77219607],
                                                  [0.77232489, -0.00580669, 0.63520119],
                                                  [0.00502834, -0.99987101, -0.01525415]]),
               'sensor2lidar_translation': np.array([-0.49561003, 0.54750373, 1.3472672])},
    'CAM_R0': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[-0.82454901, 0.01165722, 0.56567043],
                                                  [-0.56528395, 0.02532491, -0.82450755],
                                                  [-0.02393702, -0.9996113, -0.01429199]]),
               'sensor2lidar_translation': np.array([1.61828343, -0.15532203, 1.49007665])},
    'CAM_R1': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[-0.92684778, 0.02177016, -0.37480562],
                                                  [0.37497631, 0.00421964, -0.92702479],
                                                  [-0.01859993, -0.9997541, -0.01207426]]),
               'sensor2lidar_translation': np.array([1.27299407, -0.60973112, 1.37217911])},
    'CAM_R2': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[-0.62253245, 0.03706878, -0.78171558],
                                                  [0.78163434, -0.02000083, -0.62341618],
                                                  [-0.03874424, -0.99911254, -0.01652307]]),
               'sensor2lidar_translation': np.array([-0.48771615, -0.493167, 1.35027683])},
    'CAM_B0': {'distortion': np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
               'intrinsics': np.array([[1.545e+03, 0.000e+00, 9.600e+02],
                                       [0.000e+00, 1.545e+03, 5.600e+02],
                                       [0.000e+00, 0.000e+00, 1.000e+00]]),
               'sensor2lidar_rotation': np.array([[0.00802542, 0.01047463, -0.99991293],
                                                  [0.99989075, -0.01249671, 0.00789433],
                                                  [-0.01241293, -0.99986705, -0.01057378]]),
               'sensor2lidar_translation': np.array([-0.47463312, 0.02368552, 1.4341838])}
}

NAVSIM_ALL_CAM_IDS = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']

NAVSIM_CAM_CONFIGS = {
    'CAM_F0': {'x':  1.65506747, 'y': -0.01168732, 'z': 1.49112208, 'yaw':  -0.45034535, 'pitch':  0.41701669, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_L0': {'x':  1.63069485, 'y':  0.11956747, 'z': 1.48117884, 'yaw':  54.86237801, 'pitch': -0.18398852, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_L1': {'x':  1.29939471, 'y':  0.63819702, 'z': 1.36736822, 'yaw': 111.37533125, 'pitch':  0.28832217, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_L2': {'x': -0.49561003, 'y':  0.54750373, 'z': 1.34726720, 'yaw': 140.56400434, 'pitch': -0.28810387, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_R0': {'x':  1.61828343, 'y': -0.15532203, 'z': 1.49007665, 'yaw': -55.56673404, 'pitch':  1.37162123, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_R1': {'x':  1.27299407, 'y': -0.60973112, 'z': 1.37217911, 'yaw': 247.97314134, 'pitch':  1.06575895, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_R2': {'x': -0.48771615, 'y': -0.49316700, 'z': 1.35027683, 'yaw': 218.53555903, 'pitch':  2.22043718, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_B0': {'x': -0.47463312, 'y':  0.02368552, 'z': 1.43418380, 'yaw': 179.54013694, 'pitch':  0.71122677, 'roll': 0.0, 'fov': 70.0, 'fov_h': 63.71026476, 'fov_v': 39.84706408, 'width': 1920, 'height': 1120},
    'CAM_THIRD_PERSON': {'x': -9.0, 'y': 0.0, 'z': 6.0, 'yaw': 0.0, 'pitch': -20.0, 'roll': 0.0, 'fov': 50, 'width': 900, 'height': 900},
}


# =============================================================================
# OpenScene 9-Camera Setup
# =============================================================================

OPENSCENE_CAMERA_PARAMS = {
    "CAM_F0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [-0.00785972, -0.02271912, 0.99971099],
                [-0.99994262, 0.00745516, -0.00769211],
                [-0.00727825, -0.99971409, -0.02277642],
            ]
        ),
        "sensor2lidar_translation": np.array([1.65506747, -0.01168732, 1.49112208]),
    },
    "CAM_L0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [0.81776776, -0.0057693, 0.57551942],
                [-0.57553938, -0.01377628, 0.81765802],
                [0.0032112, -0.99988846, -0.01458626],
            ]
        ),
        "sensor2lidar_translation": np.array([1.63069485, 0.11956747, 1.48117884]),
    },
    "CAM_L1": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [0.93120104, 0.00261563, -0.36449662],
                [0.36447127, -0.02048653, 0.93098926],
                [-0.00503215, -0.99978671, -0.0200304],
            ]
        ),
        "sensor2lidar_translation": np.array([1.29939471, 0.63819702, 1.36736822]),
    },
    "CAM_L2": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [0.63520782, 0.01497516, -0.77219607],
                [0.77232489, -0.00580669, 0.63520119],
                [0.00502834, -0.99987101, -0.01525415],
            ]
        ),
        "sensor2lidar_translation": np.array([-0.49561003, 0.54750373, 1.3472672]),
    },
    "CAM_R0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [-0.82454901, 0.01165722, 0.56567043],
                [-0.56528395, 0.02532491, -0.82450755],
                [-0.02393702, -0.9996113, -0.01429199],
            ]
        ),
        "sensor2lidar_translation": np.array([1.61828343, -0.15532203, 1.49007665]),
    },
    "CAM_R1": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [-0.92684778, 0.02177016, -0.37480562],
                [0.37497631, 0.00421964, -0.92702479],
                [-0.01859993, -0.9997541, -0.01207426],
            ]
        ),
        "sensor2lidar_translation": np.array([1.27299407, -0.60973112, 1.37217911]),
    },
    "CAM_R2": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [-0.62253245, 0.03706878, -0.78171558],
                [0.78163434, -0.02000083, -0.62341618],
                [-0.03874424, -0.99911254, -0.01652307],
            ]
        ),
        "sensor2lidar_translation": np.array([-0.48771615, -0.493167, 1.35027683]),
    },
    "CAM_B0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array(
            [
                [1.545e03, 0.000e00, 9.600e02],
                [0.000e00, 1.545e03, 5.600e02],
                [0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "sensor2lidar_rotation": np.array(
            [
                [0.00802542, 0.01047463, -0.99991293],
                [0.99989075, -0.01249671, 0.00789433],
                [-0.01241293, -0.99986705, -0.01057378],
            ]
        ),
        "sensor2lidar_translation": np.array([-0.47463312, 0.02368552, 1.4341838]),
    },
}
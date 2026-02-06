"""
Replay a single NavSim log (converted to MetaDrive) and capture camera images.
Based on generate_images() from convert_openscene_with_filter.py

Usage:
    python tools/replay_navsim_log.py \
        --scenario-path <metadrive_pkl> \
        --output-root <output_dir> \
        [--max-frames N] \
        [--cameras CAM_F0,CAM_L0]
"""

import argparse
import os
import pickle
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Camera Parameters (from convert_openscene_with_filter.py:178-243) ---
CAMERA_PARAMS = {
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

ALL_CAM_IDS = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']


def rotation_matrix_to_euler_angles(R_mat):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    sy = np.sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    else:
        roll = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        yaw = 0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def calculate_fov_from_intrinsics(K, image_width, image_height):
    """Calculate horizontal and vertical FOV from camera intrinsics."""
    fx = K[0, 0]
    fy = K[1, 1]
    fov_horizontal = 2 * np.arctan(image_width / (2 * fx))
    fov_vertical = 2 * np.arctan(image_height / (2 * fy))
    return np.degrees(fov_horizontal), np.degrees(fov_vertical)


def get_metadrive_cam_params(cam_params, image_width=1920, image_height=1120):
    """Convert NavSim camera params to MetaDrive pose format."""
    translation = cam_params['sensor2lidar_translation']
    rotation = cam_params['sensor2lidar_rotation']
    intrinsics = cam_params['intrinsics']

    x_md = -translation[1]
    y_md = translation[0]
    z_md = translation[2]

    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation)
    yaw_md = yaw + 90.0
    pitch_md = pitch
    roll_md = 0.0

    fov_h, fov_v = calculate_fov_from_intrinsics(intrinsics, image_width, image_height)

    return {
        'pos': (x_md, y_md, z_md),
        'hpr': (yaw_md, pitch_md, roll_md),
        'fov': fov_h,
        'width': image_width,
        'height': image_height
    }


def replay_and_capture(scenario_path, output_root, max_frames=None, cameras=None):
    """
    Replay scenario and capture images.

    Args:
        scenario_path: Path to converted MetaDrive .pkl file
        output_root: Output directory root
        max_frames: Optional limit on frames to process
        cameras: List of camera IDs to capture (default: all 8)
    """
    # Use all cameras if none specified
    cam_ids = cameras if cameras else ALL_CAM_IDS

    logger.info(f"Loading scenario: {scenario_path}")

    # Load scenario and extract frame_info from metadata
    with open(scenario_path, 'rb') as f:
        scenario = pickle.load(f)

    metadata = scenario.get("metadata", {})
    frame_info = metadata.get("frame_info", [])
    log_name = metadata.get("log_name", "unknown_log")
    scenario_length = scenario.get("length", len(frame_info))

    if not frame_info:
        logger.error("No frame_info in scenario metadata. Re-convert with updated converter.")
        return

    output_root = Path(output_root)

    # Pre-calculate MetaDrive configs for selected cameras
    md_cam_configs = {}
    for cam_id in cam_ids:
        if cam_id in CAMERA_PARAMS:
            md_cam_configs[cam_id] = get_metadrive_cam_params(CAMERA_PARAMS[cam_id], 1920, 1120)
        else:
            logger.warning(f"Unknown camera: {cam_id}, skipping")

    logger.info(f"Capturing cameras: {list(md_cam_configs.keys())}")

    sensors = {"rgb_camera": (RGBCamera, 1920, 1120)}
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(scenario_path)))

    # Calculate render horizon
    render_horizon = min(scenario_length, max_frames) if max_frames else scenario_length

    env_config = {
        "use_render": False,
        "image_on_cuda": True,
        "agent_policy": ReplayEgoCarPolicy,
        "num_scenarios": 1,
        "horizon": render_horizon,
        "sensors": sensors,
        "data_directory": data_dir,
        "image_observation": True,
        "show_interface": False,
        "show_logo": False,
        "vehicle_config": {
            "image_source": "rgb_camera",
            "show_navi_mark": False,
        }
    }

    env = ScenarioEnv(env_config)

    try:
        env.reset(seed=0)
        ego = env.agent

        logger.info(f"Rendering {render_horizon} frames...")

        for current_step in tqdm(range(render_horizon), desc="Rendering Frames"):
            env.step([0, 0])

            # Get frame data
            frame_data = frame_info[current_step] if current_step < len(frame_info) else {}
            frame_token = frame_data.get("token", f"{current_step:05d}")

            # Get sensor
            sensor = env.engine.get_sensor("rgb_camera")

            # Capture each selected camera
            for cam_id in md_cam_configs:
                cfg = md_cam_configs[cam_id]

                if hasattr(sensor.lens, 'setFov'):
                    sensor.lens.setFov(70)

                img = sensor.perceive(
                    to_float=False,
                    new_parent_node=ego.origin,
                    position=cfg['pos'],
                    hpr=cfg['hpr']
                ).get()

                # Get image token
                image_token = None
                cams = frame_data.get("cams", {})
                if cam_id in cams:
                    image_token = cams[cam_id].get("sample_data_token")
                if not image_token:
                    image_token = frame_token

                # Save to NavSim folder structure
                save_dir = output_root / log_name / cam_id
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{image_token}.jpg"
                cv2.imwrite(str(save_path), img)

        logger.info(f"Done! Images saved to: {output_root / log_name}")

    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replay NavSim log in MetaDrive and capture camera images"
    )
    parser.add_argument("--scenario-path", required=True,
                        help="Path to converted MetaDrive .pkl file")
    parser.add_argument("--output-root", required=True,
                        help="Output directory root")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to render (default: all)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated camera IDs to capture (default: all 8). "
                             "Options: CAM_F0,CAM_L0,CAM_L1,CAM_L2,CAM_R0,CAM_R1,CAM_R2,CAM_B0")

    args = parser.parse_args()

    # Parse camera list
    cameras = None
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",")]

    replay_and_capture(
        scenario_path=args.scenario_path,
        output_root=args.output_root,
        max_frames=args.max_frames,
        cameras=cameras
    )


if __name__ == "__main__":
    main()

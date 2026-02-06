import urllib.request
import shutil
import argparse
import logging
import os
import time

import logging

import pkg_resources  # for suppress warning
import argparse
import os
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.utils import get_number_of_scenarios

# --- Added Imports from convert_bench2drive.py and renderer.py ---
import sys
import cv2
import numpy as np
import math
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.type import MetaDriveType

# Import renderer from same directory
from renderer_md import ScenarioRenderer, COLOR_TABLE
# -----------------------------------------------------------------


# Define fixed relative poses for capturing RGB images
# (pos, hpr) -> (x, y, z) and (heading, pitch, roll) relative to ego vehicle
STATIC_SENSOR_CONFIGS = {
    'rgb_front': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'rgb_front_left': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'rgb_front_right': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'rgb_back': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
    'rgb_back_left': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'rgb_back_right': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
}

# Define input directory
DATABASE_PATH = os.path.abspath("/workspace/converted_nuscenes/_3")

# Define output directory for all rendered images
IMAGE_OUTPUT_DIR = Path("/tmp/converted_nuscenes/scene-0655")

def _create_sensor_directories():
    """Create directory structure for all sensor data."""
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Directories for MetaDrive RGB sensors
    for sensor_name in STATIC_SENSOR_CONFIGS.keys():
        sensor_path = IMAGE_OUTPUT_DIR / 'camera' / sensor_name.upper()
        sensor_path.mkdir(parents=True, exist_ok=True)
    
    # Directories for 3D-Rasterised images
    for sensor_name in ['raster_front', 'raster_front_left', 'raster_front_right', 
                        'raster_back', 'raster_back_left', 'raster_back_right']:
        sensor_path = IMAGE_OUTPUT_DIR / 'camera' / sensor_name.upper()
        sensor_path.mkdir(parents=True, exist_ok=True)
        
    logging.info(f"Created sensor directories in {IMAGE_OUTPUT_DIR}")


def _build_raster_scenario_dict(env: ScenarioEnv, frame_idx: int) -> Dict[str, Any]:
    """
    Build the 'scenario' dictionary required by the ScenarioRenderer
    by querying the current state of the MetaDrive environment.
    (Copied from convert_bench2drive.py)
    """
    if not env or not env.agent:
        return {}

    ego_vehicle = env.agent
    scenario = {
        'ego_pos_3d': np.array(ego_vehicle.get_state()["position"]),
        'ego_heading': ego_vehicle.heading_theta,
        'map_features': env.engine.data_manager.current_scenario["map_features"],
        'traffic_lights': []
    }

    # --- 1. Get Traffic Light States ---
    dynamic_map = env.engine.data_manager.current_scenario.get("dynamic_map_states", {})
    for lane_id, tl_data in dynamic_map.items():
        if "state" in tl_data and frame_idx < len(tl_data["state"]["object_state"]):
            state_str = tl_data["state"]["object_state"][frame_idx]
            is_red = (state_str == "LANE_STATE_STOP")
            # Use the stop_point as the [x, y] position
            pos_xy = tl_data["stop_point"][:2]
            scenario['traffic_lights'].append((lane_id, is_red, pos_xy))

    # --- 2. Get All Other Dynamic Objects (Vehicles, Pedestrians) ---
    gt_boxes_world = []
    gt_names = []

    # Get all objects from the MetaDrive engine
    objects = env.engine.get_objects(filter=lambda o: o.name != ego_vehicle.name)
    
    for obj in objects.values():
        if obj.metadrive_type == MetaDriveType.TRAFFIC_LIGHT:
            continue  # Skip traffic lights, already processed

        obj_state = obj.get_state()
        #if obj.metadrive_type == MetaDriveType.TRAFFIC_BARRIER or obj.metadrive_type == MetaDriveType.TRAFFIC_CONE:
        #print("OBJ STATE:", obj.metadrive_type, obj.LENGTH, obj.WIDTH, obj.HEIGHT)

        # Format: [x, y, z, L, W, H, yaw]
        box_data = [
            obj_state["position"][0],
            obj_state["position"][1],
            obj_state["position"][2],
            obj.LENGTH,
            obj.WIDTH,
            obj.HEIGHT,
            obj.heading_theta
        ]
        gt_boxes_world.append(box_data)
        
        # Get object type
        if obj.metadrive_type == MetaDriveType.VEHICLE:
            gt_names.append("vehicle")
        elif obj.metadrive_type == MetaDriveType.PEDESTRIAN:
            gt_names.append("pedestrian")
        elif obj.metadrive_type == MetaDriveType.CYCLIST:
            gt_names.append("bicycle")
        else:
            gt_names.append("vehicle") # Default for other types

    if gt_boxes_world:
        scenario['anns'] = {
            'gt_boxes_world': np.array(gt_boxes_world, dtype=np.float32),
            'gt_names': np.array(gt_names)
        }
    else:
        # Add empty structure if no other objects are present
        scenario['anns'] = {
            'gt_boxes_world': np.empty((0, 7), dtype=np.float32),
            'gt_names': np.empty((0,), dtype=str)
        }

    return scenario


def test_nuscenes_and_sim():
    # Create output directories for images
    _create_sensor_directories()
    
    # Initialize the ScenarioRenderer
    renderer = ScenarioRenderer()

    env = ScenarioEnv(
        {
            "use_render": False,        # Off-screen
            "image_on_cuda": True,      # Use GPU for rendering (as requested)
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": False,    # No window
            "log_level": logging.INFO,
            "num_scenarios": 3,
            "horizon": 1000,
            
            # Add a single RGB camera that we will move around
            "image_observation": True,  # Required for sensors
            "sensors": {
                "rgb_camera": (RGBCamera, 1600, 900), # 1600x900 for speed
            },

            "vehicle_config": dict(
                no_wheel_friction=True,
            ),
            "data_directory": DATABASE_PATH,
        }
    )

    # Get the single RGB sensor from the engine
    rgb_sensor = None

    try:
        for index in range(2, 3):
            logging.info(f"--- Running Scenario {index} ---")
            env.reset(seed=index)
            
            # Get sensor after reset
            if not rgb_sensor:
                rgb_sensor = env.engine.get_sensor("rgb_camera")

            scenario_length = env.engine.data_manager.current_scenario_length

            for t in tqdm(range(scenario_length), desc="Rendering frames"):

                # Step the environment
                obs, _, _, _, _ = env.step([0, 0])
                
                frame_idx = env.episode_step
                frame_str = f"{frame_idx:05d}.jpg"
                ego_vehicle = env.agent

                # --- A) Generate and Save RGB Sensor Images ---
                try:
                    for cam_name, config in STATIC_SENSOR_CONFIGS.items():
                        img_data = rgb_sensor.perceive(
                            to_float=False,
                            new_parent_node=env.agent.origin,
                            position=(config['y'], config['x'], config['z']),
                            hpr=(-config['yaw'], config['pitch'], -config['roll'])
                        ).get()

                        output_path = IMAGE_OUTPUT_DIR / 'camera' / cam_name.upper() / frame_str
                        cv2.imwrite(str(output_path), img_data)
                        
                except Exception as e:
                    logging.error(f"Failed to generate RGB sensor image for frame {frame_idx}: {e}")
                    traceback.print_exc()

                # --- B) Generate and Save 3D-Rasterised Images ---
                try:
                    # 1. Build the scenario dict from the current env state
                    raster_scenario_dict = _build_raster_scenario_dict(env, frame_idx)
                    
                    # 2. Render the hardcoded nuPlan camera views
                    rendered_cameras = renderer.observe(raster_scenario_dict)
                    
                    # 3. Save the images
                    for cam_id, image in rendered_cameras.items():
                        # Map cam_id (e.g., 'CAM_F0') to dir name (e.g., 'raster_f0')
                        folder_suffix = cam_id.replace('CAM_', '').lower()
                        raster_dir_name = f"raster_{folder_suffix}"
                        output_path = IMAGE_OUTPUT_DIR / 'camera' / raster_dir_name.upper() / frame_str
                        # save as BGR
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_path), image)
                        
                except Exception as e:
                    logging.error(f"Failed to generate raster image for frame {frame_idx}: {e}")
                    traceback.print_exc()
                
                # Check if scenario is done
                if env.episode_step >= env.engine.data_manager.current_scenario_length:
                    logging.info(f"Scenario {index} complete.")
                    break
    
    except Exception as e:
        logging.error(f"An error occurred during simulation: {e}")
        traceback.print_exc()
        
    finally:
        env.close()
        logging.info("Simulation finished and environment closed.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    paths = [os.path.abspath("/workspace/converted_nuscenes/_9")]


    # Define output directory for all rendered images
    dirs = [Path("/tmp/converted_nuscenes/scene-1100")]
    
    for db_path, img_path in zip(paths, dirs):
        DATABASE_PATH = db_path
        IMAGE_OUTPUT_DIR = img_path
        test_nuscenes_and_sim()

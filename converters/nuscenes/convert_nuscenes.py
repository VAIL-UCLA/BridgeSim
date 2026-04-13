desc = "Build database from nuScenes scenarios"

prediction_split = ["mini_train", "mini_val", "train", "train_val", "val"]
scene_split = ["v1.0-mini", "v1.0-trainval", "v1.0-test"]

import argparse
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from converters.nuscenes.utils import (
    convert_nuscenes_scenario,
    get_nuscenes_scenarios,
    get_nuscenes_prediction_split,
)
from converters.common.utils import write_to_directory

try:
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.component.sensors.rgb_camera import RGBCamera
except ImportError:
    pass

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
CAM_CONFIGS = {
    'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_BACK': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
    'CAM_BACK_LEFT': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_BACK_RIGHT': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
}


def generate_images(dataset_path, num_scenarios_to_render=None):
    print("Starting Image Generation...")

    scenario_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".pkl") and file.startswith("sd_nuscenes"):
                scenario_files.append(os.path.join(root, file))

    scenario_files.sort()

    if not scenario_files:
        print("No converted scenario files found to render.")
        return

    if num_scenarios_to_render:
        scenario_files = scenario_files[:num_scenarios_to_render]

    print(f"Found {len(scenario_files)} scenarios to render.")

    env_config = {
        "use_render": False,
        "image_on_cuda": True,
        "agent_policy": ReplayEgoCarPolicy,
        "num_scenarios": len(scenario_files),
        "sensors": {"rgb_camera": (RGBCamera, 1600, 900)},
        "data_directory": dataset_path,
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
        for idx, scenario_file in enumerate(tqdm(scenario_files, desc="Rendering Scenarios")):
            env.config["start_scenario_index"] = idx

            try:
                env.reset(seed=idx)
            except Exception as e:
                print(f"Error resetting env for index {idx}: {e}")
                continue

            current_id = env.engine.data_manager.current_scenario_summary["id"]

            with open(scenario_file, "rb") as f:
                pickle_data = pickle.load(f)
            target_id = pickle_data["id"]

            if current_id != target_id:
                print(f"CRITICAL MISMATCH: Env loaded {current_id}, expected {target_id}. Skipping.")
                continue

            ego = env.agent
            sensor = env.engine.get_sensor("rgb_camera")

            frame_info = pickle_data["metadata"].get("frame_info", [])
            if not frame_info:
                continue

            samples_dir = os.path.join(dataset_path, "samples")

            num_steps_per_original = 5
            total_sim_steps = pickle_data["length"]

            for i in range(total_sim_steps):
                env.step([0, 0])

                if i % num_steps_per_original == 0:
                    original_frame_idx = i // num_steps_per_original

                    if original_frame_idx >= len(frame_info):
                        break

                    current_images_info = frame_info[original_frame_idx]
                    if not current_images_info:
                        continue

                    for cam_name in CAM_NAMES:
                        if cam_name in current_images_info:
                            original_filename = current_images_info[cam_name]

                            cfg = CAM_CONFIGS[cam_name]
                            if hasattr(sensor.lens, 'setFov'):
                                sensor.lens.setFov(cfg['fov'])

                            img = sensor.perceive(
                                to_float=False,
                                new_parent_node=ego.origin,
                                position=(cfg['y'], cfg['x'], cfg['z']),
                                hpr=(-cfg['yaw'], cfg['pitch'], -cfg['roll'])
                            ).get()

                            save_dir = os.path.join(samples_dir, cam_name)
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, original_filename)
                            cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--database_path", "-d", required=True, help="Output directory for converted scenarios")
    parser.add_argument("--dataset_name", "-n", default="nuscenes")
    parser.add_argument("--split", default="v1.0-mini", choices=scene_split + prediction_split)
    parser.add_argument("--dataroot", default="/data/sets/nuscenes", help="Path to nuScenes raw data")
    parser.add_argument("--map_radius", default=500, type=float)
    parser.add_argument("--future", default=6, type=int)
    parser.add_argument("--past", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--render", action="store_true", help="Generate rendered images after conversion")

    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.split

    if version in scene_split:
        scenarios, nuscs = get_nuscenes_scenarios(args.dataroot, version, args.num_workers)
    else:
        scenarios, nuscs = get_nuscenes_prediction_split(
            args.dataroot, version, args.past, args.future, args.num_workers
        )

    write_to_directory(
        convert_func=convert_nuscenes_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        nuscenes=nuscs,
        past=[args.past for _ in range(args.num_workers)],
        future=[args.future for _ in range(args.num_workers)],
        prediction=[version in prediction_split for _ in range(args.num_workers)],
        map_radius=[args.map_radius for _ in range(args.num_workers)],
    )

    if args.render:
        generate_images(output_path)

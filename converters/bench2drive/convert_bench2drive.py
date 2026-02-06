#!/usr/bin/env python3
"""
Converter script to transform Bench2Drive scenarios to MetaDrive format.
This script focuses on converting vehicles, pedestrians, and non-ego vehicles.
"""

import gzip
import json
import pickle
import numpy as np
import tarfile
import tempfile
import shutil
import re
import os
import cv2
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
import logging
import argparse
import traceback
from tqdm import tqdm
from functools import lru_cache
from importlib.machinery import SourceFileLoader

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.type import MetaDriveType

# *** ADDED IMPORTS ***
from renderer import ScenarioRenderer, COLOR_TABLE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_camera_intrinsics(sensor, width, height, fov):
    """
    Set camera intrinsics to match exactly what Bench2Drive uses.
    This is critical for pixel-perfect alignment.
    """
    try:
        if hasattr(sensor, 'lens'):
            # Calculate exact focal length using Bench2Drive formula
            focal = width / (2.0 * math.tan(fov * math.pi / 360.0))
            
            # Set focal length directly if possible
            if hasattr(sensor.lens, 'setFocalLength'):
                sensor.lens.setFocalLength(focal)
                logger.debug(f"Set focal length to {focal:.3f}")
            
            # Also set FOV (but focal length is more precise)
            if hasattr(sensor.lens, 'setFov'):
                sensor.lens.setFov(fov)
                logger.debug(f"Set FOV to {fov}°")
            
            # Set film size to match resolution
            if hasattr(sensor.lens, 'setFilmSize'):
                sensor.lens.setFilmSize(width, height)
                logger.debug(f"Set film size to {width}x{height}")
            
            # Verify FOV was actually set
            success = True
            if hasattr(sensor.lens, 'getFov'):
                new_fov = sensor.lens.getFov()
                success = abs(float(new_fov.x if hasattr(new_fov, 'x') else new_fov) - fov) < 1.0
                if not success:
                    logger.warning(f"FOV setting may have failed: expected {fov}°, got {new_fov}")
                    
            return success
        else:
            logger.warning("Sensor does not have lens attribute")
    except Exception as e:
        logger.error(f"Failed to set camera intrinsics: {e}")
    return False


@dataclass
class ScenarioBounds:
    """Represents spatial bounds of a scenario"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    
    def contains_point(self, x: float, y: float, buffer: float = 0) -> bool:
        """Check if point is within bounds with optional buffer"""
        return (self.min_x - buffer <= x <= self.max_x + buffer and 
                self.min_y - buffer <= y <= self.max_y + buffer)


@dataclass
class ObjectTypeMapping:
    """Centralised object type conversion logic"""
    
    VEHICLE_TYPES = frozenset(['vehicle', 'ego_vehicle'])
    PEDESTRIAN_TYPES = frozenset(['walker', 'pedestrian'])
    CYCLIST_TYPES = frozenset(['cyclist', 'bicycle', 'bike'])
    
    TRAFFIC_SIGN_MAPPINGS = {
        # vehicles
        'vehicle.bh.crossbike': 'CYCLIST',
        "vehicle.diamondback.century": 'CYCLIST',
        "vehicle.gazelle.omafiets": 'CYCLIST',
        "vehicle.audi.etron": 'VEHICLE',
        "vehicle.chevrolet.impala": 'VEHICLE',
        "vehicle.dodge.charger_2020": 'VEHICLE',
        "vehicle.dodge.charger_police": 'VEHICLE',
        "vehicle.dodge.charger_police_2020": 'VEHICLE',
        "vehicle.lincoln.mkz_2017": 'VEHICLE',
        "vehicle.lincoln.mkz_2020": 'VEHICLE',
        "vehicle.mini.cooper_s_2021": 'VEHICLE',
        "vehicle.mercedes.coupe_2020": 'VEHICLE',
        "vehicle.ford.mustang": 'VEHICLE',
        "vehicle.nissan.patrol_2021": 'VEHICLE',
        'static.prop.constructioncone': 'TRAFFIC_CONE',
        'static.prop.warningaccident': 'TRAFFIC_BARRIER',
        'static.prop.warningconstruction': 'TRAFFIC_BARRIER',
        'traffic.stop': 'STOP_SIGN'
    }
    
    IGNORED_SIGNS = frozenset([
        'static.prop.dirtdebris', 'static.prop.dirtdebris01',
        'static.prop.dirtdebris02', 'traffic.yield'
    ])
    
    STATIC_OBJECT_DIMENSIONS = {
        'TRAFFIC_CONE': (0.5, 0.5, 0.8),
        'TRAFFIC_BARRIER': (2.0, 0.5, 1.0),
        'STOP_SIGN': (0.3, 0.1, 2.5),
        'TRAFFIC_OBJECT': (1.0, 1.0, 2.0)
    }
    
    @classmethod
    def convert(cls, bench2drive_type: str, type_id: Optional[str] = None, base_type: Optional[str] = None) -> Optional[str]:
        """Convert Bench2Drive object type to MetaDrive type"""
        if base_type:
            if base_type in cls.VEHICLE_TYPES:
                return 'VEHICLE'
            if base_type in cls.PEDESTRIAN_TYPES:
                return 'PEDESTRIAN'
            if base_type in cls.CYCLIST_TYPES:
                return 'CYCLIST'

        if bench2drive_type == 'traffic_sign' and type_id:
            if type_id in cls.IGNORED_SIGNS:
                return None
            return cls.TRAFFIC_SIGN_MAPPINGS.get(type_id, 'TRAFFIC_OBJECT')
        
        if type_id and type_id in cls.TRAFFIC_SIGN_MAPPINGS:
             return cls.TRAFFIC_SIGN_MAPPINGS[type_id]

        if bench2drive_type in cls.PEDESTRIAN_TYPES:
            return 'PEDESTRIAN'
        if bench2drive_type in cls.CYCLIST_TYPES:
            return 'CYCLIST'
        elif bench2drive_type == 'traffic_light':
            return 'TRAFFIC_LIGHT'
        elif bench2drive_type == 'traffic_cone':
            return 'TRAFFIC_CONE'
        elif bench2drive_type == 'traffic_barrier':
            return 'TRAFFIC_BARRIER'

        return 'VEHICLE'  # Default
    
    @classmethod
    def get_static_dimensions(cls, obj_type: str) -> Tuple[float, float, float]:
        """Get default dimensions for static objects"""
        return cls.STATIC_OBJECT_DIMENSIONS.get(obj_type, (1.0, 1.0, 1.0))
    
    @classmethod
    def is_static(cls, obj_type: str) -> bool:
        """Check if object type is static"""
        return obj_type in cls.STATIC_OBJECT_DIMENSIONS


class Bench2DriveConverter:
    """Converter from Bench2Drive format to MetaDrive format"""
    
    # Class constants
    TRAFFIC_LIGHT_STATE_MAP = {
        0: 'LANE_STATE_STOP',     # Red
        1: 'LANE_STATE_CAUTION',  # Yellow
        2: 'LANE_STATE_GO'         # Green
    }
    
    LANE_TYPE_MAP = {
        'SolidSolid': 'ROAD_LINE_SOLID_DOUBLE',
        'Solid': 'ROAD_LINE_SOLID_SINGLE',
        'Broken': 'ROAD_LINE_BROKEN_SINGLE'
    }
    
    def __init__(self, scenario_path: str, hd_map_path: str, output_dir: Optional[str] = None, generate_sensors: bool = True, generate_rasters: bool = True, parked_vehicles_path: Optional[str] = None):
        self.scenario_path = Path(scenario_path)
        self.generate_sensors = generate_sensors
        self.generate_rasters = generate_rasters
        self.parked_vehicles_path = parked_vehicles_path
        self._setup_paths(output_dir)
        self._extract_if_needed()
        self._load_hd_map(hd_map_path)
        self.parked_vehicles = self._load_parked_vehicles()
        self._load_annotations()
        
        # Initialize MetaDrive environment for sensor data generation
        if self.generate_sensors:
            self.env = None
            self.renderer = None # Will be initialized in generate_sensor_data
            self._setup_metadrive_env()
        
    def _setup_paths(self, output_dir: Optional[str]):
        """Setup output directory and scenario name"""
        if output_dir is None:
            base_name = (self.scenario_path.stem.replace('.tar', '') 
                        if str(self.scenario_path).endswith('.tar.gz')
                        else self.scenario_path.name)
            self.output_dir = Path("converted_scenarios") / base_name
            self.scenario_name = base_name
        else:
            self.output_dir = Path(output_dir)
            self.scenario_name = self.output_dir.name
        
        self.scenario_id = self.scenario_name.replace('_', '-')
        self.town_name = self._extract_town_name()
    
    def _extract_if_needed(self):
        """Extract tar.gz if needed"""
        self.temp_dir = None
        
        if str(self.scenario_path).endswith('.tar.gz'):
            logger.info(f"Extracting {self.scenario_path}...")
            self.temp_dir = tempfile.mkdtemp()
            
            with tarfile.open(self.scenario_path, 'r:gz') as tar:
                tar.extractall(path=self.temp_dir)
            
            temp_contents = list(Path(self.temp_dir).iterdir())
            if len(temp_contents) != 1 or not temp_contents[0].is_dir():
                raise ValueError(f"Expected single directory in tar.gz, found: {temp_contents}")
            
            self.extracted_scenario_path = temp_contents[0]
        else:
            self.extracted_scenario_path = self.scenario_path
        
        self.anno_dir = self.extracted_scenario_path / "anno"
        if not self.anno_dir.exists():
            raise ValueError(f"Anno directory not found: {self.anno_dir}")
    
    def _extract_town_name(self) -> str:
        """Extract town name from scenario path"""
        match = re.search(r'Town(\d+)', self.scenario_name)
        return f"Town{match.group(1)}" if match else "UnknownTown"
    
    def _load_hd_map(self, hd_map_path: str):
        """Load HD map data - required for conversion"""
        self.hd_map_data = None
        self._lane_id_mapping = {}
        
        hd_map_path = Path(hd_map_path)
        if not hd_map_path.exists():
            raise FileNotFoundError(f"HD map file not found: {hd_map_path}")
        
        if not hd_map_path.suffix == '.npz':
            raise ValueError(f"HD map file must be a .npz file, got: {hd_map_path}")
        
        # Validate that HD map town matches scenario town
        hd_map_town_match = re.search(r'Town(\d+)', hd_map_path.name)
        if hd_map_town_match and self.town_name != "UnknownTown":
            hd_map_town = f"Town{hd_map_town_match.group(1)}"
            if hd_map_town != self.town_name:
                logger.warning(f"HD map town ({hd_map_town}) does not match scenario town ({self.town_name}). "
                             f"This may cause issues with map feature extraction.")
        
        try:
            logger.info(f"Loading HD map from: {hd_map_path}")
            with np.load(hd_map_path, allow_pickle=True) as data:
                if 'arr' not in data:
                    raise ValueError(f"HD map file is missing required 'arr' key: {hd_map_path}")
                self.hd_map_data = data['arr']
            logger.info(f"Loaded HD map with {len(self.hd_map_data)} roads")
            
            # Build lane ID mapping for traffic lights
            self._build_lane_id_mapping()
        except Exception as e:
            raise RuntimeError(f"Failed to load HD map from {hd_map_path}: {e}")
    
    def _load_annotations(self):
        """Load and cache all annotation files"""
        self.annotation_files = sorted(list(self.anno_dir.glob("*.json.gz")))
        self.track_length = len(self.annotation_files)
        
        # Pre-load all annotations for efficiency
        logger.info(f"Loading {self.track_length} annotation frames...")
        self.annotations = []
        for frame_idx in tqdm(range(self.track_length), desc="Loading annotations"):
            self.annotations.append(self._load_single_annotation(frame_idx))
        
        logger.info(f"Loaded {self.track_length} frames for scenario {self.scenario_name}")
    
    def _load_single_annotation(self, frame_idx: int) -> Dict[str, Any]:
        """Load a single annotation file"""
        anno_file = self.anno_dir / f"{frame_idx:05d}.json.gz"
        if not anno_file.exists():
            return {}
        
        with gzip.open(anno_file, 'rt') as f:
            return json.load(f)
    
    def _build_lane_id_mapping(self):
        """Build mapping from (road_id, lane_id) to map feature IDs for traffic lights"""
        self._lane_id_mapping = {}
        
        if self.hd_map_data is None:
            return
        
        for road_id, lane_data in self.hd_map_data:
            if not isinstance(lane_data, dict):
                continue
            
            for lane_id, lane_info in lane_data.items():
                if lane_id == "Trigger_Volumes" or not isinstance(lane_info, list):
                    continue
                
                # Look for Center lane elements (actual drivable lanes)
                for element_idx, element in enumerate(lane_info):
                    if (isinstance(element, dict) and 
                        element.get("Type") == "Center"):
                        
                        # Create the map feature ID
                        feature_id = f"road_{road_id}_lane_{lane_id}_element_{element_idx}"
                        
                        # Map from (road_id, lane_id) to feature_id
                        lane_key = (road_id, int(lane_id))
                        self._lane_id_mapping[lane_key] = feature_id
                        
                        # Also try negative lane_id (CARLA uses negative for opposite direction)
                        neg_lane_key = (road_id, -int(lane_id))
                        if neg_lane_key not in self._lane_id_mapping:
                            self._lane_id_mapping[neg_lane_key] = feature_id
                        break  # Only need first center element per lane

    def _load_parked_vehicles(self) -> List[Dict[str, Any]]:
        """Load parked vehicle data for the current town."""
        if not self.parked_vehicles_path:
            logger.info("No parked vehicles path provided, skipping.")
            return []

        parked_vehicles_file = Path(self.parked_vehicles_path)
        if not parked_vehicles_file.exists():
            logger.warning(f"Parked vehicles file not found: {parked_vehicles_file}")
            return []

        try:
            module = SourceFileLoader("parked_vehicles", str(parked_vehicles_file)).load_module()
            town_data = getattr(module, self.town_name.replace("Town", "Town"), None)
            if town_data:
                logger.info(f"Loaded {len(town_data)} parked vehicles for {self.town_name}")
                return town_data
        except Exception as e:
            logger.error(f"Failed to load parked vehicles from {parked_vehicles_file}: {e}")
        return []
    
    def _map_traffic_light_to_lane(self, road_id: int, lane_id: int) -> Optional[str]:
        """Map Bench2Drive road_id/lane_id to map feature lane ID"""
        if not hasattr(self, '_lane_id_mapping') or not self._lane_id_mapping:
            return None
        
        # Try to find matching lane center element
        lane_key = (road_id, lane_id)
        return self._lane_id_mapping.get(lane_key)
    
    def _cleanup(self):
        """Clean up temporary directory if created"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
    
    def _setup_metadrive_env(self):
        """Setup MetaDrive environment for sensor data generation"""
        # Configure sensors according to Bench2Drive specifications
        sensors = {
            # Camera sensors with Bench2Drive resolution (1600x900)
            "rgb_camera": (RGBCamera, 1600, 900),
            # "depth_camera": (DepthCamera, 1600, 900), 
            # "semantic_camera": (SemanticCamera, 1600, 900),
            # "instance_camera": (InstanceCamera, 1600, 900)
        }
        
        # Store sensor config for later use - ScenarioEnv will be created after scenario is saved
        self.sensor_configs = sensors
        logger.info("MetaDrive sensor configuration prepared")
    
    def _create_sensor_directories(self):
        """Create directory structure for sensor data"""
        sensor_dirs = {
            'camera': ['rgb_front', 'rgb_front_left', 'rgb_front_right', 'rgb_back', 'rgb_back_left', 'rgb_back_right', 'rgb_top_down',
                       'depth_front', 'depth_front_left', 'depth_front_right', 'depth_back', 'depth_back_left', 'depth_back_right',
                       'semantic_front', 'semantic_front_left', 'semantic_front_right', 'semantic_back', 'semantic_back_left', 'semantic_back_right',
                       'instance_front', 'instance_front_left', 'instance_front_right', 'instance_back', 'instance_back_left', 'instance_back_right',
                      ]
        }
        if self.generate_rasters:
            sensor_dirs['camera'].extend(['raster_f0', 'raster_l0', 'raster_l1', 'raster_l2', 'raster_r0', 'raster_r1', 'raster_r2', 'raster_b0'])
        
        base_sensor_dir = self.output_dir
        
        # Create camera subdirectories
        for sensor_type in sensor_dirs['camera']:
            sensor_path = base_sensor_dir / 'camera' / sensor_type
            sensor_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created sensor directories in {base_sensor_dir}")
        return base_sensor_dir
    
    def generate_sensor_data(self, scenario_file_path: str):
        """Generate sensor data for all frames of the scenario using the saved scenario file"""
        if not self.generate_sensors:
            logger.info("Sensor data generation disabled")
            return
            
        logger.info("Starting sensor data generation...")
        sensor_base_dir = self._create_sensor_directories()
        
        # Use the existing output directory as the dataset directory
        # The scenario file and dataset files should already exist there
        dataset_dir = self.output_dir
        
        # Configure ScenarioEnv with sensors matching Bench2Drive specifications
        config = {
            "use_render": False,  # Required for camera sensors
            "image_on_cuda": True,
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "num_scenarios": 1,
            "horizon": 1000,
            "sensors": self.sensor_configs,
            "data_directory": str(dataset_dir.absolute()),
            
            # Image and camera settings
            "image_observation": True,  # Required for camera sensors
            
            # Vehicle configuration
            "vehicle_config": {
                "image_source": "rgb_camera",
                "no_wheel_friction": False
            },
            
            # UI settings
            "show_interface": False,  # Hide UI while generating data
            "show_logo": False,
            "show_fps": False,
            "window_size": (800, 600),
        }
        
        # *** INITIALIZE RENDERER ***
        if self.generate_rasters:
            self.renderer = ScenarioRenderer(
                camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0']
            )
        else:
            self.renderer = None
        
        # Initialize ScenarioEnv
        self.env = ScenarioEnv(config)
        
        try:
            # Reset to load the scenario
            obs, _ = self.env.reset(seed=0)
            logger.info("ScenarioEnv reset successful")
            
            # Get scenario length
            scenario_length = self.env.engine.data_manager.current_scenario_length
            logger.info(f"Scenario length: {scenario_length} frames")
            
            # Check available sensors
            logger.info(f"Available sensors: {list(self.env.engine.sensors.keys())}")
            
            # Check ego vehicle
            ego_vehicle = self.env.agent
            logger.info(f"Ego vehicle: {ego_vehicle}")
            if ego_vehicle:
                logger.info(f"Ego position: {ego_vehicle.position}")
            
            # Generate sensor data for each frame
            for frame_idx in tqdm(range(scenario_length), desc="Generating sensor data"):
                logger.debug(f"Processing frame {frame_idx}")
                
                # Step environment to update the scenario state
                obs, _, _, _, _ = self.env.step([0.0, 0.0])  # No action, just replay
                
                # Capture sensor data
                self._capture_frame_sensors(frame_idx, sensor_base_dir)
                
                # *** ADDED RASTERIZATION STEP ***
                if self.generate_rasters and self.renderer:
                    try:
                        # 1. Build the scenario dict from the current env state
                        raster_scenario_dict = self._build_raster_scenario_dict(frame_idx)
                        
                        # 2. Render the hardcoded nuPlan camera views
                        rendered_cameras = self.renderer.observe(raster_scenario_dict)
                        
                        # 3. Save the images
                        frame_str = f"{frame_idx:05d}.jpg"
                        for cam_id, image in rendered_cameras.items():
                            # Map cam_id (e.g., 'CAM_F0') to dir name (e.g., 'raster_f0')
                            raster_dir_name = f"raster_{cam_id.split('_')[-1].lower()}"
                            output_path = sensor_base_dir / 'camera' / raster_dir_name / frame_str
                            # save as BGR
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(output_path), image)
                            
                    except Exception as e:
                        logger.error(f"Failed to generate raster image for frame {frame_idx}: {e}")
                        traceback.print_exc()
                
                # Check if scenario is complete
                if self.env.episode_step >= scenario_length:
                    logger.info(f"Scenario completed at frame {frame_idx}")
                    break
            
            logger.info("Sensor data generation completed")
                    
        except Exception as e:
            logger.error(f"Error during sensor data generation: {e}")
            traceback.print_exc()
            
        finally:
            if self.env:
                self.env.close()
                self.env = None

    def _build_raster_scenario_dict(self, frame_idx: int) -> Dict[str, Any]:
        """
        Build the 'scenario' dictionary required by the ScenarioRenderer
        by querying the current state of the MetaDrive environment.
        """
        if not self.env or not self.env.agent:
            return {}

        ego_vehicle = self.env.agent
        scenario = {
            'ego_pos_3d': np.array(ego_vehicle.get_state()["position"]),
            'ego_heading': ego_vehicle.heading_theta,
            'map_features': self.env.engine.data_manager.current_scenario["map_features"],
            'traffic_lights': []
        }

        # --- 1. Get Traffic Light States ---
        dynamic_map = self.env.engine.data_manager.current_scenario["dynamic_map_states"]
        for lane_id, tl_data in dynamic_map.items():
            if frame_idx < len(tl_data["state"]["object_state"]):
                state_str = tl_data["state"]["object_state"][frame_idx]
                is_red = (state_str == "LANE_STATE_STOP")
                # Use the stop_point as the [x, y] position
                # The renderer's `draw_cuboid_at` expects [x, y, z_base]
                # stop_point is [x, y, z], so we can use [x, y]
                pos_xy = tl_data["stop_point"][:2]
                scenario['traffic_lights'].append((lane_id, is_red, pos_xy))

        # --- 2. Get All Other Dynamic Objects (Vehicles, Pedestrians) ---
        gt_boxes_world = []
        gt_names = []

        # Get all objects from the MetaDrive engine
        objects = self.env.engine.get_objects(filter=lambda o: o.name != ego_vehicle.name)
        
        for obj in objects.values():
            if obj.metadrive_type == MetaDriveType.TRAFFIC_LIGHT:
                continue  # Skip traffic lights, already processed

            obj_state = obj.get_state()

            # Skip objects with no usable geometry (e.g., static cones/barriers without length/width/height)
            length = obj_state.get("length", getattr(obj, "LENGTH", None))
            width = obj_state.get("width", getattr(obj, "WIDTH", None))
            height = obj_state.get("height", getattr(obj, "HEIGHT", None))
            position = obj_state.get("position")
            if position is None or length is None or width is None or height is None:
                logger.debug(f"Skipping object {obj.name} ({obj.metadrive_type}) due to missing size/position")
                continue

            # Ignore far-away objects to match renderer depth culling
            max_depth = getattr(self.renderer, "depth_max", None)
            if max_depth is not None:
                ego_xy = np.array(ego_vehicle.position[:2])
                obj_xy = np.array(position[:2])
                if np.linalg.norm(obj_xy - ego_xy) > (max_depth + 10.0):  # small buffer
                    continue

            # Format: [x, y, z, L, W, H, yaw]
            box_data = [
                position[0],
                position[1],
                position[2],
                length,
                width,
                height,
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

    def _capture_frame_sensors(self, frame_idx: int, sensor_base_dir: Path):
        """Capture sensor data for a single frame using per-frame annotation sensor configurations"""
        frame_str = f"{frame_idx:05d}"
        if frame_idx == 0:  # Only log for first frame
            logger.info(f"Capturing sensors for frame {frame_str} using per-frame annotation data")
        
        # Get ego vehicle for sensor positioning
        ego_vehicle = self.env.agent
        if not ego_vehicle:
            logger.warning(f"No ego vehicle found for frame {frame_idx}")
            return
        
        # Get annotation data for this frame
        if frame_idx >= len(self.annotations):
            logger.warning(f"Frame {frame_idx} exceeds available annotations ({len(self.annotations)})")
            return
            
        annotation = self.annotations[frame_idx]
        if not annotation or "sensors" not in annotation:
            logger.warning(f"No sensor data available for frame {frame_idx}")
            return
        
        sensors_data = annotation["sensors"]
        
        if frame_idx == 0:  # Only log for first frame
            logger.info(f"Ego vehicle position: {ego_vehicle.position}")
            logger.info(f"Available sensors: {list(self.env.engine.sensors.keys())}")
            logger.info(f"Annotation sensors available: {list(sensors_data.keys())}")
            
        # Process each camera sensor from annotation data
        for sensor_name, sensor_config in sensors_data.items():
            # Skip non-camera sensors (radar, lidar)
            # Include CAM_* sensors and TOP_DOWN (which is a camera sensor)
            if not (sensor_name.startswith('CAM_') or sensor_name == 'TOP_DOWN'):
                continue
                
            # Extract sensor configuration from annotation
            fov = sensor_config["fov"]
            image_size_x = sensor_config["image_size_x"]
            image_size_y = sensor_config["image_size_y"]
            intrinsic = sensor_config["intrinsic"]
            cam2ego = sensor_config["cam2ego"]  # 4x4 matrix with relative position/rotation
            
            # Extract relative position and rotation from cam2ego matrix
            # cam2ego matrix contains the sensor position relative to ego vehicle
            md_x, md_y, md_z, md_yaw, md_pitch, md_roll = self._extract_relative_pose_from_cam2ego_matrix(
                cam2ego
            )
            
            # Map sensor names to directory names
            direction_mapping = {
                'CAM_FRONT': 'front',
                'CAM_FRONT_LEFT': 'front_left',
                'CAM_FRONT_RIGHT': 'front_right',
                'CAM_BACK': 'back',
                'CAM_BACK_LEFT': 'back_left',
                'CAM_BACK_RIGHT': 'back_right',
                'TOP_DOWN': 'top_down'
            }
            
            direction = direction_mapping.get(sensor_name, sensor_name.lower())
            
            if frame_idx == 0:  # Log sensor configuration summary
                sensor_types_str = "RGB only" if sensor_name == 'TOP_DOWN' else "RGB+Depth+Semantic+Instance"
                logger.info(f"Configuring {sensor_name}: FOV={fov}°, size={image_size_x}x{image_size_y}, types={sensor_types_str}")
                logger.debug(f"  MetaDrive relative pos: ({md_x:.2f}, {md_y:.2f}, {md_z:.2f}), rot: ({md_yaw:.1f}°, {md_pitch:.1f}°, {md_roll:.1f}°)")
            
            # Capture sensor data with per-frame configuration
            self._capture_sensor_with_config(
                sensor_name, direction, md_x, md_y, md_z, md_yaw, md_pitch, md_roll,
                fov, image_size_x, image_size_y, intrinsic,
                ego_vehicle, frame_idx, frame_str, sensor_base_dir
            )
    
    def _extract_relative_pose_from_cam2ego_matrix(self, cam2ego_matrix):
        """Extract relative sensor position and rotation from cam2ego matrix.
        
        The cam2ego matrix contains the transformation from camera to ego vehicle coordinates.
        This gives us the relative position of the sensor with respect to the ego vehicle.
        """
        # Convert to numpy array if it's a list
        if isinstance(cam2ego_matrix, list):
            cam2ego = np.array(cam2ego_matrix)
        else:
            cam2ego = cam2ego_matrix
        
        # Extract position (relative to ego vehicle)
        rel_x = cam2ego[0, 3]
        rel_y = cam2ego[1, 3] 
        rel_z = cam2ego[2, 3]
        
        # Extract rotation angles from rotation matrix
        # This follows the same extraction method as in extract_position_rotation_from_matrix        
        # Extract yaw (rotation around Z-axis)
        yaw = math.atan2(cam2ego[1, 0], cam2ego[0, 0])
        
        # Extract pitch (rotation around Y-axis)  
        pitch = math.asin(np.clip(cam2ego[2, 0], -1.0, 1.0))
        
        # Extract roll (rotation around X-axis)
        roll = math.atan2(-cam2ego[2, 1], cam2ego[2, 2])
        
        # Convert to degrees
        yaw_deg = math.degrees(yaw)
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        
        # Convert coordinate system from CARLA (left-handed) to MetaDrive (right-handed)
        # For MetaDrive, we need to:
        # 1. Invert the Y coordinate (CARLA left-handed vs MetaDrive right-handed)
        # 2. Adjust yaw angle accordingly
        
        md_x = rel_y
        md_y = rel_x  # Invert Y coordinate
        md_z = rel_z
        
        # Adjust angles - MetaDrive uses different angle conventions
        md_yaw = -yaw_deg    # Invert yaw due to Y-axis flip
        md_pitch = pitch_deg  # Pitch stays the same
        md_roll = -roll_deg   # Roll inverts due to coordinate system change
        
        return md_x, md_y, md_z, md_yaw, md_pitch, md_roll
    
    def _capture_sensor_with_config(self, sensor_name, direction, x, y, z, yaw, pitch, roll,
                                   fov, image_size_x, image_size_y, intrinsic,
                                   ego_vehicle, frame_idx, frame_str, sensor_base_dir):
        """Capture sensor data with specific configuration from annotation data"""
        
        # TOP_DOWN sensor is only available as RGB, not depth/semantic/instance
        if sensor_name == 'TOP_DOWN':
            sensor_types = ["rgb_camera"]  # Only RGB for TOP_DOWN
        else:
            sensor_types = ["rgb_camera", "depth_camera", "semantic_camera", "instance_camera"]
        
        for sensor_type in sensor_types:
            if sensor_type not in self.env.engine.sensors:
                continue
                
            try:
                sensor = self.env.engine.get_sensor(sensor_type)
                
                # Set FOV using the exact value from annotation
                success = set_camera_intrinsics(sensor, image_size_x, image_size_y, fov)
                if frame_idx == 0 and not success:
                    logger.warning(f"Failed to set FOV for {sensor_type} to {fov}° for {sensor_name}")
                
                # Capture sensor data with exact positioning from annotation
                sensor_data = sensor.perceive(
                    to_float=False,
                    new_parent_node=ego_vehicle.origin,
                    position=(x, y, z),
                    hpr=(yaw, pitch, roll)
                ).get()
                
                # Save the captured image
                if sensor_type == "rgb_camera":
                    output_path = sensor_base_dir / 'camera' / f"rgb_{direction}" / f"{frame_str}.jpg"
                elif sensor_type == "depth_camera":
                    output_path = sensor_base_dir / 'camera' / f"depth_{direction}" / f"{frame_str}.jpg"
                elif sensor_type == "semantic_camera":
                    output_path = sensor_base_dir / 'camera' / f"semantic_{direction}" / f"{frame_str}.jpg"
                elif sensor_type == "instance_camera":
                    output_path = sensor_base_dir / 'camera' / f"instance_{direction}" / f"{frame_str}.jpg"
                
                cv2.imwrite(str(output_path), sensor_data)
                
                if frame_idx == 0:
                    logger.debug(f"Captured {sensor_type} for {sensor_name}")
                    
            except Exception as e:
                logger.error(f"Error capturing {sensor_type} for {sensor_name}: {e}")
                if frame_idx == 0:
                    traceback.print_exc()
    
    def _build_object_registry(self) -> Tuple[Dict[str, str], Optional[str]]:
        """Build registry of all objects and find SDC ID"""
        object_registry = {}
        sdc_id = None
        
        for anno in tqdm(self.annotations, desc="Building object registry"):
            if not anno or "bounding_boxes" not in anno:
                continue
            
            for obj in anno["bounding_boxes"]:
                # Check for ego vehicle
                if obj["class"] == "ego_vehicle" and sdc_id is None:
                    sdc_id = str(obj["id"])
                
                obj_type = ObjectTypeMapping.convert(
                    obj["class"], obj.get("type_id"), obj.get("base_type")
                )
                
                if obj_type is None:
                    continue
                
                obj_id = str(obj["id"])
                if obj_id not in object_registry:
                    object_registry[obj_id] = obj_type
                elif object_registry[obj_id] != obj_type:
                    raise ValueError(f"Object {obj_id} type mismatch")

        # Add parked vehicles to the registry
        for i, pv in enumerate(self.parked_vehicles):
            obj_id = f"parked_{i}"
            # Use the mesh path as a type_id for mapping
            type_id = pv.get('mesh')
            # Parked vehicles are always of class 'vehicle'
            obj_type = ObjectTypeMapping.convert('vehicle', type_id=type_id)

            if obj_type:
                if obj_id not in object_registry:
                    object_registry[obj_id] = obj_type
                elif object_registry[obj_id] != obj_type:
                    raise ValueError(f"Object {obj_id} type mismatch")
        
        return object_registry, sdc_id
    
    def _create_track_templates(self, object_registry: Dict[str, str]) -> Tuple[Dict, Dict]:
        """Create track and dynamic map state templates"""
        tracks = {}
        dynamic_map_states = {}
        
        for obj_id, obj_type in object_registry.items():
            if obj_type == "TRAFFIC_LIGHT":
                dynamic_map_states[obj_id] = self._create_traffic_light_template(obj_id)
            else:
                tracks[obj_id] = self._create_object_template(obj_id, obj_type)
        
        return tracks, dynamic_map_states
    
    def _create_object_template(self, object_id: str, obj_type: str) -> Dict[str, Any]:
        """Create template for object state data"""
        return {
            "type": obj_type,
            "state": {
                "position": np.zeros([self.track_length, 3], dtype=np.float32),
                "length": np.zeros([self.track_length], dtype=np.float32),
                "width": np.zeros([self.track_length], dtype=np.float32),
                "height": np.zeros([self.track_length], dtype=np.float32),
                "heading": np.zeros([self.track_length], dtype=np.float32),
                "velocity": np.zeros([self.track_length, 2], dtype=np.float32),
                "valid": np.zeros([self.track_length], dtype=bool),
            },
            "metadata": {
                "track_length": self.track_length,
                "type": obj_type,
                "object_id": object_id,
                "dataset": "bench2drive"
            }
        }
    
    def _create_traffic_light_template(self, object_id: str) -> Dict[str, Any]:
        """Create template for traffic light dynamic states"""
        return {
            "type": "TRAFFIC_LIGHT",
            "state": {
                "object_state": ["LANE_STATE_UNKNOWN"] * self.track_length
            },
            "lane": None,
            "stop_point": np.zeros([3], dtype=np.float32),
            "metadata": {
                "track_length": self.track_length,
                "type": "TRAFFIC_LIGHT",
                "object_id": object_id,
                "dataset": "bench2drive"
            }
        }
    
    def _process_frame_objects(self, frame_idx: int, anno: Dict, 
                               tracks: Dict, dynamic_map_states: Dict):
        """Process all objects in a single frame"""
        if not anno or "bounding_boxes" not in anno:
            return
        
        # Build frame lookup for efficiency
        frame_objects = {}
        for obj in anno["bounding_boxes"]:
            obj_type = ObjectTypeMapping.convert(obj["class"], obj.get("type_id"), obj.get("base_type"))
            if obj_type is not None:
                frame_objects[str(obj["id"])] = (obj, obj_type)
        
        # Process traffic lights
        for obj_id, tl_state in dynamic_map_states.items():
            if obj_id in frame_objects:
                obj, _ = frame_objects[obj_id]
                self._update_traffic_light(tl_state, obj, frame_idx)
        
        # Process regular objects
        for obj_id, track in tracks.items():
            if obj_id not in frame_objects:
                continue
            
            obj, obj_type = frame_objects[obj_id]
            self._update_object_state(track, obj, obj_type, frame_idx)
    
    def _update_traffic_light(self, tl_state: Dict, obj: Dict, frame_idx: int):
        """Update traffic light state for a frame"""
        if "state" in obj:
            state = self.TRAFFIC_LIGHT_STATE_MAP.get(obj["state"], "LANE_STATE_UNKNOWN")
            tl_state["state"]["object_state"][frame_idx] = state
        
        # Set position once (traffic lights are stationary)
        if np.all(tl_state["stop_point"] == 0) and "location" in obj:
            tl_state["stop_point"] = np.array([
                float(obj["location"][0]),
                -float(obj["location"][1]),
                float(obj["location"][2])
            ], dtype=np.float32)
            
            # Map Bench2Drive lane ID to map feature lane ID
            if tl_state["lane"] is None and "road_id" in obj and "lane_id" in obj:
                mapped_lane_id = self._map_traffic_light_to_lane(obj["road_id"], obj["lane_id"])
                if mapped_lane_id:
                    tl_state["lane"] = mapped_lane_id
    
    def _update_object_state(self, track: Dict, obj: Dict, obj_type: str, frame_idx: int):
        """Update object state for a frame"""
        state = track["state"]
        
        # Position
        if "location" in obj:
            state["position"][frame_idx] = [
                float(obj["location"][0]),
                -float(obj["location"][1]),  # Invert Y-axis
                float(obj["location"][2])
            ]
        
        # Dimensions
        if "extent" in obj:
            state["length"][frame_idx] = float(obj["extent"][0]) * 2
            state["width"][frame_idx] = float(obj["extent"][1]) * 2
            state["height"][frame_idx] = float(obj["extent"][2]) * 2
        elif ObjectTypeMapping.is_static(obj_type):
            length, width, height = ObjectTypeMapping.get_static_dimensions(obj_type)
            state["length"][frame_idx] = length
            state["width"][frame_idx] = width
            state["height"][frame_idx] = height
        
        # Heading
        if "rotation" in obj:
            yaw_degrees = float(obj["rotation"][2])
            state["heading"][frame_idx] = np.radians(-yaw_degrees)
        
        # Velocity
        if ObjectTypeMapping.is_static(obj_type):
            state["velocity"][frame_idx] = [0.0, 0.0]
        elif "speed" in obj and obj["speed"] is not None:
            speed = float(obj["speed"])
            heading = state["heading"][frame_idx]
            state["velocity"][frame_idx] = [
                speed * np.cos(heading),
                speed * np.sin(heading)
            ]
        elif frame_idx > 0 and state["valid"][frame_idx - 1]:
            # Calculate from position difference
            curr_pos = state["position"][frame_idx][:2]
            prev_pos = state["position"][frame_idx - 1][:2]
            dt = 0.05  # 50ms
            state["velocity"][frame_idx] = (curr_pos - prev_pos) / dt
        else:
            state["velocity"][frame_idx] = [0.0, 0.0]
        
        state["valid"][frame_idx] = True
    
    def _interpolate_tracks(self, tracks: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply carry-forward interpolation for missing frames"""
        for obj_id, track in tqdm(tracks.items(), desc="Interpolating tracks"):
            valid_mask = track["state"]["valid"]
            valid_frames = np.where(valid_mask)[0]
            
            if len(valid_frames) == 0 or len(valid_frames) == self.track_length:
                continue
            
            # Carry-forward approach
            for i in range(self.track_length):
                if not valid_mask[i] and i > 0:
                    prev_valid_indices = valid_frames[valid_frames < i]
                    if len(prev_valid_indices) > 0:
                        last_valid_idx = prev_valid_indices[-1]
                        # Copy last known state
                        for key in ['position', 'heading', 'length', 'width', 'height']:
                            track["state"][key][i] = track["state"][key][last_valid_idx]
                        track["state"]["velocity"][i] = [0.0, 0.0]
        
        return tracks
    
    def _post_process_traffic_lights(self, dynamic_map_states: Dict) -> Dict:
        """Post-process traffic light states"""
        for traffic_light in dynamic_map_states.values():
            states = traffic_light["state"]["object_state"]
            last_known = "LANE_STATE_UNKNOWN"
            
            # Carry forward last known state
            for i in range(len(states)):
                if states[i] != "LANE_STATE_UNKNOWN":
                    last_known = states[i]
                elif last_known != "LANE_STATE_UNKNOWN":
                    states[i] = last_known
        
        # Re-key by lane IDs
        rekeyed = {}
        for tl_id, tl_data in dynamic_map_states.items():
            lane_id = tl_data.get("lane", tl_id)
            tl_data["lane"] = lane_id
            rekeyed[lane_id] = tl_data
        
        return rekeyed
    
    def extract_tracks(self) -> Tuple[Dict[str, Dict], str, Dict[str, Dict]]:
        """Extract all object tracks from annotations"""
        # Build object registry
        object_registry, sdc_id = self._build_object_registry()
        
        # Create templates
        tracks, dynamic_map_states = self._create_track_templates(object_registry)
        
        # Process all frames
        for frame_idx, anno in enumerate(tqdm(self.annotations, desc="Processing frames")):
            # Add static parked vehicles state for each frame
            for i, pv in enumerate(self.parked_vehicles):
                obj_id = f"parked_{i}"
                if obj_id in tracks:
                    track = tracks[obj_id]
                    state = track["state"]
                    if not state["valid"][frame_idx]: # Only set once
                        loc = pv['location']
                        rot = pv['rotation']
                        # Assuming extent is not directly available, use default or derive if possible
                        # For now, let's use a default car size
                        state["position"][frame_idx] = [float(loc[0]), -float(loc[1]), float(loc[2])]
                        state["heading"][frame_idx] = np.radians(-float(rot[2])) # Yaw
                        state["length"][frame_idx] = 4.5
                        state["width"][frame_idx] = 2.0
                        state["height"][frame_idx] = 1.8
                        state["velocity"][frame_idx] = [0.0, 0.0]
                        state["valid"][frame_idx] = True

                        # Back-fill for all frames since it's static
                        if frame_idx == 0:
                            for key in state:
                                if key != "valid":
                                    state[key][:] = state[key][0]
                            state["valid"][:] = True
            self._process_frame_objects(frame_idx, anno, tracks, dynamic_map_states)
        
        # Post-processing
        tracks = self._interpolate_tracks(tracks)
        dynamic_map_states = self._post_process_traffic_lights(dynamic_map_states)
        
        return tracks, sdc_id, dynamic_map_states
    
    @lru_cache(maxsize=1)
    def _get_scenario_bounds(self, buffer: float = 40.0) -> Optional[ScenarioBounds]:
        """Get scenario spatial bounds with caching"""
        # Sample frames to determine bounds
        sample_size = min(20, self.track_length)
        sample_indices = np.linspace(0, self.track_length - 1, sample_size, dtype=int)
        
        positions = []
        for idx in sample_indices:
            anno = self.annotations[idx]
            if anno and "bounding_boxes" in anno:
                for obj in anno["bounding_boxes"]:
                    if obj["class"] in ["vehicle", "ego_vehicle", "walker", "pedestrian", "cyclist"]:
                        if "location" in obj:
                            x = float(obj["location"][0])
                            y = -float(obj["location"][1])
                            positions.append([x, y])
        
        if not positions:
            logger.warning("No positions found for bounds calculation")
            return None
        
        positions = np.array(positions)
        return ScenarioBounds(
            np.min(positions[:, 0]) - buffer,
            np.max(positions[:, 0]) + buffer,
            np.min(positions[:, 1]) - buffer,
            np.max(positions[:, 1]) + buffer
        )
    
    def _filter_relevant_roads(self, bounds: ScenarioBounds) -> List[int]:
        """Filter roads within scenario bounds"""
        if self.hd_map_data is None:
            return []
        
        relevant_indices = []
        
        for idx, (road_id, lane_data) in enumerate(self.hd_map_data):
            if not isinstance(lane_data, dict):
                continue
            
            # Quick spatial check
            if self._road_intersects_bounds(lane_data, bounds):
                relevant_indices.append(idx)
        
        return relevant_indices
    
    def _road_intersects_bounds(self, lane_data: Dict, bounds: ScenarioBounds) -> bool:
        """Check if road intersects with bounds"""
        # Check trigger volumes
        if "Trigger_Volumes" in lane_data:
            for volume in lane_data.get("Trigger_Volumes", []):
                if self._feature_in_bounds(volume.get("Points", []), bounds):
                    return True
        
        # Check lanes
        for lane_id, lane_info in lane_data.items():
            if lane_id == "Trigger_Volumes" or not isinstance(lane_info, list):
                continue
            
            for element in lane_info:
                if isinstance(element, dict) and self._feature_in_bounds(element.get("Points", []), bounds):
                    return True
        
        return False
    
    def _feature_in_bounds(self, points: List, bounds: ScenarioBounds) -> bool:
        """Check if feature points intersect bounds"""
        if not points:
            return False
        
        # Sample points for efficiency
        sample_size = min(10, len(points))
        indices = np.linspace(0, len(points) - 1, sample_size, dtype=int)
        
        for idx in indices:
            point = points[idx]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                location = point[0] if isinstance(point[0], (list, tuple)) else point
                if isinstance(location, (list, tuple)) and len(location) >= 3:
                    x = float(location[0])
                    y = -float(location[1])
                    if bounds.contains_point(x, y):
                        return True
        
        return False
    
    def extract_map_features(self) -> Dict[str, Any]:
        """Extract map features from HD map data"""
        if self.hd_map_data is None:
            raise RuntimeError("HD map data is required but not available")
        
        bounds = self._get_scenario_bounds()
        if bounds is None:
            return {}
        
        logger.info(f"Scenario bounds: X=[{bounds.min_x:.1f}, {bounds.max_x:.1f}], "
                   f"Y=[{bounds.min_y:.1f}, {bounds.max_y:.1f}]")
        
        relevant_roads = self._filter_relevant_roads(bounds)
        logger.info(f"Processing {len(relevant_roads)} relevant roads")
        
        map_features = {}
        crosswalk_counter = 0
        
        for road_idx in tqdm(relevant_roads, desc="Processing map sections"):
            road_id, lane_data = self.hd_map_data[road_idx]
            
            # Process crosswalks
            crosswalks = self._extract_crosswalks(lane_data, bounds)
            for crosswalk in crosswalks:
                map_features[f"crosswalk_{crosswalk_counter}"] = crosswalk
                crosswalk_counter += 1
            
            # Process lanes
            lane_features = self._extract_lanes(road_id, lane_data, bounds)
            map_features.update(lane_features)
        
        logger.info(f"Extracted {len(map_features)} map features")
        return map_features
    
    def _extract_crosswalks(self, lane_data: Dict, bounds: ScenarioBounds) -> List[Dict]:
        """Extract crosswalk features"""
        crosswalks = []
        
        trigger_volumes = lane_data.get("Trigger_Volumes", [])
        if not isinstance(trigger_volumes, list):
            return crosswalks
        
        for volume in trigger_volumes:
            if not isinstance(volume, dict):
                continue
            
            tv_type = volume.get("Type", "").lower()
            if tv_type not in ["crosswalk", "pedestrian_crossing", "zebra_crossing"]:
                continue
            
            points = volume.get("Points", [])
            if not self._feature_in_bounds(points, bounds):
                continue
            
            polygon = self._extract_polygon(points, bounds)
            if len(polygon) >= 3:
                crosswalks.append({
                    "type": "CROSSWALK",
                    "polygon": np.array(polygon, dtype=np.float32)
                })
        
        return crosswalks
    
    def _extract_lanes(self, road_id: int, lane_data: Dict, bounds: ScenarioBounds) -> Dict:
        """Extract lane features"""
        features = {}
        
        for lane_id, lane_info in lane_data.items():
            if lane_id == "Trigger_Volumes" or not isinstance(lane_info, list):
                continue
            
            # Quick bounds check
            if not any(self._feature_in_bounds(elem.get("Points", []), bounds) 
                      for elem in lane_info if isinstance(elem, dict)):
                continue
            
            # Process lane elements
            for elem_idx, element in enumerate(lane_info):
                if not isinstance(element, dict):
                    continue
                
                points = element.get("Points", [])
                if not self._feature_in_bounds(points, bounds):
                    continue
                
                feature = self._create_lane_feature(
                    road_id, lane_id, elem_idx, element, bounds
                )
                if feature:
                    feature_id = f"road_{road_id}_lane_{lane_id}_element_{elem_idx}"
                    features[feature_id] = feature
        
        return features
    
    def _create_lane_feature(self, road_id: int, lane_id: str, elem_idx: int,
                            element: Dict, bounds: ScenarioBounds) -> Optional[Dict]:
        """Create a lane feature from element data"""
        lane_type = element.get("Type", "NONE")
        color = element.get("Color", "White")
        points = element.get("Points", [])
        topology = element.get("Topology", [])  # Extract topology here
        
        polyline = self._extract_polyline(points, bounds)
        if not polyline:
            return None
        
        polyline_array = np.array(polyline, dtype=np.float32)
        
        if lane_type == "Center":
            # Drivable lane
            lane_width = 3.5
            width_array = np.full((len(polyline_array), 2), lane_width / 2.0, dtype=np.float32)
            
            feature = {
                "type": "LANE_SURFACE_STREET",
                "polyline": polyline_array,
                "speed_limit_kmh": 50.0,
                "speed_limit_mph": 31.07,
                "width": width_array,
                "interpolating": False,
                "lane_id": lane_id,
                "road_id": road_id,
                "left_boundaries": [],
                "right_boundaries": [],
                "left_neighbor": [],
                "right_neighbor": [],
                "entry_lanes": [],
                "exit_lanes": []
            }
            
            # Add topology information if available (this was missing!)
            if topology:
                for topo in topology:
                    if isinstance(topo, (list, tuple)) and len(topo) >= 2:
                        connected_road, connected_lane = topo[0], topo[1]
                        feature["entry_lanes"].append(f"road_{connected_road}_lane_{connected_lane}")
            
            return feature
        else:
            # Lane marking
            base_type = self.LANE_TYPE_MAP.get(lane_type, "ROAD_LINE_SOLID_SINGLE")
            color_suffix = "_YELLOW" if color.upper() == "YELLOW" else "_WHITE"
            
            return {
                "type": base_type + color_suffix,
                "polyline": polyline_array
            }
    
    def _extract_polygon(self, points: List, bounds: ScenarioBounds) -> List[List[float]]:
        """Extract polygon from points"""
        polygon = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                location = point[0] if isinstance(point[0], (list, tuple)) else point
                if isinstance(location, (list, tuple)) and len(location) >= 3:
                    x = float(location[0])
                    y = -float(location[1])
                    z = float(location[2])
                    polygon.append([x, y, z])
        return polygon
    
    def _extract_polyline(self, points: List, bounds: ScenarioBounds) -> List[List[float]]:
        """Extract polyline from points, filtering by bounds"""
        polyline = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                location = point[0] if isinstance(point[0], (list, tuple)) else point
                if isinstance(location, (list, tuple)) and len(location) >= 3:
                    x = float(location[0])
                    y = -float(location[1])
                    z = float(location[2])
                    if bounds.contains_point(x, y, buffer=50):
                        polyline.append([x, y, z])
        return polyline
    
    def _calculate_summaries(self, tracks: Dict, dynamic_map_states: Dict, 
                           map_features: Dict) -> Dict[str, Any]:
        """Calculate all summary statistics"""
        # Object summaries
        object_summary = {}
        type_counts = defaultdict(int)
        moving_type_counts = defaultdict(int)
        
        for obj_id, track in tracks.items():
            valid_mask = track["state"]["valid"]
            positions = track["state"]["position"]
            valid_positions = positions[valid_mask]
            
            # Calculate movement
            moving_distance = 0.0
            if len(valid_positions) > 1:
                distances = np.linalg.norm(np.diff(valid_positions, axis=0)[:, :2], axis=1)
                moving_distance = float(np.sum(distances))
            
            obj_type = track["type"]
            type_counts[obj_type] += 1
            if moving_distance > 1.0:
                moving_type_counts[obj_type] += 1
            
            object_summary[obj_id] = {
                "object_id": obj_id,
                "type": obj_type,
                "track_length": self.track_length,
                "valid_length": int(np.sum(valid_mask)),
                "continuous_valid_length": int(np.sum(valid_mask)),
                "moving_distance": moving_distance
            }
        
        # Map summary
        map_summary = {
            "num_map_features": len(map_features),
            "num_lanes": sum(1 for f in map_features.values() 
                           if f.get("type") in ["LANE_SURFACE_STREET", "LANE_FREEWAY", 
                                               "LANE_BIKE_LANE", "LANE_SURFACE_UNSTRUCTURE"]),
            "num_road_lines": sum(1 for f in map_features.values() 
                                if f.get("type", "").startswith("ROAD_LINE")),
            "num_crosswalks": sum(1 for f in map_features.values() 
                                if f.get("type") == "CROSSWALK"),
            "num_stop_signs": sum(1 for s in object_summary.values() 
                                if s["type"] == "STOP_SIGN"),
        }
        
        # Traffic light summary
        traffic_light_summary = {
            "num_traffic_lights": len(dynamic_map_states),
            "num_traffic_light_each_step": {},
            "num_traffic_light_types": ()
        }
        
        if dynamic_map_states:
            state_counts = defaultdict(int)
            all_states = set()
            
            for tl in dynamic_map_states.values():
                for state in tl["state"]["object_state"]:
                    state_counts[state] += 1
                    all_states.add(state)
            
            traffic_light_summary["num_traffic_light_each_step"] = dict(state_counts)
            traffic_light_summary["num_traffic_light_types"] = tuple(sorted(all_states))
        
        # Combined number summary
        number_summary = {
            "num_objects": len(tracks),
            "num_moving_objects": sum(1 for s in object_summary.values() 
                                    if s["moving_distance"] > 1.0),
            "num_objects_each_type": dict(type_counts),
            "num_moving_objects_each_type": dict(moving_type_counts),
            "object_types": tuple(set(type_counts.keys())),
            **map_summary,
            **traffic_light_summary
        }
        
        return {
            "object_summary": object_summary,
            "number_summary": number_summary
        }
    
    def create_scenario_data(self, tracks: Dict, sdc_id: str, 
                           dynamic_map_states: Dict) -> Dict[str, Any]:
        """Create complete scenario data structure"""
        # Extract map features
        map_features = self.extract_map_features()
        
        # Calculate summaries
        summaries = self._calculate_summaries(tracks, dynamic_map_states, map_features)
        
        # Create timestamps
        timestamps = [i * 0.05 for i in range(self.track_length)]
        
        return {
            # Top-level fields
            "id": self.scenario_id,
            "version": "1.0",
            "length": self.track_length,
            "tracks": tracks,
            "dynamic_map_states": dynamic_map_states,
            "map_features": map_features,
            
            # Metadata
            "metadata": {
                "id": self.scenario_id,
                "coordinate": "bench2drive",
                "ts": timestamps,
                "metadrive_processed": False,
                "sdc_id": sdc_id,
                "dataset": "bench2drive",
                "scenario_id": self.scenario_id,
                "source_file": str(self.scenario_path),
                "track_length": self.track_length,
                "current_time_index": 0,
                "sdc_track_index": 0,
                "objects_of_interest": [],
                "tracks_to_predict": {},
                **summaries
            }
        }
    
    def convert_scenario(self) -> Tuple[Dict[str, Any], str]:
        """Main conversion method"""
        logger.info(f"Converting scenario: {self.scenario_name}")
        
        # Extract tracks
        tracks, sdc_id, dynamic_map_states = self.extract_tracks()
        logger.info(f"Extracted {len(tracks)} objects, {len(dynamic_map_states)} traffic lights, SDC: {sdc_id}")
        
        # Create complete scenario data
        scenario_data = self.create_scenario_data(tracks, sdc_id, dynamic_map_states)
        
        # Save scenario first
        scenario_file_path = self.save_scenario(scenario_data)
        
        # Create dataset files
        DatasetManager.create_dataset_files(
            [scenario_file_path], 
            str(self.output_dir)
        )
        
        # Generate sensor data if requested (using the saved scenario file)
        if self.generate_sensors:
            self.generate_sensor_data(scenario_file_path)
        
        return scenario_data, scenario_file_path
    
    def save_scenario(self, scenario_data: Dict[str, Any]) -> str:
        """Save scenario to pickle file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"sd_bench2drive_{self.scenario_id}.pkl"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'wb') as f:
            pickle.dump(scenario_data, f)
        
        logger.info(f"Saved scenario to: {output_path}")
        return str(output_path)


class DatasetManager:
    """Manages dataset file creation and organisation"""
    
    @staticmethod
    def create_dataset_files(scenario_files: List[str], output_dir: str):
        """Create dataset_mapping.pkl and dataset_summary.pkl files"""
        output_dir = Path(output_dir)
        
        # Create subdirectory for scenarios
        scenario_subdir = output_dir / f"{output_dir.name}_0"
        scenario_subdir.mkdir(parents=True, exist_ok=True)
        
        # Move files and create mapping
        dataset_mapping = {}
        dataset_summary = {}
        
        for scenario_file in scenario_files:
            scenario_path = Path(scenario_file)
            scenario_filename = scenario_path.name
            
            # Move to subdirectory if needed
            new_path = scenario_subdir / scenario_filename
            if scenario_path.exists() and scenario_path != new_path:
                scenario_path.rename(new_path)
                scenario_path = new_path
            
            dataset_mapping[scenario_filename] = scenario_subdir.name
            
            # Load and summarise scenario
            with open(scenario_path, 'rb') as f:
                scenario_data = pickle.load(f)
            
            # Extract summary
            metadata = scenario_data["metadata"]
            dataset_summary[scenario_filename] = {
                "coordinate": metadata["coordinate"],
                "current_time_index": metadata["current_time_index"],
                "dataset": metadata["dataset"],
                "id": metadata["id"],
                "metadrive_processed": metadata["metadrive_processed"],
                "number_summary": metadata["number_summary"],
                "object_summary": metadata["object_summary"],
                "objects_of_interest": metadata["objects_of_interest"],
                "scenario_id": metadata["scenario_id"],
                "sdc_id": metadata["sdc_id"],
                "sdc_track_index": metadata["sdc_track_index"],
                "source_file": metadata["source_file"],
                "track_length": metadata["track_length"],
                "tracks_to_predict": metadata["tracks_to_predict"],
                "ts": metadata["ts"]
            }
        
        # Save dataset files
        with open(output_dir / "dataset_mapping.pkl", 'wb') as f:
            pickle.dump(dataset_mapping, f)
        
        with open(output_dir / "dataset_summary.pkl", 'wb') as f:
            pickle.dump(dataset_summary, f)
        
        logger.info(f"Created dataset files in: {output_dir}")
        logger.info(f"Scenarios moved to: {scenario_subdir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Convert Bench2Drive scenario to MetaDrive format"
    )
    parser.add_argument(
        "scenario_path", 
        help="Path to Bench2Drive scenario .tar.gz file or directory"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        default=None,
        help="Output directory for converted files (auto-derived if not specified)"
    )
    parser.add_argument(
        "--hd-map", 
        required=True,
        help="Path to HD map .npz file (e.g., Town01_HD_map.npz)"
    )
    # Set a default path for the parked vehicles file relative to this script
    script_dir = Path(__file__).parent.resolve()
    default_parked_vehicles_path = script_dir.parent / "Bench2Drive/leaderboard/leaderboard/utils/parked_vehicles.py"

    parser.add_argument(
        "--parked-vehicles-path",
        default=None,
        help="Path to the python file containing parked vehicle data. Defaults to the one in the Bench2Drive repo."
    )
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Skip sensor data generation (only convert scenario data)"
    )
    parser.add_argument(
        "--no-raster",
        action="store_true",
        help="Disable 3D rasterized camera rendering (CAM_F0/L0/L1/L2/R0/R1/R2/B0)"
    )
    
    args = parser.parse_args()
    
    converter = None
    try:
        # Convert scenario
        converter = Bench2DriveConverter(
            args.scenario_path, 
            args.hd_map,
            args.output_dir,
            generate_sensors=not args.no_sensors,
            generate_rasters=not args.no_raster,
            parked_vehicles_path=args.parked_vehicles_path
        )
        scenario_data, scenario_file = converter.convert_scenario()
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise
    finally:
        # Ensure cleanup
        if converter:
            converter._cleanup()


if __name__ == "__main__":
    main()

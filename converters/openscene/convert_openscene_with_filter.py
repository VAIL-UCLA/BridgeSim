import argparse
import pickle
import os
import logging
import numpy as np
import copy
import cv2
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

import yaml

# MetaDrive Imports
try:
    from metadrive.type import MetaDriveType
    from metadrive.scenario import ScenarioDescription as SD
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.component.sensors.rgb_camera import RGBCamera
except ImportError:
    raise ImportError("Please install metadrive to use this converter.")

# NuPlan Imports
try:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString, Point
    from shapely.ops import unary_union
    from shapely.geometry import Point as Point2D
except ImportError as e:
    logging.warning(f"NuPlan devkit import failed: {e}. Map extraction will fail.")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Scene Filter Configuration (NavSim-compatible)
# ============================================================================

@dataclass
class SceneFilterConfig:
    """Configuration for filtering OpenScene logs to extract specific scenes."""
    num_history_frames: int
    num_future_frames: int
    frame_interval: Optional[int]
    has_route: bool
    log_names: Optional[List[str]]
    tokens: Optional[List[str]]
    # max_scenes: Optional[int] = None  # Commented out: not used in workflow

    @property
    def num_frames(self) -> int:
        """Total number of frames in a scene window."""
        return self.num_history_frames + self.num_future_frames


def load_scene_filter_config(yaml_path: str) -> SceneFilterConfig:
    """
    Load scene filter configuration from YAML file.

    Args:
        yaml_path: Path to scene_filter YAML file (e.g., scene_filter/navtest.yaml)

    Returns:
        SceneFilterConfig instance
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ignore Hydra-specific fields (_target_, _convert_)
    return SceneFilterConfig(
        num_history_frames=config['num_history_frames'],
        num_future_frames=config['num_future_frames'],
        frame_interval=config.get('frame_interval'),
        has_route=config.get('has_route', True),
        log_names=config.get('log_names'),
        tokens=config.get('tokens'),
        # max_scenes=config.get('max_scenes')  # Commented out: not used in workflow
    )


def filter_log_files(input_dir: Path, log_names: Optional[List[str]]) -> List[Path]:
    """
    Filter log files by log_names from scene filter config.

    Args:
        input_dir: Directory containing OpenScene .pkl log files
        log_names: List of log names to include (None = include all)

    Returns:
        List of Path objects for matching log files
    """
    all_pkl_files = sorted(input_dir.glob("*.pkl"))

    if log_names is None:
        return all_pkl_files

    log_names_set = set(log_names)
    filtered_files = [f for f in all_pkl_files if f.stem in log_names_set]

    logger.info(f"Filtered {len(filtered_files)} / {len(all_pkl_files)} log files by log_names")
    return filtered_files


def extract_scene_windows(frames: List[Dict], config: SceneFilterConfig, num_future_frames_override: Optional[int] = None) -> List[tuple]:
    """
    Extract scene windows from log frames using sliding window (NavSim logic).

    Args:
        frames: List of frame dictionaries from one log file
        config: Scene filter configuration
        num_future_frames_override: Optional override for num_future_frames to extract larger windows
                                    while keeping the same frame_interval for sampling

    Returns:
        List of (token, frame_window) tuples
    """
    # Use override for window size if provided, otherwise use config value
    num_future_frames_for_extraction = num_future_frames_override if num_future_frames_override is not None else config.num_future_frames
    num_frames = config.num_history_frames + num_future_frames_for_extraction

    # frame_interval always uses the original num_future_frames from YAML for consistent sampling
    frame_interval = config.frame_interval if config.frame_interval else config.num_frames

    windows = []
    for i in range(0, len(frames), frame_interval):
        window = frames[i : i + num_frames]

        # Skip if window too short
        if len(window) < num_frames:
            continue

        # Current frame is at index num_history_frames - 1 (this stays the same regardless of override)
        current_frame_idx = config.num_history_frames - 1
        current_frame = window[current_frame_idx]

        # Check has_route filter
        if config.has_route:
            roadblock_ids = current_frame.get("roadblock_ids", [])
            if not roadblock_ids or len(roadblock_ids) == 0:
                continue

        # Current frame token (identifies the scene)
        token = current_frame["token"]
        windows.append((token, window))

    return windows


def filter_by_token(windows: List[tuple], tokens: Optional[List[str]]) -> List[tuple]:
    """
    Filter scene windows by token list.

    Args:
        windows: List of (token, frame_window) tuples
        tokens: List of tokens to include (None = include all)

    Returns:
        Filtered list of (token, frame_window) tuples
    """
    if tokens is None:
        return windows

    tokens_set = set(tokens)
    filtered = [(token, window) for token, window in windows if token in tokens_set]

    logger.info(f"Filtered {len(filtered)} / {len(windows)} scene windows by tokens")
    return filtered

# --- Camera Parameters (Source: renderer.py) ---
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

# --- Camera Conversion Logic ---

def rotation_matrix_to_euler_angles(R_mat):
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
    fx = K[0, 0]
    fy = K[1, 1]
    fov_horizontal = 2 * np.arctan(image_width / (2 * fx))
    fov_vertical = 2 * np.arctan(image_height / (2 * fy))
    return np.degrees(fov_horizontal), np.degrees(fov_vertical)

def get_metadrive_cam_params(cam_params, image_width=1920, image_height=1120):
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

# --- Dataset Manager ---

class DatasetManager:
    @staticmethod
    def create_dataset_files(scenario_file_path: str, output_dir: str):
        output_dir = os.path.dirname(output_dir)
        scenario_filename = os.path.basename(scenario_file_path)
        
        dataset_mapping = {scenario_filename: Path(scenario_file_path).parent.name}
        dataset_summary = {}
        
        try:
            with open(scenario_file_path, 'rb') as f:
                scenario_data = pickle.load(f)
            
            metadata = scenario_data.get("metadata", {})
            dataset_summary[scenario_filename] = {
                "id": metadata.get("scenario_id", "unknown"),
                "scenario_id": metadata.get("scenario_id", "unknown"),
                "sample_rate": metadata.get("sample_rate", 0.1),
                "ts": metadata.get("timestep", []),
                "length": metadata.get("track_length", 0),
                "coordinate": metadata.get("coordinate", "right-handed"),
                "sdc_id": metadata.get("sdc_id", "ego"),
                "dataset": metadata.get("dataset", "openscene"),
                "map_features": list(scenario_data.get("map_features", {}).keys()), 
                "number_summary": metadata.get("number_summary", {}),
                "object_summary": metadata.get("object_summary", {}),
            }
        except Exception as e:
            logger.error(f"Failed to read scenario for summary generation: {e}")
            import traceback
            traceback.print_exc()
            return

        with open(os.path.join(output_dir, "dataset_mapping.pkl"), 'wb') as f:
            pickle.dump(dataset_mapping, f)
        
        with open(os.path.join(output_dir, "dataset_summary.pkl"), 'wb') as f:
            pickle.dump(dataset_summary, f)
            
        logger.info(f"Generated dataset summary and mapping in {output_dir}")

# --- Helper Functions ---

def nuplan_to_metadrive_vector(vector, nuplan_center):
    if isinstance(vector, (list, tuple)):
        vector = np.array(vector)
    return vector - np.array(nuplan_center)

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_yaw(w, x, y, z):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

def get_transformation_matrix(translation, rotation):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    rot_array = np.array(rotation)
    if rot_array.shape == (3, 3):
        matrix[:3, :3] = rot_array
    elif rot_array.flatten().shape[0] == 4:
        w, x, y, z = rot_array.flatten()
        r = R.from_quat([x, y, z, w])
        matrix[:3, :3] = r.as_matrix()
    else:
        raise ValueError(f"Unknown rotation format: {rot_array.shape}")
    return matrix

def set_light_position(map_api, lane_id, center_3d, target_position=8):
    try:
        lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
        if lane is None: return None
        path = lane.baseline_path.discrete_path
        if not path: return None

        acc_length = 0
        point = [path[0].x, path[0].y]
        for k, p_curr in enumerate(path[1:], start=1):
            p_prev = path[k - 1]
            acc_length += np.linalg.norm([p_curr.x - p_prev.x, p_curr.y - p_prev.y])
            point = [p_curr.x, p_curr.y]
            if acc_length > target_position: break
        
        return np.array([point[0] - center_3d[0], point[1] - center_3d[1], 0])
    except Exception:
        return None

def interpolate_tracks(tracks, original_len, expansion_factor=5):
    """
    Interpolates track data from 2Hz to 10Hz (factor=5).
    """
    new_len = (original_len - 1) * expansion_factor + 1
    logger.info(f"Interpolating data: {original_len} steps -> {new_len} steps")

    # Time indices for interpolation
    # Original: 0, 1, 2...
    # Target: 0, 0.2, 0.4... (normalized to indices)
    old_indices = np.arange(original_len)
    new_indices = np.linspace(0, original_len - 1, new_len)

    for track_id, track_data in tracks.items():
        state = track_data["state"]
        valid_mask = state["valid"] == 1
        valid_indices = np.flatnonzero(valid_mask)

        # If nothing is valid, keep zeros and skip expensive interpolation work
        if valid_indices.size == 0:
            state["position"] = np.zeros((new_len, 3))
            state["velocity"] = np.zeros((new_len, 2))
            state["heading"] = np.zeros((new_len,))
            for key in ["length", "width", "height", "valid"]:
                state[key] = np.zeros((new_len,))
            track_data["metadata"]["track_length"] = new_len
            continue

        linear_or_nearest = "linear" if valid_indices.size > 1 else "nearest"
        # Interpolate validity so we can zero out invalid slots after interpolation
        f_valid = interp1d(old_indices, state["valid"], kind="nearest", bounds_error=False, fill_value=0)
        new_valid = f_valid(new_indices)

        # 1. Linear Interpolation for Position (Nx3) and Velocity (Nx2)
        # We assume position/velocity changes smoothly.
        for key in ["position", "velocity"]:
            if key in state and state[key].ndim > 1:
                f = interp1d(valid_indices, state[key][valid_indices], axis=0, kind=linear_or_nearest, bounds_error=False, fill_value="extrapolate")
                interpolated = f(new_indices)
                state[key] = np.where(new_valid[:, None] > 0.5, interpolated, 0)

        # 2. Angular Interpolation for Heading
        # We must unwrap phases so we don't interpolate across the -pi/pi boundary wrongly
        heading_unwrapped = np.unwrap(state["heading"][valid_indices])
        f_head = interp1d(valid_indices, heading_unwrapped, kind=linear_or_nearest, bounds_error=False, fill_value="extrapolate")
        heading_interp = normalize_angle(f_head(new_indices))
        state["heading"] = np.where(new_valid > 0.5, heading_interp, 0)

        # 3. Nearest/Repeat for Dimensions and Valid flags
        # If it was valid at frame i and i+1, linear interp will keep it >0. 
        # We use 'nearest' or 'previous' usually, but linear is safe for dimensions if they don't change.
        # For 'valid', we strictly want to keep it valid if it's bridging a gap.
        for key in ["length", "width", "height"]:
            f_dim = interp1d(valid_indices, state[key][valid_indices], kind='nearest', bounds_error=False, fill_value=0)
            vals = f_dim(new_indices)
            state[key] = np.where(new_valid > 0.5, vals, 0)

        state["valid"] = new_valid

        # Update metadata length
        track_data["metadata"]["track_length"] = new_len

    return tracks, new_len

def interpolate_traffic_lights(dynamic_map_states, original_len, expansion_factor=5):
    """
    Interpolates traffic light states using nearest-neighbor/zero-order hold
    via NumPy indexing (works for strings).
    """
    new_len = (original_len - 1) * expansion_factor + 1
    
    # Instead of math interpolation, we generate the indices we want to 'copy' from.
    # np.linspace gives us 0.0, 0.2, 0.4...
    # np.floor turns that into 0, 0, 0... (holding the previous frame's state)
    new_indices = np.linspace(0, original_len - 1, new_len)
    indices_to_sample = np.floor(new_indices).astype(int)
    
    # Safety clamp to ensure we don't go out of bounds due to float precision
    indices_to_sample = np.clip(indices_to_sample, 0, original_len - 1)

    for lane_id, data in dynamic_map_states.items():
        # Convert list of strings to a NumPy array
        states = np.array(data["state"]["object_state"])
        
        # Use integer array indexing to create the stretched array
        new_states = states[indices_to_sample]
        
        # Assign back
        data["state"]["object_state"] = new_states.tolist()
        data["metadata"]["track_length"] = new_len
        
    return dynamic_map_states

# --- Mappings ---

OBJECT_TYPE_MAP = {
    "vehicle": MetaDriveType.VEHICLE,
    "bicycle": MetaDriveType.CYCLIST,
    "pedestrian": MetaDriveType.PEDESTRIAN,
    "generic_object": MetaDriveType.TRAFFIC_CONE,
    "traffic_cone": MetaDriveType.TRAFFIC_CONE,
    "barrier": MetaDriveType.TRAFFIC_BARRIER,
    "czone_sign": MetaDriveType.TRAFFIC_CONE,
}

def get_metadrive_type(nuplan_label: str) -> str:
    return OBJECT_TYPE_MAP.get(nuplan_label, MetaDriveType.UNSET)

# --- Map Extraction Logic ---

def extract_centerline(map_obj, nuplan_center):
    path = map_obj.baseline_path.discrete_path
    points = np.array([[pose.x, pose.y] for pose in path])
    return nuplan_to_metadrive_vector(points, nuplan_center)

def get_points_from_boundary(boundary, center):
    path = boundary.discrete_path
    points = np.array([[pose.x, pose.y] for pose in path])
    return nuplan_to_metadrive_vector(points, center)

def calculate_lane_width(lane, centerline_global_points):
    """
    Calculates the width (left and right distance from center) for each point on the centerline
    by projecting to the global boundary lines.
    """
    try:
        left_bound = lane.left_boundary.discrete_path
        right_bound = lane.right_boundary.discrete_path
        
        if not left_bound or not right_bound:
            raise ValueError("Missing boundary points")
            
        l_line = LineString([(p.x, p.y) for p in left_bound])
        r_line = LineString([(p.x, p.y) for p in right_bound])
        
        widths = []
        for pt in centerline_global_points:
            p = Point(pt[0], pt[1])
            l_dist = l_line.distance(p)
            r_dist = r_line.distance(p)
            widths.append([l_dist, r_dist])
            
        return np.array(widths, dtype=np.float32)
        
    except Exception:
        return np.full((len(centerline_global_points), 2), 1.75, dtype=np.float32)

def extract_map_features(map_api, center, radius, anchor):
    """
    Extracts map features using a SINGLE MASSIVE RADIUS around 'center'.
    Normalizes coordinates relative to 'anchor'.
    """
    ret = {}
    np.seterr(all='ignore')
    
    # Query with the massive radius
    center_for_query = Point2D(*center)
    layer_names = [
        SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.LANE, SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION, SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]
    
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)

    if SemanticMapLayer.STOP_LINE in nearest_vector_map:
        stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
        nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
            s for s in stop_polygons if s.stop_line_type != StopLineType.TURN_STOP
        ]

    block_polygons = []
    
    for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for block in nearest_vector_map[layer]:
            edges = sorted(block.interior_edges, key=lambda lane: lane.index) \
                if layer == SemanticMapLayer.ROADBLOCK else block.interior_edges
            for index, lane_meta_data in enumerate(edges):
                if not hasattr(lane_meta_data, "baseline_path"): continue
                
                # Polygon
                if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
                    boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
                    sizes = [len(p.xy[1]) for p in boundary[0]]
                    points = boundary[0][np.argmax(sizes)].xy
                elif isinstance(lane_meta_data.polygon.boundary, LineString):
                    points = lane_meta_data.polygon.boundary.xy
                
                polygon_points = np.array(list(zip(points[0], points[1])))
                polygon = nuplan_to_metadrive_vector(polygon_points, anchor)
                
                # Lane Width & Polyline
                raw_centerline = np.array([[p.x, p.y] for p in lane_meta_data.baseline_path.discrete_path])
                width_array = calculate_lane_width(lane_meta_data, raw_centerline)
                width_array = np.clip(width_array, 0.1, 10.0)
                polyline = nuplan_to_metadrive_vector(raw_centerline, anchor)
                
                ret[lane_meta_data.id] = {
                    SD.TYPE: MetaDriveType.LANE_SURFACE_STREET,
                    SD.POLYLINE: polyline,
                    "width": width_array,
                    "roadblock_id": block.id,
                    SD.ENTRY: [edge.id for edge in lane_meta_data.incoming_edges],
                    SD.EXIT: [edge.id for edge in lane_meta_data.outgoing_edges],
                    SD.LEFT_NEIGHBORS: [edge.id for edge in block.interior_edges[:index]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    SD.RIGHT_NEIGHBORS: [edge.id for edge in block.interior_edges[index + 1:]] \
                        if layer == SemanticMapLayer.ROADBLOCK else [],
                    SD.POLYGON: polygon
                }
                
                if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR: continue
                left = lane_meta_data.left_boundary
                if left.id not in ret:
                    ret[left.id] = {
                        SD.TYPE: MetaDriveType.LINE_BROKEN_SINGLE_WHITE, 
                        SD.POLYLINE: get_points_from_boundary(left, anchor)
                    }

            if layer == SemanticMapLayer.ROADBLOCK:
                block_polygons.append(block.polygon)

    for layer_type, md_type in [(SemanticMapLayer.WALKWAYS, MetaDriveType.BOUNDARY_SIDEWALK),
                                (SemanticMapLayer.CROSSWALK, MetaDriveType.CROSSWALK)]:
        for area in nearest_vector_map[layer_type]:
            if isinstance(area.polygon.exterior, MultiLineString):
                boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
                sizes = [len(p.xy[1]) for p in boundary[0]]
                points = boundary[0][np.argmax(sizes)].xy
            elif isinstance(area.polygon.exterior, LineString):
                points = area.polygon.exterior.xy
            polygon_points = np.array(list(zip(points[0], points[1])))
            polygon = nuplan_to_metadrive_vector(polygon_points, anchor)
            ret[area.id] = {SD.TYPE: md_type, SD.POLYGON: polygon}

    interpolygons = [b.polygon for b in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(interpolygons + block_polygons)).boundary.explode(index_parts=True)
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        block_points = nuplan_to_metadrive_vector(block_points, anchor)
        ret[f"boundary_{idx}"] = {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE, SD.POLYLINE: block_points}

    np.seterr(all='warn')
    return ret

# --- Image Generation Logic ---

def generate_images(scenario_path, original_frames, output_base_dir, max_frames=None, interpolate=False):
    logger.info("Starting Image Generation...")
    
    CAM_IDS = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']
    
    # 1. Pre-calculate MetaDrive configs for all 8 views
    md_cam_configs = {}
    for cam_id in CAM_IDS:
        if cam_id in CAMERA_PARAMS:
            md_cam_configs[cam_id] = get_metadrive_cam_params(CAMERA_PARAMS[cam_id], 1920, 1120)
    
    sensors = {"rgb_camera": (RGBCamera, 1920, 1120)}
    data_dir = os.path.dirname(os.path.abspath(scenario_path))
    
    # --- UPDATED HORIZON LOGIC ---
    if interpolate:
        expansion_factor = 5
        # Total possible steps in the converted (interpolated) file
        total_interpolated_len = (len(original_frames) - 1) * expansion_factor + 1
        
        if max_frames is not None and max_frames > 0:
            # User input '20' means 20 *original* frames, which equals 100 interpolated steps.
            requested_steps = max_frames * expansion_factor
            # Clamp to make sure we don't ask for more than exists in the file
            render_horizon = min(requested_steps, total_interpolated_len)
        else:
            render_horizon = total_interpolated_len
    else:
        # Standard logic: max_frames is directly the number of steps
        render_horizon = min(len(original_frames), max_frames) if (max_frames is not None and max_frames > 0) else len(original_frames)

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
        
        # Create output directories
        for cam_id in CAM_IDS:
            os.makedirs(os.path.join(output_base_dir, "camera", cam_id), exist_ok=True)
            
        logger.info(f"Rendering {render_horizon} steps (Interpolation: {interpolate})")

        for current_step in tqdm(range(render_horizon), desc="Rendering Frames"):
            # Step the environment
            env.step([0, 0])
            
            # --- TOKEN MAPPING LOGIC ---
            if interpolate:
                # Map current 10Hz step back to original 2Hz frame index
                real_frame_idx = current_step // 5
                sub_step_idx = current_step % 5
                
                if real_frame_idx >= len(original_frames):
                    logger.warning(f"Step {current_step} maps to OOB frame {real_frame_idx}. Breaking.")
                    break
                    
                frame_data = original_frames[real_frame_idx]
                is_interpolated_frame = (sub_step_idx != 0)
                suffix = f"_{sub_step_idx - 1}" if is_interpolated_frame else ""
            else:
                frame_data = original_frames[current_step]
                is_interpolated_frame = False
                suffix = ""

            frame_token = frame_data.get("token")
            
            # Get the single sensor instance
            sensor = env.engine.get_sensor("rgb_camera")

            # Iterate through views
            for cam_id in CAM_IDS:
                if cam_id not in md_cam_configs: continue
                cfg = md_cam_configs[cam_id]
                
                if hasattr(sensor.lens, 'setFov'):
                    sensor.lens.setFov(70)
                
                img = sensor.perceive(
                    to_float=False,
                    new_parent_node=ego.origin,
                    position=cfg['pos'],
                    hpr=cfg['hpr']
                ).get()
                
                # Name Resolution
                image_token = None
                
                if "cams" in frame_data and cam_id in frame_data["cams"]:
                    cam_info = frame_data["cams"][cam_id]
                    if "sample_data_token" in cam_info:
                        image_token = cam_info["sample_data_token"]
                    elif "token" in cam_info:
                        image_token = cam_info["token"]
                    
                    if not image_token and "data_path" in cam_info:
                        filename = os.path.basename(cam_info["data_path"])
                        image_token = os.path.splitext(filename)[0]
                
                if not image_token:
                    image_token = frame_token if frame_token else f"{real_frame_idx:05d}" if interpolate else f"{current_step:05d}"

                final_filename = f"{image_token}{suffix}.jpg"
                save_path = os.path.join(output_base_dir, "camera", cam_id, final_filename)
                cv2.imwrite(save_path, img)
                
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

# --- Main Conversion Logic ---

def transform_local_boxes_to_global(local_boxes, ego_pos, ego_rot):
    if len(local_boxes) == 0: return local_boxes
    ego_yaw = quaternion_to_yaw(*ego_rot)
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]])
    local_pos = local_boxes[:, :2]
    global_pos = (R @ local_pos.T).T + ego_pos[:2]
    global_z = local_boxes[:, 2] + ego_pos[2]
    global_yaw = normalize_angle(local_boxes[:, 6] + ego_yaw)
    global_boxes = copy.deepcopy(local_boxes)
    global_boxes[:, 0:2] = global_pos
    global_boxes[:, 2] = global_z
    global_boxes[:, 6] = global_yaw
    return global_boxes

def get_trajectory_bounds(frames):
    """
    Calculates a generous bounding radius covering the entire ego trajectory.
    """
    all_pos = []
    for frame in frames:
        pos = frame["ego2global_translation"][:2] 
        all_pos.append(pos)
    
    all_pos = np.array(all_pos)
    min_x, min_y = np.min(all_pos, axis=0)
    max_x, max_y = np.max(all_pos, axis=0)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    corners = np.array([
        [min_x, min_y], [min_x, max_y],
        [max_x, min_y], [max_x, max_y]
    ])
    distances = np.linalg.norm(corners - np.array([center_x, center_y]), axis=1)
    
    # 500m Buffer added to the max distance
    radius = np.max(distances) + 100 
    
    return [center_x, center_y], radius

def convert_scenario_from_frames(
    frames: List[Dict],
    scene_token: str,
    output_path: str,
    map_root: str,
    render: bool = False,
    max_frames: int = None,
    interpolate: bool = False,
    camera_output_dir: str = None
):
    """
    Convert a scene window (list of frames) to MetaDrive format.

    Args:
        frames: List of frame dictionaries (scene window)
        scene_token: Token identifying this scene
        output_path: Path to save MetaDrive .pkl file
        map_root: Root directory of nuplan maps
        render: Whether to generate rendered images
        max_frames: Limit number of frames to render
        interpolate: Whether to interpolate 2Hz to 10Hz
        camera_output_dir: Optional separate directory for camera images and dataset metadata
    """
    if not frames:
        logger.error("Empty frames list.")
        return

    first_frame = frames[0]
    scenario_id = scene_token  # Use scene token as scenario ID
    map_location = first_frame.get("map_location")
    
    ego_start_pos_global = first_frame["ego2global_translation"]
    scenario_center_3d = np.array(ego_start_pos_global)
    
    # MEGA-BUFFER Calculation
    traj_center, traj_radius = get_trajectory_bounds(frames)
    
    logger.info(f"Processing Scenario: {scenario_id} | Map: {map_location}")
    logger.info(f"Scenario Anchor: {scenario_center_3d[:2]}")
    logger.info(f"Map Extraction: Center={traj_center}, Radius={traj_radius:.2f}m (Includes 500m buffer)")

    logger.info(f"Initializing NuPlan Map API with root: {map_root}")
    try:
        map_api = get_maps_api(map_root, "nuplan-maps-v1.0", map_location)
        map_features = extract_map_features(map_api, center=traj_center, radius=traj_radius, anchor=scenario_center_3d[:2])
        logger.info(f"Extracted {len(map_features)} map features.")
    except Exception as e:
        logger.error(f"Failed to extract map features: {e}")
        return

    if len(map_features) == 0:
        logger.warning("WARNING: Map feature dict is empty! Scenario will look empty.")

    track_ids = set()
    track_types = {}
    track_ids.add("ego")
    track_types["ego"] = MetaDriveType.VEHICLE

    for frame in frames:
        anns = frame.get("anns", {})
        if not anns: continue
        for t_token, t_name in zip(anns["track_tokens"], anns["gt_names"]):
            track_ids.add(t_token)
            if t_token not in track_types:
                track_types[t_token] = get_metadrive_type(t_name)

    episode_len = len(frames)
    tracks = {
        tid: dict(
            type=track_types.get(tid, MetaDriveType.UNSET),
            state=dict(
                position=np.zeros((episode_len, 3)),
                heading=np.zeros((episode_len,)),
                velocity=np.zeros((episode_len, 2)),
                valid=np.zeros((episode_len,)),
                length=np.zeros((episode_len,)),
                width=np.zeros((episode_len,)),
                height=np.zeros((episode_len,))
            ),
            metadata=dict(track_length=episode_len, type=track_types.get(tid, MetaDriveType.UNSET), object_id=tid)
        )
        for tid in track_ids
    }

    dynamic_map_states = {}
    
    for frame_idx, frame in enumerate(tqdm(frames, desc="Converting Frames")):
        ego_pos_global = np.array(frame["ego2global_translation"])
        ego_pos_local = ego_pos_global - scenario_center_3d
        
        ego_rot = frame["ego2global_rotation"] 
        ego_yaw = quaternion_to_yaw(*ego_rot)

        tracks["ego"]["state"]["position"][frame_idx] = ego_pos_local
        tracks["ego"]["state"]["heading"][frame_idx] = ego_yaw
        tracks["ego"]["state"]["velocity"][frame_idx] = frame["ego_dynamic_state"][:2]
        tracks["ego"]["state"]["valid"][frame_idx] = 1
        tracks["ego"]["state"]["length"][frame_idx] = 4.0
        tracks["ego"]["state"]["width"][frame_idx] = 1.8
        tracks["ego"]["state"]["height"][frame_idx] = 1.5

        anns = frame.get("anns", {})
        if anns:
            track_tokens = anns["track_tokens"]
            velocities = anns["gt_velocity_3d"]
            
            if "gt_boxes_world" in anns:
                gt_boxes = anns["gt_boxes_world"]
            elif "gt_boxes" in anns:
                gt_boxes = transform_local_boxes_to_global(anns["gt_boxes"], ego_pos_global, ego_rot)
            else:
                gt_boxes = []

            if len(gt_boxes) > 0:
                for i, token in enumerate(track_tokens):
                    box_global = gt_boxes[i] 
                    vel = velocities[i] 
                    pos_local = box_global[:3] - scenario_center_3d
                    
                    tracks[token]["state"]["position"][frame_idx] = pos_local
                    tracks[token]["state"]["length"][frame_idx] = box_global[3]
                    tracks[token]["state"]["width"][frame_idx] = box_global[4]
                    tracks[token]["state"]["height"][frame_idx] = box_global[5]
                    tracks[token]["state"]["heading"][frame_idx] = box_global[6]
                    tracks[token]["state"]["velocity"][frame_idx] = vel[:2]
                    tracks[token]["state"]["valid"][frame_idx] = 1

        if "traffic_lights" in frame:
            for tl in frame["traffic_lights"]:
                lane_id = str(tl[0])
                is_red = tl[1]
                
                if lane_id not in dynamic_map_states:
                    pos = None
                    if len(tl) > 2: pos = None 
                    if pos is None and map_api is not None:
                        pos = set_light_position(map_api, lane_id, scenario_center_3d)
                    if pos is None: pos = np.array([0, 0, 0])

                    dynamic_map_states[lane_id] = {
                        "type": "TRAFFIC_LIGHT",
                        "state": {"object_state": [MetaDriveType.LANE_STATE_UNKNOWN] * episode_len},
                        "stop_point": pos,
                        "lane": lane_id,
                        "metadata": {"track_length": episode_len, "type": "TRAFFIC_LIGHT"}
                    }
                state = MetaDriveType.LANE_STATE_STOP if is_red else MetaDriveType.LANE_STATE_GO
                dynamic_map_states[lane_id]["state"]["object_state"][frame_idx] = state

    # --- INTERPOLATION LOGIC ---
    if interpolate:
        logger.info("Interpolation enabled. Up-sampling data to 10Hz...")
        tracks, new_episode_len = interpolate_tracks(tracks, episode_len)
        dynamic_map_states = interpolate_traffic_lights(dynamic_map_states, episode_len)
        final_episode_len = new_episode_len
    else:
        final_episode_len = episode_len

    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = "openscene_converted"
    scenario[SD.LENGTH] = final_episode_len
    scenario[SD.TRACKS] = tracks
    scenario[SD.MAP_FEATURES] = map_features
    scenario[SD.DYNAMIC_MAP_STATES] = dynamic_map_states
    
    scenario[SD.METADATA] = {}
    scenario[SD.METADATA]["dataset"] = "openscene"
    scenario[SD.METADATA]["map"] = map_location
    scenario[SD.METADATA]["sdc_id"] = "ego"
    scenario[SD.METADATA]["scenario_id"] = scenario_id
    scenario[SD.METADATA]["coordinate"] = "right-handed"
    scenario[SD.METADATA]["timestep"] = np.arange(0, final_episode_len * 0.1, 0.1)
    scenario[SD.METADATA]["number_summary"] = {"num_objects": len(tracks)}
    scenario[SD.METADATA]["object_summary"] = {}
    scenario[SD.METADATA]["log_name"] = first_frame.get("log_name", "unknown_log")
    raw_roadblock_ids = [frame.get("roadblock_ids", []) for frame in frames]
    if interpolate:
        # interpolate_tracks produces (original_len - 1) * 5 + 1 steps
        # Use nearest-neighbor: step i maps to original frame round(i / 5)
        expansion_factor = 5
        new_len = (len(raw_roadblock_ids) - 1) * expansion_factor + 1
        interpolated_roadblock_ids = [
            raw_roadblock_ids[min(round(i / expansion_factor), len(raw_roadblock_ids) - 1)]
            for i in range(new_len)
        ]
        scenario[SD.METADATA]["roadblock_ids_per_frame"] = interpolated_roadblock_ids
    else:
        scenario[SD.METADATA]["roadblock_ids_per_frame"] = raw_roadblock_ids

    # Store frame info for replay script (frame tokens and camera tokens)
    frame_info = []
    for frame in frames:
        frame_entry = {
            "token": frame.get("token"),
            "timestamp": frame.get("timestamp"),
            "cams": {}
        }
        if "cams" in frame:
            for cam_id, cam_data in frame["cams"].items():
                frame_entry["cams"][cam_id] = {
                    "sample_data_token": cam_data.get("sample_data_token",
                                         cam_data.get("token",
                                         os.path.splitext(os.path.basename(cam_data.get("data_path", "")))[0])),
                    "data_path": cam_data.get("data_path")
                }
        frame_info.append(frame_entry)
    scenario[SD.METADATA]["frame_info"] = frame_info

    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(scenario, f)
    logger.info(f"Scenario saved to {output_path}")

    DatasetManager.create_dataset_files(output_path, output_dir)

    if render:
        assert False, "Does not support sequential rendering now"
         # Use camera_output_dir for dataset files and images if provided, otherwise use output_dir
        assets_dir = camera_output_dir if camera_output_dir else (output_dir if output_dir else ".")
        if camera_output_dir:
            os.makedirs(camera_output_dir, exist_ok=True)
        generate_images(output_path, frames, assets_dir, max_frames=max_frames, interpolate=interpolate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OpenScene logs to MetaDrive format with NavSim scene filtering"
    )

    # New scene filter workflow arguments
    parser.add_argument(
        "--scene-filter",
        type=str,
        help="Path to scene_filter YAML file (e.g., scene_filter/navtest.yaml)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing OpenScene .pkl log files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./converted_scenes",
        help="Directory to save converted MetaDrive .pkl files (default: ./converted_scenes)"
    )

    # Legacy single-file conversion arguments (for backwards compatibility)
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="[LEGACY] Path to single input OpenScene .pkl file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sd_output.pkl",
        help="[LEGACY] Name of output MetaDrive .pkl file"
    )

    # Common arguments
    parser.add_argument(
        "--map-root",
        type=str,
        default=os.environ.get("NUPLAN_MAPS_ROOT", "/avl-west/nuplan/maps/"),
        help="Root directory of nuplan maps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Generate rendered images after conversion"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Limit number of frames to render (default: all)"
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Interpolate 2Hz input to 10Hz output"
    )
    parser.add_argument(
        "--num-future-frames-extract",
        type=int,
        default=None,
        help="Override num_future_frames for extraction window size (for converting more frames while keeping same sampling interval)"
    )

    args = parser.parse_args()

    # ========================================================================
    # NEW WORKFLOW: Scene Filter Batch Processing
    # ========================================================================
    if args.scene_filter and args.input_dir:
        logger.info("=" * 80)
        logger.info("BATCH CONVERSION MODE: Using scene filter")
        logger.info("=" * 80)

        # Load scene filter configuration
        logger.info(f"Loading scene filter config: {args.scene_filter}")
        scene_filter = load_scene_filter_config(args.scene_filter)
        logger.info(f"Scene filter: {scene_filter.num_history_frames} history + {scene_filter.num_future_frames} future frames")

        if args.num_future_frames_extract is not None:
            logger.info(f"Override: Extracting {scene_filter.num_history_frames} history + {args.num_future_frames_extract} future frames (window size: {scene_filter.num_history_frames + args.num_future_frames_extract})")
            logger.info(f"Sampling interval remains: {scene_filter.num_frames} frames")

        # Filter log files
        input_dir = Path(args.input_dir)
        log_files = filter_log_files(input_dir, scene_filter.log_names)
        logger.info(f"Found {len(log_files)} matching log files")

        if len(log_files) == 0:
            logger.error("No matching log files found. Exiting.")
            exit(1)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")

        # Process and convert scenes immediately (memory-efficient streaming approach)
        total_scenes_converted = 0

        for log_file in tqdm(log_files, desc="Processing logs"):
            logger.info(f"Loading {log_file.name}...")
            with open(log_file, 'rb') as f:
                frames = pickle.load(f)

            # Extract scene windows
            windows = extract_scene_windows(frames, scene_filter, args.num_future_frames_extract)

            # Filter by tokens
            filtered_windows = filter_by_token(windows, scene_filter.tokens)

            # Convert each scene immediately (no accumulation in memory)
            for token, frame_window in filtered_windows:
                # Save .pkl file in main output directory
                output_path = output_dir / f"sd_{token}" / f"sd_{token}_0" / f"sd_{token}.pkl"

                # Create scene-specific directory for camera images
                # generate_images will create output_dir/sd_token/camera/CAM_XX/ structure
                camera_dir = output_dir / "camera" / f"sd_{token}"

                try:
                    convert_scenario_from_frames(
                        frames=frame_window,
                        scene_token=token,
                        output_path=str(output_path),
                        map_root=args.map_root,
                        render=args.render,
                        max_frames=args.frames,
                        interpolate=args.interpolate,
                        camera_output_dir=str(camera_dir)
                    )
                    total_scenes_converted += 1
                except Exception as e:
                    logger.error(f"Failed to convert scene {token}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Free memory after processing each log file
            del frames

            # # Commented out: max_scenes limit not used in workflow
            # if scene_filter.max_scenes and total_scenes_converted >= scene_filter.max_scenes:
            #     logger.info(f"Reached max_scenes limit ({scene_filter.max_scenes})")
            #     break

        logger.info("=" * 80)
        logger.info(f"BATCH CONVERSION COMPLETE: {total_scenes_converted} scenes converted")
        logger.info(f"Output location: {output_dir.absolute()}")
        logger.info("=" * 80)

    # ========================================================================
    # LEGACY WORKFLOW: Single File Conversion
    # ========================================================================
    elif args.input:
        logger.info("=" * 80)
        logger.info("LEGACY MODE: Single file conversion")
        logger.info("=" * 80)

        # Load frames from single file
        logger.info(f"Loading OpenScene data from {args.input}...")
        with open(args.input, "rb") as f:
            frames = pickle.load(f)

        if not frames:
            logger.error("Empty pickle file.")
            exit(1)

        # Generate output path
        input_stem = os.path.splitext(os.path.basename(args.input))[0]
        user_output_base_dir = os.path.dirname(args.output)
        user_output_filename = os.path.basename(args.output)
        final_output_dir = os.path.join(user_output_base_dir, input_stem)
        os.makedirs(final_output_dir, exist_ok=True)
        final_output_path = os.path.join(final_output_dir, user_output_filename)

        logger.info(f"Created output directory: {os.path.abspath(final_output_dir)}")

        # Use first frame's scene_name or token as identifier
        first_frame = frames[0]
        scene_id = first_frame.get("scene_name") or first_frame.get("token", "unknown_scene")

        # Convert using new function
        convert_scenario_from_frames(
            frames=frames,
            scene_token=scene_id,
            output_path=final_output_path,
            map_root=args.map_root,
            render=args.render,
            max_frames=args.frames,
            interpolate=args.interpolate
        )

        logger.info(f"Conversion complete: {final_output_path}")

    else:
        logger.error("Invalid arguments!")
        logger.error("Choose one of:")
        logger.error("  1. Batch mode: --scene-filter <yaml> --input-dir <dir>")
        logger.error("  2. Legacy mode: --input <file.pkl>")
        parser.print_help()
        exit(1)

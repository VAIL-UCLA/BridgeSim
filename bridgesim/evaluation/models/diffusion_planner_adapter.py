"""
Model adapter for Diffusion Planner (ICLR 2025).
A vectorized diffusion-based trajectory planner — no cameras needed.
Uses agent histories, HD map lanes, and route info as structured tensor inputs.

Vendored model code lives in bridgesim/modelzoo/diffusion_planner/.
"""

import json
import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional
from shapely import LineString

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.modelzoo.diffusion_planner.model.diffusion_planner import Diffusion_Planner
from bridgesim.modelzoo.diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer


# ---------------------------------------------------------------------------
# Default model hyper-parameters (match the released checkpoint)
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_DIM = 192
DEFAULT_NUM_HEADS = 6
DEFAULT_ENCODER_DEPTH = 3
DEFAULT_DECODER_DEPTH = 3
DEFAULT_ENCODER_DROP_PATH = 0.1
DEFAULT_DECODER_DROP_PATH = 0.0
DEFAULT_AGENT_NUM = 32
DEFAULT_STATIC_OBJECTS_NUM = 5
DEFAULT_STATIC_OBJECTS_STATE_DIM = 10
DEFAULT_LANE_NUM = 70
DEFAULT_LANE_LEN = 20
DEFAULT_ROUTE_NUM = 25
DEFAULT_ROUTE_LEN = 20         # same as lane_len in released config
DEFAULT_TIME_LEN = 21          # 2s history @ 10Hz + current = 21
DEFAULT_FUTURE_LEN = 80        # 8s future @ 10Hz
DEFAULT_PREDICTED_NEIGHBOR_NUM = 10
DEFAULT_DIFFUSION_MODEL_TYPE = "x_start"


def _build_config(args_json_path: Optional[str], normalization_json_path: str, device: str):
    """Build a lightweight config namespace that Diffusion_Planner accepts."""

    # Start from defaults; override with args.json if provided
    cfg = {
        "hidden_dim": DEFAULT_HIDDEN_DIM,
        "num_heads": DEFAULT_NUM_HEADS,
        "encoder_depth": DEFAULT_ENCODER_DEPTH,
        "decoder_depth": DEFAULT_DECODER_DEPTH,
        "encoder_drop_path_rate": DEFAULT_ENCODER_DROP_PATH,
        "decoder_drop_path_rate": DEFAULT_DECODER_DROP_PATH,
        "agent_num": DEFAULT_AGENT_NUM,
        "static_objects_num": DEFAULT_STATIC_OBJECTS_NUM,
        "static_objects_state_dim": DEFAULT_STATIC_OBJECTS_STATE_DIM,
        "lane_num": DEFAULT_LANE_NUM,
        "lane_len": DEFAULT_LANE_LEN,
        "route_num": DEFAULT_ROUTE_NUM,
        "route_len": DEFAULT_ROUTE_LEN,
        "time_len": DEFAULT_TIME_LEN,
        "future_len": DEFAULT_FUTURE_LEN,
        "predicted_neighbor_num": DEFAULT_PREDICTED_NEIGHBOR_NUM,
        "diffusion_model_type": DEFAULT_DIFFUSION_MODEL_TYPE,
        "device": device,
    }

    if args_json_path is not None:
        with open(args_json_path, 'r') as f:
            saved = json.load(f)
        for k, v in saved.items():
            if k in ("state_normalizer", "observation_normalizer"):
                continue  # handled separately below
            cfg[k] = v

    # Build normalizers from normalization.json
    with open(normalization_json_path, 'r') as f:
        norm_data = json.load(f)

    predicted_neighbor_num = cfg["predicted_neighbor_num"]
    state_mean = [[norm_data["ego"]["mean"]]] + [[norm_data["neighbor"]["mean"]]] * predicted_neighbor_num
    state_std = [[norm_data["ego"]["std"]]] + [[norm_data["neighbor"]["std"]]] * predicted_neighbor_num
    cfg["state_normalizer"] = StateNormalizer(state_mean, state_std)

    obs_ndt = {}
    for k, v in norm_data.items():
        if k not in ("ego", "neighbor"):
            obs_ndt[k] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float32),
                "std": torch.tensor(v["std"], dtype=torch.float32),
            }
    cfg["observation_normalizer"] = ObservationNormalizer(obs_ndt)

    # No guidance by default
    cfg["guidance_fn"] = None

    # Return as a SimpleNamespace so attribute access works
    class _Cfg:
        pass
    c = _Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# Coordinate transform helpers
# ---------------------------------------------------------------------------
def _world_to_ego(points_world: np.ndarray, ego_x: float, ego_y: float, ego_heading: float) -> np.ndarray:
    """Transform (N, 2) world-frame points to ego-local frame."""
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    dx = points_world[:, 0] - ego_x
    dy = points_world[:, 1] - ego_y
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return np.stack([local_x, local_y], axis=-1)


def _heading_world_to_ego(heading_world: float, ego_heading: float) -> float:
    """Transform a world heading to ego-local heading."""
    return heading_world - ego_heading


def _velocity_world_to_ego(vx: float, vy: float, ego_heading: float):
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    local_vx = cos_h * vx + sin_h * vy
    local_vy = -sin_h * vx + cos_h * vy
    return local_vx, local_vy


def _interpolate_polyline(points: np.ndarray, num_points: int) -> np.ndarray:
    """Interpolate a polyline to a fixed number of points using shapely."""
    if len(points) < 2:
        return np.zeros((num_points, 2), dtype=np.float64)
    line = LineString(points)
    if line.length < 1e-6:
        return np.tile(points[0], (num_points, 1))
    distances = np.linspace(0, line.length, num_points)
    new_points = np.array([line.interpolate(d).coords[0] for d in distances])
    return new_points


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class DiffusionPlannerAdapter(BaseModelAdapter):
    """
    Adapter for Diffusion Planner — a vector-input (no camera) diffusion model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        args_json_path: str = None,
        normalization_json_path: str = None,
        **kwargs,
    ):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.args_json_path = args_json_path

        # Resolve normalization.json: fallback to vendored copy
        if normalization_json_path is None:
            normalization_json_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "modelzoo" / "diffusion_planner" / "normalization.json"
            )
        self.normalization_json_path = normalization_json_path

        # Model config & instance (set in load_model)
        self._cfg = None

        # History buffers for temporal input accumulation
        self._ego_history: deque = deque(maxlen=DEFAULT_TIME_LEN)  # list of 10-dim arrays
        self._frame_ego_states: deque = deque(maxlen=DEFAULT_TIME_LEN)  # raw ego_state dicts

    # ----- BaseModelAdapter interface -----

    def load_model(self):
        self._cfg = _build_config(self.args_json_path, self.normalization_json_path, self.device)
        self.model = Diffusion_Planner(self._cfg)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        # Handle different checkpoint formats
        if "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Strip module. prefix if present (DDP)
        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "")
            cleaned[key] = v

        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[DiffusionPlanner] Loaded model from {self.checkpoint_path}")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {}  # no cameras needed

    def get_waypoint_dt(self) -> float:
        return 0.5  # subsample from 10Hz to 2Hz for controller (same convention as other models)

    def get_trajectory_time_horizon(self) -> float:
        return 8.0  # 80 steps * 0.1s

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        """
        Build vectorized inputs for Diffusion Planner from MetaBench data.
        """
        # Store ego state for history accumulation
        self._frame_ego_states.append(ego_state)

        # Current ego pose (world frame)
        ego_x, ego_y = ego_state['position'][0], ego_state['position'][1]
        ego_heading = ego_state['heading']
        ego_vel = ego_state['velocity']
        ego_acc = ego_state['acceleration']
        ego_angular = ego_state['angular_velocity']

        # ----- 1. Ego current state (10-dim, in ego-local = zeros for position/heading) -----
        vx_local, vy_local = _velocity_world_to_ego(ego_vel[0], ego_vel[1], ego_heading)
        ax_local, ay_local = _velocity_world_to_ego(ego_acc[0], ego_acc[1], ego_heading)
        steering_angle = 0.0  # not available from MetaBench
        yaw_rate = ego_angular[2] if len(ego_angular) > 2 else 0.0

        ego_current = np.array([
            0.0, 0.0,        # x, y in ego frame = origin
            1.0, 0.0,        # cos(0), sin(0) — heading in ego frame = 0
            vx_local, vy_local,
            ax_local, ay_local,
            steering_angle, yaw_rate,
        ], dtype=np.float32)

        # ----- 2. Neighbor agents past (32 agents, 21 timesteps, 11-dim) -----
        neighbor_agents_past = self._build_neighbor_agents(
            scenario_data, frame_id, ego_x, ego_y, ego_heading
        )

        # ----- 3. Static objects (5, 10-dim) -----
        static_objects = self._build_static_objects(
            scenario_data, frame_id, ego_x, ego_y, ego_heading
        )

        # ----- 4. Lane features (70 lanes, 20 points, 12-dim) -----
        lanes, lanes_speed_limit, lanes_has_speed_limit, lane_candidates = self._build_lane_features(
            scenario_data, ego_x, ego_y, ego_heading
        )

        # ----- 5. Route lanes (25 lanes, 20 points, 12-dim) -----
        route_lanes = self._build_route_lanes(
            scenario_data, ego_state, ego_x, ego_y, ego_heading, lanes, frame_id, lane_candidates
        )

        # Build tensor dict
        device = self.device
        data = {
            "ego_current_state": torch.tensor(ego_current, dtype=torch.float32).unsqueeze(0).to(device),
            "neighbor_agents_past": torch.tensor(neighbor_agents_past, dtype=torch.float32).unsqueeze(0).to(device),
            "static_objects": torch.tensor(static_objects, dtype=torch.float32).unsqueeze(0).to(device),
            "lanes": torch.tensor(lanes, dtype=torch.float32).unsqueeze(0).to(device),
            "lanes_speed_limit": torch.tensor(lanes_speed_limit, dtype=torch.float32).unsqueeze(0).to(device),
            "lanes_has_speed_limit": torch.tensor(lanes_has_speed_limit, dtype=torch.bool).unsqueeze(0).to(device),
            "route_lanes": torch.tensor(route_lanes, dtype=torch.float32).unsqueeze(0).to(device),
        }

        # Apply observation normalization (z-score)
        obs_normalizer = self._cfg.observation_normalizer
        data = obs_normalizer(data)

        return data

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            encoder_outputs, decoder_outputs = self.model(model_input)
        return decoder_outputs

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # prediction shape: [B, P, future_len, 4]  (x, y, cos_h, sin_h)
        prediction = model_output["prediction"]  # [1, 11, 80, 4]
        ego_traj = prediction[0, 0].cpu().numpy()  # [80, 4] — ego agent

        # ego_traj is in ego-local frame: (x, y, cos_h, sin_h)
        # Subsample from 10Hz (80 steps) to 2Hz (16 steps) = every 5th step
        subsample_stride = 5
        ego_traj_sub = ego_traj[subsample_stride - 1::subsample_stride]  # [16, 4]

        # MetaBench expects (N, 2) in ego frame with columns (lateral, forward)
        # Diffusion Planner output: x = forward, y = lateral in ego frame
        # So swap: col0 = y (lateral), col1 = x (forward)
        trajectory = np.column_stack([ego_traj_sub[:, 1], ego_traj_sub[:, 0]])

        return {"trajectory": trajectory}

    def reset_temporal_history(self):
        """Clear history buffers between scenarios."""
        self._ego_history.clear()
        self._frame_ego_states.clear()

    # ----- Private helpers -----

    def _build_neighbor_agents(
        self,
        scenario_data: Dict[str, Any],
        frame_id: int,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
    ) -> np.ndarray:
        """
        Build neighbor_agents_past: (agent_num, time_len, 11) in ego-local frame.
        11-dim: x, y, cos_h, sin_h, vx, vy, length, width, type_onehot(3)
        """
        agent_num = getattr(self._cfg, 'agent_num', DEFAULT_AGENT_NUM)
        time_len = getattr(self._cfg, 'time_len', DEFAULT_TIME_LEN)
        result = np.zeros((agent_num, time_len, 11), dtype=np.float32)

        tracks = scenario_data.get('tracks', {})
        sdc_id = scenario_data.get('metadata', {}).get('sdc_id', None)

        # Collect valid agents at current frame, sorted by distance
        agent_infos = []
        for obj_id, track in tracks.items():
            if obj_id == sdc_id:
                continue
            positions = track['state']['position']
            valid = track['state'].get('valid', None)

            if frame_id >= len(positions):
                continue
            if valid is not None and not valid[frame_id]:
                continue

            pos = positions[frame_id][:2]
            dist = np.linalg.norm(pos - np.array([ego_x, ego_y]))
            if dist > 100.0:  # 100m radius
                continue
            agent_infos.append((obj_id, dist))

        # Sort by distance, keep closest agent_num
        agent_infos.sort(key=lambda x: x[1])
        agent_infos = agent_infos[:agent_num]

        # Fill history for each agent
        for agent_idx, (obj_id, _) in enumerate(agent_infos):
            track = tracks[obj_id]
            positions = track['state']['position']
            headings = track['state']['heading']
            velocities = track['state'].get('velocity', None)
            valid = track['state'].get('valid', None)
            obj_length = track['state'].get('length', np.zeros_like(headings))
            obj_width = track['state'].get('width', np.zeros_like(headings))
            obj_type = track.get('type', 'VEHICLE')

            # Type one-hot: vehicle=[1,0,0], pedestrian=[0,1,0], bicycle=[0,0,1]
            type_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if isinstance(obj_type, str):
                obj_type_lower = obj_type.lower()
                if 'pedestrian' in obj_type_lower:
                    type_vec = np.array([0.0, 1.0, 0.0])
                elif 'bicycle' in obj_type_lower or 'cyclist' in obj_type_lower:
                    type_vec = np.array([0.0, 0.0, 1.0])
                else:
                    type_vec = np.array([1.0, 0.0, 0.0])

            for t_idx in range(time_len):
                # Map history index: frame_id - (time_len - 1 - t_idx)
                hist_frame = frame_id - (time_len - 1 - t_idx)
                if hist_frame < 0 or hist_frame >= len(positions):
                    continue
                if valid is not None and not valid[hist_frame]:
                    continue

                pos = positions[hist_frame][:2]
                heading = headings[hist_frame]

                # Transform to current ego local frame
                local_pos = _world_to_ego(pos.reshape(1, 2), ego_x, ego_y, ego_heading)[0]
                local_heading = _heading_world_to_ego(heading, ego_heading)

                if velocities is not None and hist_frame < len(velocities):
                    vel = velocities[hist_frame][:2]
                    lvx, lvy = _velocity_world_to_ego(vel[0], vel[1], ego_heading)
                else:
                    lvx, lvy = 0.0, 0.0

                length_val = obj_length[hist_frame] if hasattr(obj_length, '__getitem__') and hist_frame < len(obj_length) else 4.5
                width_val = obj_width[hist_frame] if hasattr(obj_width, '__getitem__') and hist_frame < len(obj_width) else 2.0

                # Handle scalar vs array for length/width
                if isinstance(length_val, np.ndarray):
                    length_val = float(length_val.flat[0]) if length_val.size > 0 else 4.5
                if isinstance(width_val, np.ndarray):
                    width_val = float(width_val.flat[0]) if width_val.size > 0 else 2.0

                result[agent_idx, t_idx] = [
                    local_pos[0], local_pos[1],
                    np.cos(local_heading), np.sin(local_heading),
                    lvx, lvy,
                    float(length_val), float(width_val),
                    type_vec[0], type_vec[1], type_vec[2],
                ]

        return result

    def _build_static_objects(
        self,
        scenario_data: Dict[str, Any],
        frame_id: int,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
    ) -> np.ndarray:
        """Build static_objects: (static_num, 10) in ego-local frame."""
        static_num = getattr(self._cfg, 'static_objects_num', DEFAULT_STATIC_OBJECTS_NUM)
        result = np.zeros((static_num, 10), dtype=np.float32)

        tracks = scenario_data.get('tracks', {})
        sdc_id = scenario_data.get('metadata', {}).get('sdc_id', None)

        static_infos = []
        for obj_id, track in tracks.items():
            if obj_id == sdc_id:
                continue
            obj_type = track.get('type', 'VEHICLE')
            if isinstance(obj_type, str):
                obj_type_lower = obj_type.lower()
            else:
                obj_type_lower = ''

            # Static objects: pedestrians, bicycles, generic objects
            is_static = ('pedestrian' in obj_type_lower or 'bicycle' in obj_type_lower
                         or 'cyclist' in obj_type_lower or 'generic' in obj_type_lower
                         or 'barrier' in obj_type_lower or 'cone' in obj_type_lower)
            if not is_static:
                continue

            positions = track['state']['position']
            valid = track['state'].get('valid', None)
            if frame_id >= len(positions):
                continue
            if valid is not None and not valid[frame_id]:
                continue

            pos = positions[frame_id][:2]
            dist = np.linalg.norm(pos - np.array([ego_x, ego_y]))
            if dist > 100.0:
                continue
            static_infos.append((obj_id, dist))

        static_infos.sort(key=lambda x: x[1])
        static_infos = static_infos[:static_num]

        for idx, (obj_id, _) in enumerate(static_infos):
            track = tracks[obj_id]
            pos = track['state']['position'][frame_id][:2]
            heading = track['state']['heading'][frame_id]
            obj_type = track.get('type', 'VEHICLE')

            local_pos = _world_to_ego(pos.reshape(1, 2), ego_x, ego_y, ego_heading)[0]
            local_heading = _heading_world_to_ego(heading, ego_heading)

            obj_width = track['state'].get('width', None)
            obj_length = track['state'].get('length', None)
            w = float(obj_width[frame_id]) if obj_width is not None and frame_id < len(obj_width) else 1.0
            l = float(obj_length[frame_id]) if obj_length is not None and frame_id < len(obj_length) else 1.0

            # Type one-hot: vehicle=[1,0,0,0], ped=[0,1,0,0], bicycle=[0,0,1,0], generic=[0,0,0,1]
            type_vec = [0.0, 0.0, 0.0, 0.0]
            if isinstance(obj_type, str):
                obj_type_lower = obj_type.lower()
                if 'pedestrian' in obj_type_lower:
                    type_vec = [0.0, 1.0, 0.0, 0.0]
                elif 'bicycle' in obj_type_lower or 'cyclist' in obj_type_lower:
                    type_vec = [0.0, 0.0, 1.0, 0.0]
                else:
                    type_vec = [0.0, 0.0, 0.0, 1.0]

            result[idx] = [
                local_pos[0], local_pos[1],
                np.cos(local_heading), np.sin(local_heading),
                w, l,
                type_vec[0], type_vec[1], type_vec[2], type_vec[3],
            ]

        return result

    def _build_lane_features(
        self,
        scenario_data: Dict[str, Any],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
    ):
        """
        Build lane features from scenario_data['map_features'].
        Returns:
            lanes: (lane_num, lane_len, 12) — x, y, dx, dy, left_dx, left_dy, right_dx, right_dy, tl(4)
            lanes_speed_limit: (lane_num, 1)
            lanes_has_speed_limit: (lane_num, 1)
        """
        lane_num = getattr(self._cfg, 'lane_num', DEFAULT_LANE_NUM)
        lane_len = getattr(self._cfg, 'lane_len', DEFAULT_LANE_LEN)
        lanes = np.zeros((lane_num, lane_len, 12), dtype=np.float32)
        lanes_speed_limit = np.zeros((lane_num, 1), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((lane_num, 1), dtype=np.bool_)

        map_features = scenario_data.get('map_features', {})
        if not map_features:
            return lanes, lanes_speed_limit, lanes_has_speed_limit, []

        # Collect lane-type features with their centerline polylines
        lane_candidates = []

        for feat_id, feat in map_features.items():
            feat_type = feat.get('type', '')
            # MetaDrive scenario data uses type strings like "LANE_SURFACE_STREET"
            if not isinstance(feat_type, str):
                continue
            if 'LANE' not in feat_type.upper():
                continue

            polyline = feat.get('polyline', None)
            if polyline is None:
                continue
            polyline = np.array(polyline, dtype=np.float64)
            if len(polyline) < 2:
                continue

            # Distance from ego to nearest point on lane
            dists = np.linalg.norm(polyline[:, :2] - np.array([ego_x, ego_y]), axis=-1)
            min_dist = dists.min()
            if min_dist > 100.0:
                continue

            lane_candidates.append((feat_id, feat, min_dist, polyline))

        # Sort by distance, keep closest lane_num
        lane_candidates.sort(key=lambda x: x[2])
        lane_candidates = lane_candidates[:lane_num]

        # Get traffic light data
        dynamic_map = scenario_data.get('dynamic_map_states', {})

        for idx, (feat_id, feat, _, polyline) in enumerate(lane_candidates):
            # Interpolate centerline to fixed size
            centerline = _interpolate_polyline(polyline[:, :2], lane_len)
            # Transform to ego-local frame
            centerline_local = _world_to_ego(centerline, ego_x, ego_y, ego_heading)

            # Direction vectors (consecutive point differences)
            direction = np.zeros_like(centerline_local)
            direction[:-1] = centerline_local[1:] - centerline_local[:-1]
            # Last point copies previous direction
            if lane_len > 1:
                direction[-1] = direction[-2]

            # Left/right boundary offsets
            # Try to get polygon or width info for boundary estimation
            width_arr = feat.get('width', None)
            if width_arr is not None:
                width_arr = np.array(width_arr, dtype=np.float64)
                # Interpolate width to lane_len
                if len(width_arr) != lane_len:
                    width_interp = np.interp(
                        np.linspace(0, 1, lane_len),
                        np.linspace(0, 1, len(width_arr)),
                        width_arr
                    )
                else:
                    width_interp = width_arr
            else:
                width_interp = np.full(lane_len, 1.75)  # default half-width

            # Estimate left/right boundaries from centerline + width
            # Normal vector: rotate direction 90 degrees
            dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
            dir_norm = np.maximum(dir_norm, 1e-6)
            dir_unit = direction / dir_norm
            normal_left = np.stack([-dir_unit[:, 1], dir_unit[:, 0]], axis=-1)
            normal_right = -normal_left

            half_width = width_interp[:, None] / 2.0
            left_offset = normal_left * half_width
            right_offset = normal_right * half_width

            # Traffic light encoding (4-dim one-hot)
            tl_encoding = np.zeros((lane_len, 4), dtype=np.float32)
            tl_encoding[:, 3] = 1.0  # default: unknown

            # TODO: match dynamic_map_states to lane features if available

            # Assemble 12-dim feature
            lanes[idx] = np.concatenate([
                centerline_local,       # x, y (2)
                direction,              # dx, dy (2)
                left_offset,            # left_x-x, left_y-y (2)
                right_offset,           # right_x-x, right_y-y (2)
                tl_encoding,            # traffic_light (4)
            ], axis=-1)

            # Speed limit (not typically available in MetaDrive scenarios)
            lanes_speed_limit[idx] = 0.0
            lanes_has_speed_limit[idx] = False

        return lanes, lanes_speed_limit, lanes_has_speed_limit, lane_candidates

    def _build_route_lanes(
        self,
        scenario_data: Dict[str, Any],
        ego_state: Dict[str, Any],
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        lanes: np.ndarray,
        frame_id: int,
        lane_candidates: List,
    ) -> np.ndarray:
        """
        Build route_lanes: (route_num, lane_len, 12).

        If scenario_data has roadblock_ids_per_frame (set by converter), use them to
        precisely identify route lanes by matching roadblock_id in map_features.
        Otherwise fall back to direction-alignment scoring.
        """
        route_num = getattr(self._cfg, 'route_num', DEFAULT_ROUTE_NUM)
        lane_len = getattr(self._cfg, 'lane_len', DEFAULT_LANE_LEN)
        route_lanes = np.zeros((route_num, lane_len, 12), dtype=np.float32)

        # --- Precise method: use roadblock_ids_per_frame ---
        roadblock_ids_per_frame = scenario_data.get('metadata', {}).get('roadblock_ids_per_frame', None)
        if roadblock_ids_per_frame is not None and frame_id < len(roadblock_ids_per_frame):
            route_roadblock_ids = set(str(rid) for rid in roadblock_ids_per_frame[frame_id])
            if route_roadblock_ids:
                route_idx = 0
                for rb_id in roadblock_ids_per_frame[frame_id]:
                    rb_id = str(rb_id)
                    for tensor_idx, (_, feat, _, _) in enumerate(lane_candidates):
                        if route_idx >= route_num:
                            break
                        if str(feat.get('roadblock_id', '')) == rb_id:
                            route_lanes[route_idx] = lanes[tensor_idx]
                            route_idx += 1
                    if route_idx >= route_num:
                        break
                if route_idx > 0:
                    return route_lanes
                # If no matches found (roadblock_ids don't overlap map_features), fall through

        # --- Fallback: direction-alignment scoring ---
        waypoint = ego_state.get('waypoint', None)
        if waypoint is None:
            return route_lanes

        wp_local = _world_to_ego(np.array(waypoint).reshape(1, 2), ego_x, ego_y, ego_heading)[0]
        route_dir = wp_local / (np.linalg.norm(wp_local) + 1e-6)

        lane_scores = []
        for i in range(lanes.shape[0]):
            lane_center = lanes[i, :, :2]
            if np.abs(lane_center).sum() < 1e-6:
                continue
            lane_mean_dir = lanes[i, :, 2:4].mean(axis=0)
            lane_mean_dir_norm = lane_mean_dir / (np.linalg.norm(lane_mean_dir) + 1e-6)
            alignment = np.dot(route_dir, lane_mean_dir_norm)
            lane_center_mean = lane_center.mean(axis=0)
            dist_to_ego = np.linalg.norm(lane_center_mean)
            score = alignment - 0.01 * dist_to_ego
            lane_scores.append((i, score))

        lane_scores.sort(key=lambda x: -x[1])
        for route_idx, (lane_idx, _) in enumerate(lane_scores[:route_num]):
            route_lanes[route_idx] = lanes[lane_idx]

        return route_lanes

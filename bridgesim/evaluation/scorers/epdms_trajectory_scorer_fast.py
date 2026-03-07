"""
Optimized EPDMS trajectory scorer for inference scaling.

Key optimizations over epdms_trajectory_scorer.py:
1. Pre-compute agent polygons and traffic light states per frame (shared across all candidates)
2. Cache ego polygons and nearby lanes within each candidate evaluation
3. AABB pre-filter before expensive Shapely intersection checks
4. Vectorized trajectory state computation across all candidates
"""

import math
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from shapely.geometry import Polygon, Point
from shapely import affinity
from scipy.signal import savgol_filter

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer

# --- Constants ---
VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH = 1.852
VEHICLE_HALF_DIAG = math.sqrt((VEHICLE_LENGTH / 2) ** 2 + (VEHICLE_WIDTH / 2) ** 2)

VEHICLE_POLYGON_COORDS = np.array([
    [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2],
    [VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2]
])

W_PROGRESS, W_TTC, W_LANE_KEEPING, W_HISTORY_COMFORT, W_EXTENDED_COMFORT = 5.0, 5.0, 2.0, 2.0, 2.0
W_TOTAL = W_PROGRESS + W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT
LANE_DEVIATION_LIMIT = 0.5
LANE_KEEPING_WINDOW = 2.0
TTC_HORIZON = 1.0
STOPPED_SPEED_THRESHOLD = 5e-2
MAX_ACCEL, MAX_JERK, MAX_YAW_RATE = 4.89, 8.37, 0.95
EC_ACCEL_THRESH, EC_JERK_THRESH, EC_YAW_RATE_THRESH = 0.7, 0.5, 0.1


class EPDMSTrajectoryScorer_Fast(BaseTrajectoryScorer):
    """
    Optimized EPDMS trajectory scorer.

    Pre-computes per-frame data (agent polygons, traffic lights) shared across
    all N candidates, caches per-candidate data (ego polygons, nearby lanes),
    and uses AABB pre-filtering to minimize expensive Shapely operations.
    """

    def __init__(self):
        self._initialized = False
        self.scenario_data = None
        self.env = None
        self.sdc_id = None
        self.all_lanes = []
        self.all_lane_bounds = None  # (num_lanes, 4) array of [minx, miny, maxx, maxy]
        self.traffic_lights = {}
        self.world_to_sim_offset = np.array([0.0, 0.0])
        self.scenario_dt = 0.1
        self.planner_dt = 0.5
        self.gt_stride = 5
        self.prev_frame_idx = None

    def initialize(self, scenario_data: dict, env):
        """Initialize scorer with scenario data and environment."""
        self.scenario_data = scenario_data
        self.env = env
        self.sdc_id = scenario_data['metadata']['sdc_id']

        # Time step alignment
        self.scenario_dt = 0.1
        metadata = scenario_data.get('metadata', {})
        ts_key = 'timestep' if 'timestep' in metadata else 'ts'
        if ts_key in metadata:
            timestep = metadata[ts_key]
            if isinstance(timestep, np.ndarray):
                if timestep.size == 1:
                    self.scenario_dt = float(timestep.item())
                elif timestep.size > 1:
                    self.scenario_dt = float(timestep.flat[0]) if timestep.flat[0] > 0 else 0.1
            elif isinstance(timestep, (list, tuple)):
                self.scenario_dt = float(timestep[0]) if len(timestep) > 0 else 0.1
            else:
                self.scenario_dt = float(timestep)
        self.planner_dt = 0.5
        if self.scenario_dt > 0:
            self.gt_stride = int(round(self.planner_dt / self.scenario_dt))
        else:
            self.gt_stride = 5

        # Map extraction + pre-compute lane bounds for vectorized nearby query
        self.all_lanes = []
        self.traffic_lights = scenario_data.get('dynamic_map_states', {})

        if self.env and self.env.engine.map_manager.current_map:
            road_network = self.env.engine.map_manager.current_map.road_network
            if hasattr(road_network, 'get_all_lanes'):
                for lane in road_network.get_all_lanes():
                    if hasattr(lane, 'shapely_polygon'):
                        self.all_lanes.append((lane, lane.shapely_polygon))
            else:
                for start_node, end_dict in road_network.graph.items():
                    for end_node, lanes in end_dict.items():
                        for lane in lanes:
                            if hasattr(lane, 'shapely_polygon'):
                                self.all_lanes.append((lane, lane.shapely_polygon))

        # Pre-compute lane bounds as numpy array for vectorized nearby queries
        if self.all_lanes:
            bounds = np.array([poly.bounds for _, poly in self.all_lanes])  # (num_lanes, 4)
            self.all_lane_bounds = bounds  # [minx, miny, maxx, maxy]
        else:
            self.all_lane_bounds = np.empty((0, 4))

        # Coordinate calibration (world -> sim)
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if sdc_track['state']['valid'][0]:
            start_pos_world = sdc_track['state']['position'][0][:2]
            start_pos_sim = self.env.agent.position
            self.world_to_sim_offset = np.array(start_pos_sim) - np.array(start_pos_world)
            print(f"[EPDMS Fast] Calibrated World->Sim Offset: {self.world_to_sim_offset}")
        else:
            self.world_to_sim_offset = np.array([0.0, 0.0])
            print("[EPDMS Fast] Could not calibrate coordinates (Frame 0 invalid).")

        self.prev_frame_idx = None
        self._initialized = True
        print(f"[EPDMS Fast] Initialized with {len(self.all_lanes)} lanes")

    # ---------------------------------------------------------------
    # Vectorized helpers
    # ---------------------------------------------------------------

    def _to_sim_frame(self, points: np.ndarray) -> np.ndarray:
        return points + self.world_to_sim_offset

    def _query_nearby_lanes_vec(self, x: float, y: float, radius: float) -> List[int]:
        """Return indices into self.all_lanes using vectorized bounds check."""
        if len(self.all_lane_bounds) == 0:
            return []
        bounds = self.all_lane_bounds
        mask = ((bounds[:, 0] - radius < x) & (x < bounds[:, 2] + radius) &
                (bounds[:, 1] - radius < y) & (y < bounds[:, 3] + radius))
        return np.where(mask)[0].tolist()

    def _get_ego_polygon(self, x: float, y: float, heading: float) -> Polygon:
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        corners = np.empty((4, 2))
        for i, (cx, cy) in enumerate(VEHICLE_POLYGON_COORDS):
            corners[i, 0] = x + cx * cos_h - cy * sin_h
            corners[i, 1] = y + cx * sin_h + cy * cos_h
        return Polygon(corners)

    def _get_ego_aabb(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """Fast AABB for ego vehicle (conservative circle-based)."""
        return (x - VEHICLE_HALF_DIAG, y - VEHICLE_HALF_DIAG,
                x + VEHICLE_HALF_DIAG, y + VEHICLE_HALF_DIAG)

    @staticmethod
    def _aabb_overlap(a: Tuple, b: Tuple) -> bool:
        """Check if two AABBs (minx, miny, maxx, maxy) overlap."""
        return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

    def _get_best_lane(self, x: float, y: float, heading: float,
                       nearby_indices: List[int], speed: float = None):
        if not nearby_indices:
            return None
        candidates = []
        is_stopped = (speed is not None) and (speed < 1.0)
        pos = np.array([x, y])

        for idx in nearby_indices:
            lane = self.all_lanes[idx][0]
            s, r = lane.local_coordinates(pos)
            s_clamped = max(0, min(s, lane.length))
            lane_heading_vec = lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            dist = lane.distance(pos)

            if is_stopped or (abs(diff) < (np.pi / 2)):
                candidates.append((lane, dist))

        if candidates:
            return min(candidates, key=lambda c: c[1])[0]
        return None

    # ---------------------------------------------------------------
    # Pre-computation (called once per select_best, shared by all N)
    # ---------------------------------------------------------------

    def _precompute_frame_data(self, frame_idx: int, horizon: int):
        """Pre-compute per-timestep agent polygons and traffic light states."""
        tracks = self.scenario_data['tracks']
        scenario_length = self.scenario_data['length']

        # Agent data: list of lists, one per timestep
        # Each entry: (poly, aabb, x, y, heading, type)
        agents_per_t = []
        for t in range(horizon):
            sim_frame = frame_idx + (t * self.gt_stride)
            agents_at_t = []
            if sim_frame < scenario_length:
                for obj_id, track in tracks.items():
                    if obj_id == self.sdc_id:
                        continue
                    if track['type'] not in ['VEHICLE', 'CYCLIST']:
                        continue
                    pos_arr = track['state']['position']
                    valid_arr = track['state']['valid']
                    if sim_frame >= len(pos_arr) or not valid_arr[sim_frame]:
                        continue
                    obj_pos_world = pos_arr[sim_frame][:2]
                    ax, ay = obj_pos_world + self.world_to_sim_offset
                    ah = track['state']['heading'][sim_frame]
                    poly = self._get_ego_polygon(ax, ay, ah)
                    aabb = poly.bounds  # (minx, miny, maxx, maxy)
                    agents_at_t.append((poly, aabb, ax, ay))
            agents_per_t.append(agents_at_t)

        # Traffic light red lane IDs per timestep
        red_lanes_per_t = []
        for t in range(horizon):
            sim_frame = frame_idx + (t * self.gt_stride)
            red_ids = set()
            if sim_frame < scenario_length:
                for lane_id, tl_data in self.traffic_lights.items():
                    s_list = tl_data['state']['object_state']
                    if sim_frame < len(s_list) and s_list[sim_frame] == "LANE_STATE_STOP":
                        red_ids.add(str(lane_id))
            red_lanes_per_t.append(red_ids)

        return agents_per_t, red_lanes_per_t

    # ---------------------------------------------------------------
    # Batched trajectory state computation
    # ---------------------------------------------------------------

    def _get_all_trajectory_states(self, all_paths: np.ndarray, dt: float):
        """
        Vectorized trajectory states for all N candidates.

        Args:
            all_paths: (N, T, 2) array of paths in sim frame
            dt: time step between waypoints

        Returns:
            dict of arrays, each (N, T):
                x, y, heading, speed, acceleration, jerk, yaw_rate
        """
        N, T, _ = all_paths.shape
        paths = all_paths.copy()

        # Savgol filtering per candidate (can't fully vectorize due to per-row filter)
        window_size = min(7, T if T % 2 != 0 else T - 1)
        if window_size >= 3:
            for i in range(N):
                try:
                    paths[i, :, 0] = savgol_filter(paths[i, :, 0], window_length=window_size, polyorder=2)
                    paths[i, :, 1] = savgol_filter(paths[i, :, 1], window_length=window_size, polyorder=2)
                except Exception:
                    pass

        # Vectorized finite differences across all candidates: (N, T-1, 2)
        dpath = np.diff(paths, axis=1) / dt
        # Pad last row: (N, T, 2)
        vel_vec = np.concatenate([dpath, dpath[:, -1:, :]], axis=1)

        heading = np.arctan2(vel_vec[:, :, 1], vel_vec[:, :, 0])  # (N, T)
        heading = np.unwrap(heading, axis=1)

        dacc = np.diff(vel_vec, axis=1) / dt
        acc_vec = np.concatenate([dacc, dacc[:, -1:, :]], axis=1)
        acc_mag = np.linalg.norm(acc_vec, axis=2)  # (N, T)

        djerk = np.diff(acc_vec, axis=1) / dt
        jerk_vec = np.concatenate([djerk, djerk[:, -1:, :]], axis=1)
        jerk_mag = np.linalg.norm(jerk_vec, axis=2)  # (N, T)

        yaw_rate_raw = np.diff(heading, axis=1) / dt
        yaw_rate = np.concatenate([yaw_rate_raw, yaw_rate_raw[:, -1:]], axis=1)  # (N, T)

        speed = np.linalg.norm(vel_vec, axis=2)  # (N, T)
        stopped_mask = speed < STOPPED_SPEED_THRESHOLD
        acc_mag[stopped_mask] = 0.0
        jerk_mag[stopped_mask] = 0.0
        yaw_rate[stopped_mask] = 0.0

        return {
            "x": paths[:, :, 0],  # (N, T)
            "y": paths[:, :, 1],
            "heading": heading,
            "speed": speed,
            "acceleration": acc_mag,
            "jerk": jerk_mag,
            "yaw_rate": yaw_rate,
        }

    # ---------------------------------------------------------------
    # Per-candidate metric evaluation (with caches + precomputed data)
    # ---------------------------------------------------------------

    def _calculate_metrics(self, states: dict, horizon: int, frame_idx: int,
                           ego_state: Optional[Dict[str, Any]] = None,
                           n_execute: int = None,
                           agents_per_t: list = None,
                           red_lanes_per_t: list = None) -> dict:
        metrics = {
            "nc": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
            "ep": 0.0, "ttc": 1.0, "lk": 1.0, "hc": 1.0, "ec": 1.0
        }

        # --- Pre-build caches for this candidate ---
        # Ego polygons: create once, reuse for DAC, NC, TLC
        ego_polys = [self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
                     for t in range(horizon)]
        ego_aabbs = [self._get_ego_aabb(states['x'][t], states['y'][t])
                     for t in range(horizon)]

        # Nearby lanes cache: key = (t, radius) -> list of lane indices
        nearby_cache = {}

        def get_nearby(t, radius):
            key = (t, radius)
            if key not in nearby_cache:
                nearby_cache[key] = self._query_nearby_lanes_vec(
                    states['x'][t], states['y'][t], radius)
            return nearby_cache[key]

        # 1. DAC (ratio of waypoints inside drivable area)
        dac_valid = 0
        for t in range(horizon):
            nearby_idx = get_nearby(t, 15.0)
            corners = list(ego_polys[t].exterior.coords[:-1])
            corners_valid = 0
            for cx, cy in corners:
                pt = Point(cx, cy)
                for idx in nearby_idx:
                    if self.all_lanes[idx][1].contains(pt):
                        corners_valid += 1
                        break
            if corners_valid >= 4:
                dac_valid += 1
        metrics['dac'] = dac_valid / horizon if horizon > 0 else 1.0

        # 2. NC & TTC (with AABB pre-filter + pre-computed agent polys)
        if agents_per_t is not None:
            for t in range(horizon):
                ego_v = states['speed'][t]
                ego_aabb = ego_aabbs[t]
                for agent_poly, agent_aabb, ax, ay in agents_per_t[t]:
                    # AABB pre-filter
                    if not self._aabb_overlap(ego_aabb, agent_aabb):
                        continue
                    if ego_polys[t].intersects(agent_poly):
                        is_at_fault = True
                        if ego_v < STOPPED_SPEED_THRESHOLD:
                            is_at_fault = False
                        dx, dy = ax - states['x'][t], ay - states['y'][t]
                        ego_hx = math.cos(states['heading'][t])
                        ego_hy = math.sin(states['heading'][t])
                        longitudinal_dist = dx * ego_hx + dy * ego_hy
                        if longitudinal_dist < -1.0:
                            is_at_fault = False
                        if is_at_fault:
                            metrics['nc'] = 0.0
                            if t * self.planner_dt <= TTC_HORIZON:
                                metrics['ttc'] = 0.0
                            break
                if metrics['nc'] == 0.0:
                    break

        # 3. DDC
        if metrics['dac'] > 0:
            for t in range(0, horizon, 5):
                nearby_idx = get_nearby(t, 15.0)
                speed = states['speed'][t]
                best_lane = self._get_best_lane(
                    states['x'][t], states['y'][t], states['heading'][t],
                    nearby_idx, speed=speed)
                if best_lane and speed > 1.0:
                    s, r = best_lane.local_coordinates(
                        np.array([states['x'][t], states['y'][t]]))
                    s_clamped = max(0, min(s, best_lane.length))
                    lane_heading_vec = best_lane.heading_at(s_clamped)
                    lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
                    diff = abs(states['heading'][t] - lane_heading)
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    if abs(diff) > np.pi / 2:
                        metrics['ddc'] = 0.0
                        break

        # 4. TLC (with pre-computed red lane IDs)
        if red_lanes_per_t is not None:
            for t in range(horizon):
                red_ids = red_lanes_per_t[t]
                if not red_ids:
                    continue
                nearby_idx = get_nearby(t, 5.0)
                for idx in nearby_idx:
                    lane = self.all_lanes[idx][0]
                    curr_id = str(lane.index)
                    if isinstance(lane.index, (tuple, list)) and len(lane.index) > 2:
                        curr_id = str(lane.index[2])
                    if curr_id not in red_ids:
                        continue
                    if states['speed'][t] > 1.0 and self.all_lanes[idx][1].intersects(ego_polys[t]):
                        s, _ = lane.local_coordinates(
                            np.array([states['x'][t], states['y'][t]]))
                        dist_to_end = lane.length - s
                        if (lane.length < 10.0) or (dist_to_end < 5.0):
                            metrics['tlc'] = 0.0
                            break
                if metrics['tlc'] == 0.0:
                    break

        # 5. Lane Keeping
        consecutive_bad = 0
        if metrics['dac'] > 0:
            for t in range(horizon):
                nearby_idx = get_nearby(t, 10.0)
                speed = states['speed'][t]
                best_lane = self._get_best_lane(
                    states['x'][t], states['y'][t], states['heading'][t],
                    nearby_idx, speed=speed)
                if best_lane:
                    _, lat = best_lane.local_coordinates(
                        np.array([states['x'][t], states['y'][t]]))
                    if abs(lat) > LANE_DEVIATION_LIMIT:
                        consecutive_bad += 1
                    else:
                        consecutive_bad = 0
                if consecutive_bad * self.planner_dt > LANE_KEEPING_WINDOW:
                    metrics['lk'] = 0.0
                    break

        # 6. HC
        max_acc = np.max(states['acceleration'])
        max_jerk = np.max(states['jerk'])
        max_yr = np.max(np.abs(states['yaw_rate']))
        if max_acc > MAX_ACCEL or max_jerk > MAX_JERK or max_yr > MAX_YAW_RATE:
            metrics['hc'] = 0.0

        # 7. EC (smoothness from current ego state through executed waypoints)
        if ego_state is not None:
            ego_acc = np.linalg.norm(ego_state['acceleration'][:2])
            ego_yr = abs(ego_state.get('angular_velocity', np.zeros(3))[2])
            n_exec = min(n_execute if n_execute is not None else horizon, len(states['acceleration']))
            acc_seq = np.concatenate([[ego_acc], states['acceleration'][:n_exec]])
            yr_seq = np.concatenate([[ego_yr], np.abs(states['yaw_rate'][:n_exec])])
            diff_acc = np.abs(np.diff(acc_seq))
            diff_yr = np.abs(np.diff(yr_seq))
            comfortable = (diff_acc <= EC_ACCEL_THRESH) & (diff_yr <= EC_YAW_RATE_THRESH)
            metrics['ec'] = float(comfortable.sum()) / len(comfortable) if len(comfortable) > 0 else 1.0

        # 8. Progress
        dist = math.sqrt((states['x'][-1] - states['x'][0]) ** 2 +
                         (states['y'][-1] - states['y'][0]) ** 2)
        metrics['ep'] = min(dist / 30.0, 1.0)

        return metrics

    # ---------------------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------------------

    def _ego_to_world(self, candidates_np: np.ndarray, ego_state: Dict[str, Any]) -> np.ndarray:
        """Vectorized ego→world transform. (N, 8, 3) → (N, 8, 2)."""
        ego_x, ego_y = ego_state['position'][:2]
        ego_heading = ego_state['heading']
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
        x_fwd = candidates_np[:, :, 0]
        y_lat = candidates_np[:, :, 1]
        world_x = ego_x + x_fwd * cos_h - y_lat * sin_h
        world_y = ego_y + x_fwd * sin_h + y_lat * cos_h
        return np.stack([world_x, world_y], axis=-1)

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("EPDMSTrajectoryScorer_Fast not initialized.")

        t_start = time.time()

        ego_state = kwargs['ego_state']
        frame_idx = kwargs['frame_idx']

        # Compute n_execute from replan interval
        if self.prev_frame_idx is not None:
            replan_interval_s = (frame_idx - self.prev_frame_idx) * self.scenario_dt
            n_execute = max(1, int(round(replan_interval_s / self.planner_dt)))
        else:
            n_execute = None

        all_candidates = model_output["all_candidates"]  # (B, N, 8, 3) tensor
        candidates_np = all_candidates[0].cpu().numpy()  # (N, 8, 3)
        N = candidates_np.shape[0]
        horizon = candidates_np.shape[1]  # 8

        # --- Step 1: Pre-compute per-frame data (shared across all candidates) ---
        agents_per_t, red_lanes_per_t = self._precompute_frame_data(frame_idx, horizon)

        # --- Step 2: Transform all candidates to world frame ---
        candidates_world = self._ego_to_world(candidates_np, ego_state)  # (N, 8, 2)

        # --- Step 3: Build all full paths (prepend ego pos) and transform to sim ---
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if not sdc_track['state']['valid'][frame_idx]:
            # No valid ego position — return zeros
            best_traj = all_candidates[0, 0]
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.zeros(1, N),
                "best_idx": torch.tensor([0]),
            }

        current_pos = sdc_track['state']['position'][frame_idx][:2]
        # (N, 1, 2) ego pos broadcast + (N, 8, 2) candidates → (N, 9, 2) full paths
        ego_pos_tiled = np.broadcast_to(current_pos[None, None, :], (N, 1, 2))
        all_paths_world = np.concatenate([ego_pos_tiled, candidates_world], axis=1)  # (N, 9, 2)
        all_paths_sim = all_paths_world + self.world_to_sim_offset  # (N, 9, 2)

        # --- Step 4: Vectorized trajectory states ---
        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)

        # --- Step 5: Score each candidate ---
        scores = np.zeros(N)
        for i in range(N):
            # Extract per-candidate states (views, no copy)
            states_i = {
                "x": all_states["x"][i],
                "y": all_states["y"][i],
                "heading": all_states["heading"][i],
                "speed": all_states["speed"][i],
                "acceleration": all_states["acceleration"][i],
                "jerk": all_states["jerk"][i],
                "yaw_rate": all_states["yaw_rate"][i],
            }

            metrics = self._calculate_metrics(
                states_i, horizon, frame_idx,
                ego_state=ego_state, n_execute=n_execute,
                agents_per_t=agents_per_t, red_lanes_per_t=red_lanes_per_t,
            )

            multi_prod = metrics['nc'] * metrics['dac'] * metrics['ddc'] * metrics['tlc']
            weighted_sum = (W_PROGRESS * metrics['ep'] + W_TTC * metrics['ttc'] +
                            W_LANE_KEEPING * metrics['lk'] + W_HISTORY_COMFORT * metrics['hc'] +
                            W_EXTENDED_COMFORT * metrics['ec'])
            scores[i] = multi_prod * weighted_sum / W_TOTAL

        best_idx = int(np.argmax(scores))
        self.prev_frame_idx = frame_idx

        elapsed = time.time() - t_start
        print(f"[EPDMS Fast] Frame {frame_idx}: best_idx={best_idx}/{N}, "
              f"score={scores[best_idx]:.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, min={scores.min():.4f}, "
              f"time={elapsed:.2f}s ({elapsed/N*1000:.1f}ms/cand)")

        best_traj = all_candidates[0, best_idx]
        return {
            "trajectory": best_traj.unsqueeze(0),
            "scores": torch.from_numpy(scores).unsqueeze(0).float(),
            "best_idx": torch.tensor([best_idx]),
        }

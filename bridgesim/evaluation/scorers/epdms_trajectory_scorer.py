"""
Self-contained EPDMS trajectory scorer for inference scaling.
Scores each candidate trajectory using the full EPDMS metric suite
(NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC) without human filtering.
"""

import math
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from shapely.geometry import Polygon, Point
from shapely import affinity
from scipy.signal import savgol_filter

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer

# --- Constants (mirrored from epdms_scorer_md.py) ---
VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH = 1.852

VEHICLE_POLYGON_COORDS = np.array([
    [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2],
    [VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2]
])

W_PROGRESS, W_TTC, W_LANE_KEEPING, W_HISTORY_COMFORT, W_EXTENDED_COMFORT = 5.0, 5.0, 2.0, 2.0, 2.0
LANE_DEVIATION_LIMIT = 0.5
LANE_KEEPING_WINDOW = 2.0
TTC_HORIZON = 1.0
STOPPED_SPEED_THRESHOLD = 5e-2
MAX_ACCEL, MAX_JERK, MAX_YAW_RATE = 4.89, 8.37, 0.95
EC_ACCEL_THRESH, EC_JERK_THRESH, EC_YAW_RATE_THRESH = 0.7, 0.5, 0.1


class EPDMSTrajectoryScorer(BaseTrajectoryScorer):
    """
    Scorer that evaluates all trajectory candidates using the full EPDMS metric
    suite. Each candidate is transformed from ego-frame to world-frame and scored
    against scenario ground truth (collisions, drivable area, lane keeping, etc.).

    No human filtering is applied — candidates are scored purely on their own merit.

    Applicable to both DiffusionDrive v1 and v2.
    """

    def __init__(self):
        self._initialized = False
        self.scenario_data = None
        self.env = None
        self.sdc_id = None
        self.all_lanes = []
        self.traffic_lights = {}
        self.world_to_sim_offset = np.array([0.0, 0.0])
        self.scenario_dt = 0.1
        self.planner_dt = 0.5
        self.gt_stride = 5
        self.prev_frame_idx = None

    def initialize(self, scenario_data: dict, env):
        """
        Initialize scorer with scenario data and environment.
        Called by base_evaluator after env is created.
        """
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
            else:
                self.scenario_dt = float(timestep)
        self.planner_dt = 0.5
        if self.scenario_dt > 0:
            self.gt_stride = int(round(self.planner_dt / self.scenario_dt))
        else:
            self.gt_stride = 5

        # Map extraction
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

        # Coordinate calibration (world -> sim)
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if sdc_track['state']['valid'][0]:
            start_pos_world = sdc_track['state']['position'][0][:2]
            start_pos_sim = self.env.agent.position
            self.world_to_sim_offset = np.array(start_pos_sim) - np.array(start_pos_world)
            print(f"[EPDMS TrajScorer] Calibrated World->Sim Offset: {self.world_to_sim_offset}")
        else:
            self.world_to_sim_offset = np.array([0.0, 0.0])
            print("[EPDMS TrajScorer] Could not calibrate coordinates (Frame 0 invalid).")

        self.prev_frame_idx = None
        self._initialized = True
        print(f"[EPDMS TrajScorer] Initialized with {len(self.all_lanes)} lanes")

    # --- Helper methods ---

    def _to_sim_frame(self, points: np.ndarray) -> np.ndarray:
        return points + self.world_to_sim_offset

    def _get_trajectory_states(self, path: np.ndarray, dt: float) -> Optional[Dict[str, np.ndarray]]:
        if path.shape[0] < 2:
            return None

        path = path.copy()  # Don't mutate input
        window_size = min(7, path.shape[0] if path.shape[0] % 2 != 0 else path.shape[0] - 1)
        if window_size >= 3:
            try:
                path[:, 0] = savgol_filter(path[:, 0], window_length=window_size, polyorder=2)
                path[:, 1] = savgol_filter(path[:, 1], window_length=window_size, polyorder=2)
            except Exception:
                pass

        vel_vec = np.vstack([np.diff(path, axis=0) / dt, [0, 0]])
        vel_vec[-1] = vel_vec[-2]

        heading = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
        heading = np.unwrap(heading)

        acc_vec = np.vstack([np.diff(vel_vec, axis=0) / dt, [0, 0]])
        acc_vec[-1] = acc_vec[-2]
        acc_mag = np.linalg.norm(acc_vec, axis=1)

        jerk_vec = np.vstack([np.diff(acc_vec, axis=0) / dt, [0, 0]])
        jerk_vec[-1] = jerk_vec[-2]
        jerk_mag = np.linalg.norm(jerk_vec, axis=1)

        yaw_rate = np.diff(heading) / dt
        yaw_rate = np.append(yaw_rate, yaw_rate[-1])

        speed = np.linalg.norm(vel_vec, axis=1)
        stopped_mask = speed < STOPPED_SPEED_THRESHOLD
        acc_mag[stopped_mask] = 0.0
        jerk_mag[stopped_mask] = 0.0
        yaw_rate[stopped_mask] = 0.0

        return {
            "x": path[:, 0], "y": path[:, 1], "heading": heading,
            "velocity": vel_vec, "speed": speed,
            "acceleration": acc_mag, "jerk": jerk_mag, "yaw_rate": yaw_rate
        }

    def _get_ego_polygon(self, x: float, y: float, heading: float) -> Polygon:
        base_poly = Polygon(VEHICLE_POLYGON_COORDS)
        rotated_poly = affinity.rotate(base_poly, heading, origin=(0, 0), use_radians=True)
        return affinity.translate(rotated_poly, xoff=x, yoff=y)

    def _query_nearby_lanes(self, x: float, y: float, radius: float = 50.0) -> list:
        nearby = []
        for lane, poly in self.all_lanes:
            minx, miny, maxx, maxy = poly.bounds
            if (minx - radius < x < maxx + radius) and (miny - radius < y < maxy + radius):
                nearby.append(lane)
        return nearby

    def _get_best_lane(self, x: float, y: float, heading: float, nearby_lanes: list, speed: float = None):
        if not nearby_lanes:
            return None
        candidates = []
        is_stopped = (speed is not None) and (speed < 1.0)

        for lane in nearby_lanes:
            s, r = lane.local_coordinates(np.array([x, y]))
            s_clamped = max(0, min(s, lane.length))
            lane_heading_vec = lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            dist = lane.distance(np.array([x, y]))

            if is_stopped or (abs(diff) < (np.pi / 2)):
                candidates.append((lane, dist))

        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return None

    def _calculate_metrics(self, states: dict, horizon: int, frame_idx: int,
                           ego_state: Optional[Dict[str, Any]] = None,
                           n_execute: int = None) -> dict:
        metrics = {
            "nc": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
            "ep": 0.0, "ttc": 1.0, "lk": 1.0, "hc": 1.0, "ec": 1.0
        }

        # 1. DAC (ratio of waypoints inside drivable area)
        dac_valid = 0
        for t in range(horizon):
            nearby = self._query_nearby_lanes(states['x'][t], states['y'][t], radius=15.0)
            ego_poly = self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
            corners = [Point(c) for c in ego_poly.exterior.coords[:-1]]
            corners_valid = 0
            for corner in corners:
                for lane in nearby:
                    if lane.shapely_polygon.contains(corner):
                        corners_valid += 1
                        break
            if corners_valid >= 4:
                dac_valid += 1
        metrics['dac'] = dac_valid / horizon if horizon > 0 else 1.0

        # 2. NC & TTC
        tracks = self.scenario_data['tracks']
        for t in range(horizon):
            current_sim_frame = frame_idx + (t * self.gt_stride)
            if current_sim_frame >= self.scenario_data['length']:
                break

            ego_poly = self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
            ego_v = states['speed'][t]

            for obj_id, track in tracks.items():
                if obj_id == self.sdc_id:
                    continue
                if track['type'] not in ['VEHICLE', 'CYCLIST']:
                    continue

                pos_arr = track['state']['position']
                valid_arr = track['state']['valid']

                if current_sim_frame >= len(pos_arr) or not valid_arr[current_sim_frame]:
                    continue

                obj_pos_world = pos_arr[current_sim_frame][:2]
                ax, ay = self._to_sim_frame(obj_pos_world)
                ah = track['state']['heading'][current_sim_frame]
                agent_poly = self._get_ego_polygon(ax, ay, ah)

                if ego_poly.intersects(agent_poly):
                    is_at_fault = True
                    if ego_v < STOPPED_SPEED_THRESHOLD:
                        is_at_fault = False

                    dx, dy = ax - states['x'][t], ay - states['y'][t]
                    ego_hx, ego_hy = np.cos(states['heading'][t]), np.sin(states['heading'][t])
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
                nearby = self._query_nearby_lanes(states['x'][t], states['y'][t], radius=15.0)
                speed = states['speed'][t]
                best_lane = self._get_best_lane(states['x'][t], states['y'][t], states['heading'][t], nearby, speed=speed)

                if best_lane and speed > 1.0:
                    s, r = best_lane.local_coordinates(np.array([states['x'][t], states['y'][t]]))
                    s_clamped = max(0, min(s, best_lane.length))
                    lane_heading_vec = best_lane.heading_at(s_clamped)
                    lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
                    diff = abs(states['heading'][t] - lane_heading)
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi

                    if abs(diff) > np.pi / 2:
                        metrics['ddc'] = 0.0
                        break

        # 4. TLC
        for t in range(horizon):
            current_sim_frame = frame_idx + (t * self.gt_stride)
            if current_sim_frame >= self.scenario_data['length']:
                break

            red_lane_ids = set()
            for lane_id, tl_data in self.traffic_lights.items():
                s_list = tl_data['state']['object_state']
                if current_sim_frame < len(s_list) and s_list[current_sim_frame] == "LANE_STATE_STOP":
                    red_lane_ids.add(str(lane_id))
            if not red_lane_ids:
                continue

            ego_poly = self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
            nearby = self._query_nearby_lanes(states['x'][t], states['y'][t], radius=5.0)

            for lane in nearby:
                curr_id = str(lane.index)
                if isinstance(lane.index, (tuple, list)) and len(lane.index) > 2:
                    curr_id = str(lane.index[2])

                is_red = curr_id in red_lane_ids

                if is_red and states['speed'][t] > 1.0 and lane.shapely_polygon.intersects(ego_poly):
                    s, _ = lane.local_coordinates(np.array([states['x'][t], states['y'][t]]))
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
                nearby = self._query_nearby_lanes(states['x'][t], states['y'][t], radius=10.0)
                speed = states['speed'][t]
                best_lane = self._get_best_lane(states['x'][t], states['y'][t], states['heading'][t], nearby, speed=speed)
                if best_lane:
                    _, lat = best_lane.local_coordinates(np.array([states['x'][t], states['y'][t]]))
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
            # Current actual dynamics from simulation
            ego_acc = np.linalg.norm(ego_state['acceleration'][:2])
            ego_yr = abs(ego_state.get('angular_velocity', np.zeros(3))[2])

            # Build execution sequence: [current_ego, traj[0], ..., traj[n_execute-1]]
            n_exec = min(n_execute if n_execute is not None else horizon, len(states['acceleration']))
            acc_seq = np.concatenate([[ego_acc], states['acceleration'][:n_exec]])
            yr_seq = np.concatenate([[ego_yr], np.abs(states['yaw_rate'][:n_exec])])

            # Ratio of comfortable transitions over the execution window
            diff_acc = np.abs(np.diff(acc_seq))
            diff_yr = np.abs(np.diff(yr_seq))
            comfortable = (diff_acc <= EC_ACCEL_THRESH) & (diff_yr <= EC_YAW_RATE_THRESH)
            metrics['ec'] = float(comfortable.sum()) / len(comfortable) if len(comfortable) > 0 else 1.0

        # 8. Progress
        dist = np.linalg.norm(np.array([states['x'][-1], states['y'][-1]]) - np.array([states['x'][0], states['y'][0]]))
        metrics['ep'] = min(dist / 30.0, 1.0)

        return metrics

    def _score_single_candidate(self, traj_world: np.ndarray, frame_idx: int,
                                ego_state: Optional[Dict[str, Any]] = None,
                                n_execute: int = None) -> float:
        """
        Score a single candidate trajectory in world frame.
        Returns the EPDMS score (float). No human filtering, no prints.
        """
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if not sdc_track['state']['valid'][frame_idx]:
            return 0.0
        current_pos = sdc_track['state']['position'][frame_idx][:2]

        # Build full path: current position + predicted waypoints
        full_path_world = np.vstack([current_pos, traj_world])
        full_path_sim = self._to_sim_frame(full_path_world)

        states = self._get_trajectory_states(full_path_sim, self.planner_dt)
        if states is None:
            return 0.0

        metrics = self._calculate_metrics(states, len(traj_world), frame_idx,
                                          ego_state=ego_state, n_execute=n_execute)

        multi_prod = metrics['nc'] * metrics['dac'] * metrics['ddc'] * metrics['tlc']
        weighted_sum = (W_PROGRESS * metrics['ep'] + W_TTC * metrics['ttc'] +
                        W_LANE_KEEPING * metrics['lk'] + W_HISTORY_COMFORT * metrics['hc'] +
                        W_EXTENDED_COMFORT * metrics['ec'])
        weighted_avg = weighted_sum / (W_PROGRESS + W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT)
        return multi_prod * weighted_avg

    def _ego_to_world(self, candidates_np: np.ndarray, ego_state: Dict[str, Any]) -> np.ndarray:
        """
        Transform candidates from ego frame to world frame (vectorized).

        Args:
            candidates_np: (N, 8, 3) array in ego frame (x_fwd, y_lat, heading)
            ego_state: dict with 'position' and 'heading'

        Returns:
            (N, 8, 2) array in world frame (x, y)
        """
        ego_x, ego_y = ego_state['position'][:2]
        ego_heading = ego_state['heading']
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)

        x_fwd = candidates_np[:, :, 0]  # (N, 8)
        y_lat = candidates_np[:, :, 1]  # (N, 8)

        world_x = ego_x + x_fwd * cos_h - y_lat * sin_h
        world_y = ego_y + x_fwd * sin_h + y_lat * cos_h

        return np.stack([world_x, world_y], axis=-1)  # (N, 8, 2)

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select best trajectory from candidates using EPDMS scoring.

        Args:
            model_output: dict with "all_candidates" (B, N, 8, 3)
            **kwargs: must include ego_state and frame_idx

        Returns:
            dict with trajectory (1, 8, 3), scores (1, N), best_idx (1,)
        """
        if not self._initialized:
            raise RuntimeError(
                "EPDMSTrajectoryScorer not initialized. "
                "Ensure base_evaluator calls scorer.initialize(scenario_data, env)."
            )

        ego_state = kwargs['ego_state']
        frame_idx = kwargs['frame_idx']

        # Compute number of waypoints that will be executed before next replan
        if self.prev_frame_idx is not None:
            replan_interval_s = (frame_idx - self.prev_frame_idx) * self.scenario_dt
            n_execute = max(1, int(round(replan_interval_s / self.planner_dt)))
        else:
            n_execute = None  # First call — use full horizon

        all_candidates = model_output["all_candidates"]  # (B, N, 8, 3) tensor
        candidates_np = all_candidates[0].cpu().numpy()  # (N, 8, 3)
        N = candidates_np.shape[0]

        # Transform all candidates to world frame
        candidates_world = self._ego_to_world(candidates_np, ego_state)  # (N, 8, 2)

        # Score each candidate
        scores = np.zeros(N)
        for i in range(N):
            scores[i] = self._score_single_candidate(
                candidates_world[i], frame_idx,
                ego_state=ego_state, n_execute=n_execute
            )

        best_idx = int(np.argmax(scores))
        self.prev_frame_idx = frame_idx

        print(f"[EPDMS TrajScorer] Frame {frame_idx}: best_idx={best_idx}/{N}, "
              f"score={scores[best_idx]:.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, min={scores.min():.4f}")

        best_traj = all_candidates[0, best_idx]  # (8, 3) tensor
        return {
            "trajectory": best_traj.unsqueeze(0),  # (1, 8, 3)
            "scores": torch.from_numpy(scores).unsqueeze(0).float(),  # (1, N)
            "best_idx": torch.tensor([best_idx]),  # (1,)
        }
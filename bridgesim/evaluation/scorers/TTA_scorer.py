"""
Test-Time Adaptation (TTA) trajectory scorer with plan continuity.

This scorer extends the ground-truth-based EPDMS scoring (see GT_scorer.py)
with two test-time adaptation mechanisms:

1. Plan continuity: at each replan step, the remaining segment of the previous
   best trajectory is re-scored at the current ego position and compared against
   all new candidates. The old plan is retained if it remains competitive, reducing
   unnecessary replanning and improving temporal smoothness.

2. Collision scoring via agent extrapolation: instead of only checking collisions
   against ground-truth agent positions (which may not match closed-loop
   simulation), future agent positions are linearly extrapolated from their
   current velocities. This makes the collision metric (COL) more robust to
   distribution shift between log-replay and the ego's actual trajectory.

Selection logic:
  - Score all N new candidates over the full planning horizon.
  - Re-score the remaining segment of the previous trajectory at the current
    replan time.
  - If the old trajectory has fewer than 2×K waypoints remaining, force a
    switch to the best new candidate.
  - Otherwise, keep the old trajectory if its re-scored reward >= best new score.

Scoring formula:
  (COL × DAC × DDC × TLC) × (5×EP + 2×LK + 2×HC + 2×EC) / 11
"""

import math
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List
from shapely.geometry import Polygon, Point
from scipy.signal import savgol_filter

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer
from bridgesim.evaluation.utils.ttc_2d import compute_ttc_2d

# --- Constants ---
VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH = 1.852

VEHICLE_POLYGON_COORDS = np.array([
    [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2],
    [VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2]
])

VEHICLE_HALF_DIAG = math.sqrt((VEHICLE_LENGTH / 2) ** 2 + (VEHICLE_WIDTH / 2) ** 2)

W_PROGRESS = 5.0
W_LANE_KEEPING = 2.0
W_HISTORY_COMFORT = 2.0
W_EXTENDED_COMFORT = 2.0
W_TOTAL = W_PROGRESS + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT

LANE_DEVIATION_LIMIT = 0.5
LANE_KEEPING_WINDOW = 2.0
STOPPED_SPEED_THRESHOLD = 5e-2
MAX_ACCEL, MAX_JERK, MAX_YAW_RATE = 4.89, 8.37, 0.95
EC_ACCEL_THRESH, EC_YAW_RATE_THRESH = 0.7, 0.1
TTC_SAFE_THRESHOLD = 3.0    # TTC above this -> score 1.0
TTC_DANGER_THRESHOLD = 0.5  # TTC below this -> score 0.0


class TTAScorer(BaseTrajectoryScorer):
    """
    TTA trajectory scorer with plan continuity and agent-extrapolation-based
    collision scoring.

    At each replan step, new candidates are scored over the full planning
    horizon. The best new candidate is then compared against a freshly
    re-scored remaining segment of the previous trajectory. The previous plan
    is retained when it still has sufficient remaining waypoints and its score
    remains competitive, providing smoother behavior than always replanning.

    Collision detection uses linear extrapolation of agent future positions,
    making the COL metric adaptive to the ego's actual closed-loop trajectory
    rather than relying solely on ground-truth log positions.

    See module docstring for full details on the selection logic and scoring formula.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self._initialized = False
        self.scenario_data = None
        self.env = None
        self.sdc_id = None
        self.all_lanes = []
        self.all_lane_bounds = None
        self.traffic_lights = {}
        self.world_to_sim_offset = np.array([0.0, 0.0])
        self.scenario_dt = 0.1
        self.planner_dt = 0.5
        self.gt_stride = 5
        self.prev_frame_idx = None

        # Cached previous best trajectory info (plan continuity)
        self._prev_best_interp = None       # (H*gt_stride, 3) interpolated sim-frame waypoints
        self._prev_best_consumed = 0        # int: how many sim frames consumed so far
        self._prev_best_ego_pos = None      # (2,): ego world position at prediction frame
        self._prev_best_ego_heading = None  # float: ego world heading at prediction frame

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
            else:
                self.scenario_dt = float(timestep)
        self.planner_dt = 0.5
        if self.scenario_dt > 0:
            self.gt_stride = int(round(self.planner_dt / self.scenario_dt))
        else:
            self.gt_stride = 5

        # Map extraction + pre-compute lane bounds
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

        if self.all_lanes:
            bounds = np.array([poly.bounds for _, poly in self.all_lanes])
            self.all_lane_bounds = bounds
        else:
            self.all_lane_bounds = np.empty((0, 4))

        # Coordinate calibration (world -> sim)
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if sdc_track['state']['valid'][0]:
            start_pos_world = sdc_track['state']['position'][0][:2]
            start_pos_sim = self.env.agent.position
            self.world_to_sim_offset = np.array(start_pos_sim) - np.array(start_pos_world)
            print(f"[EPDMS AdaptiveCol] Calibrated World->Sim Offset: {self.world_to_sim_offset}")
        else:
            self.world_to_sim_offset = np.array([0.0, 0.0])
            print("[EPDMS AdaptiveCol] Could not calibrate coordinates (Frame 0 invalid).")

        self.prev_frame_idx = None
        self._prev_best_interp = None
        self._prev_best_consumed = 0
        self._prev_best_ego_pos = None
        self._prev_best_ego_heading = None
        self._initialized = True
        print(f"[EPDMS AdaptiveCol] Initialized with {len(self.all_lanes)} lanes, "
              f"gamma={self.gamma} (COL + plan continuity)")

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    def _query_nearby_lanes_vec(self, x: float, y: float, radius: float) -> List[int]:
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
    # Pre-computation
    # ---------------------------------------------------------------

    def _precompute_red_lanes(self, frame_idx: int, horizon: int):
        """Pre-compute per-timestep traffic light red lane IDs."""
        scenario_length = self.scenario_data['length']
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
        return red_lanes_per_t

    def _precompute_agent_futures_linear(self, frame_idx: int, horizon: int, max_agents: int = 10):
        """Pre-compute per-timestep agent polygons via linear extrapolation.

        Only the closest `max_agents` agents (by distance to ego) are kept.
        """
        tracks = self.scenario_data['tracks']
        agents_per_t = [[] for _ in range(horizon)]

        # Get ego position for distance filtering
        sdc_track = tracks[self.sdc_id]
        ego_pos = sdc_track['state']['position'][frame_idx][:2]

        # First pass: collect valid agents with their distance to ego
        agent_entries = []  # (distance, obj_id)
        for obj_id, track in tracks.items():
            if obj_id == self.sdc_id:
                continue
            if track['type'] not in ['VEHICLE', 'CYCLIST']:
                continue
            pos_arr = track['state']['position']
            valid_arr = track['state']['valid']
            if frame_idx >= len(pos_arr) or not valid_arr[frame_idx]:
                continue
            pos = pos_arr[frame_idx][:2]
            dist = float(np.sum((pos - ego_pos) ** 2))
            agent_entries.append((dist, obj_id))

        # Keep only the closest max_agents
        agent_entries.sort(key=lambda x: x[0])
        selected_ids = {obj_id for _, obj_id in agent_entries[:max_agents]}

        for obj_id in selected_ids:
            track = tracks[obj_id]

            pos_arr = track['state']['position']
            valid_arr = track['state']['valid']
            heading_arr = track['state']['heading']

            pos = pos_arr[frame_idx][:2]
            heading = float(heading_arr[frame_idx])

            # Compute velocity
            if 'velocity' in track['state'] and frame_idx < len(track['state']['velocity']):
                velocity = track['state']['velocity'][frame_idx][:2].copy()
            elif frame_idx > 0 and valid_arr[frame_idx - 1]:
                prev_pos = pos_arr[frame_idx - 1][:2]
                velocity = (pos - prev_pos) / self.scenario_dt
            else:
                velocity = np.zeros(2)

            # Compute acceleration
            if 'velocity' in track['state'] and frame_idx > 0 and frame_idx < len(track['state']['velocity']) and valid_arr[frame_idx - 1]:
                prev_velocity = track['state']['velocity'][frame_idx - 1][:2].copy()
                acceleration = (velocity - prev_velocity) / self.scenario_dt
            elif frame_idx > 1 and valid_arr[frame_idx - 1] and valid_arr[frame_idx - 2]:
                prev_pos = pos_arr[frame_idx - 1][:2]
                prev_prev_pos = pos_arr[frame_idx - 2][:2]
                prev_velocity = (prev_pos - prev_prev_pos) / self.scenario_dt
                curr_velocity = (pos - prev_pos) / self.scenario_dt
                acceleration = (curr_velocity - prev_velocity) / self.scenario_dt
            else:
                acceleration = np.zeros(2)

            # Compute heading rate
            if frame_idx > 0 and valid_arr[frame_idx - 1]:
                prev_heading = float(heading_arr[frame_idx - 1])
                heading_rate = ((heading - prev_heading + math.pi) % (2 * math.pi) - math.pi) / self.scenario_dt
            else:
                heading_rate = 0.0

            agent_vel_sim = velocity.copy()
            agent_accel_sim = acceleration.copy()

            # Extrapolate for each horizon timestep
            for t in range(horizon):
                dt_future = t * self.planner_dt
                future_pos = pos + velocity * dt_future + 0.5 * acceleration * dt_future ** 2
                future_heading = heading + heading_rate * dt_future

                ax, ay = future_pos + self.world_to_sim_offset
                agents_per_t[t].append((float(ax), float(ay), agent_vel_sim, agent_accel_sim, float(future_heading)))

        return agents_per_t

    # ---------------------------------------------------------------
    # Batched trajectory state computation
    # ---------------------------------------------------------------

    def _get_all_trajectory_states(self, all_paths: np.ndarray, dt: float):
        """Vectorized trajectory states for all N candidates."""
        N, T, _ = all_paths.shape
        paths = all_paths.copy()

        window_size = min(7, T if T % 2 != 0 else T - 1)
        if window_size >= 3:
            for i in range(N):
                try:
                    paths[i, :, 0] = savgol_filter(paths[i, :, 0], window_length=window_size, polyorder=2)
                    paths[i, :, 1] = savgol_filter(paths[i, :, 1], window_length=window_size, polyorder=2)
                except Exception:
                    pass

        dpath = np.diff(paths, axis=1) / dt
        vel_vec = np.concatenate([dpath, dpath[:, -1:, :]], axis=1)

        heading = np.arctan2(vel_vec[:, :, 1], vel_vec[:, :, 0])
        heading = np.unwrap(heading, axis=1)

        dacc = np.diff(vel_vec, axis=1) / dt
        acc_vec = np.concatenate([dacc, dacc[:, -1:, :]], axis=1)
        acc_mag = np.linalg.norm(acc_vec, axis=2)

        djerk = np.diff(acc_vec, axis=1) / dt
        jerk_vec = np.concatenate([djerk, djerk[:, -1:, :]], axis=1)
        jerk_mag = np.linalg.norm(jerk_vec, axis=2)

        yaw_rate_raw = np.diff(heading, axis=1) / dt
        yaw_rate = np.concatenate([yaw_rate_raw, yaw_rate_raw[:, -1:]], axis=1)

        speed = np.linalg.norm(vel_vec, axis=2)
        stopped_mask = speed < STOPPED_SPEED_THRESHOLD
        acc_mag[stopped_mask] = 0.0
        jerk_mag[stopped_mask] = 0.0
        yaw_rate[stopped_mask] = 0.0

        return {
            "x": paths[:, :, 0],
            "y": paths[:, :, 1],
            "heading": heading,
            "speed": speed,
            "acceleration": acc_mag,
            "jerk": jerk_mag,
            "yaw_rate": yaw_rate,
        }

    # ---------------------------------------------------------------
    # Per-candidate metric evaluation (with COL)
    # ---------------------------------------------------------------

    def _calculate_metrics(self, states: dict, horizon: int, frame_idx: int,
                           ego_state: Optional[Dict[str, Any]] = None,
                           n_execute: int = None,
                           agents_per_t: list = None,
                           red_lanes_per_t: list = None) -> dict:
        metrics = {
            "col": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
            "ep": 0.0, "lk": 1.0, "hc": 1.0, "ec": 1.0
        }

        ego_polys = [self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
                     for t in range(horizon)]
        nearby_cache = {}

        def get_nearby(t, radius):
            key = (t, radius)
            if key not in nearby_cache:
                nearby_cache[key] = self._query_nearby_lanes_vec(
                    states['x'][t], states['y'][t], radius)
            return nearby_cache[key]

        # 1. DAC
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

        # 2. COL (discounted per-timestep TTC-based collision score, γ^(t*dt) weighting)
        #    Uses 2D bounding-box TTC (Jiao 2022) for geometry-aware collision detection.
        #    Per-timestep score mapped linearly: TTC<=0.5s→0.0, TTC>=3.0s→1.0
        if agents_per_t is not None:
            n_agents_total = sum(len(agents_per_t[t]) for t in range(min(horizon, len(agents_per_t))))
            col_num, col_den = 0.0, 0.0
            ttc_range = TTC_SAFE_THRESHOLD - TTC_DANGER_THRESHOLD
            for t in range(horizon):
                w = self.gamma ** (t * self.planner_dt)
                step_score = 1.0  # safe
                ego_v = states['speed'][t]
                ego_hx = math.cos(states['heading'][t])
                ego_hy = math.sin(states['heading'][t])
                ego_vel = np.array([ego_hx * ego_v, ego_hy * ego_v])

                nearby_rows = []
                nearby_agents = []
                for ax, ay, agent_vel, agent_accel, agent_heading in agents_per_t[t]:
                    nearby_rows.append({
                        'x_i': states['x'][t], 'y_i': states['y'][t],
                        'vx_i': float(ego_vel[0]), 'vy_i': float(ego_vel[1]),
                        'hx_i': ego_hx, 'hy_i': ego_hy,
                        'length_i': VEHICLE_LENGTH, 'width_i': VEHICLE_WIDTH,
                        'x_j': ax, 'y_j': ay,
                        'vx_j': float(agent_vel[0]), 'vy_j': float(agent_vel[1]),
                        'hx_j': math.cos(agent_heading), 'hy_j': math.sin(agent_heading),
                        'length_j': VEHICLE_LENGTH, 'width_j': VEHICLE_WIDTH,
                    })
                    nearby_agents.append((ax, ay))

                if nearby_rows:
                    df = pd.DataFrame(nearby_rows)
                    ttc_values = compute_ttc_2d(df, 'values')
                    for idx, (ax, ay) in enumerate(nearby_agents):
                        ttc = float(ttc_values[idx])
                        if ttc >= TTC_SAFE_THRESHOLD:
                            ttc_score = 1.0
                        elif ttc <= TTC_DANGER_THRESHOLD:
                            ttc_score = 0.0
                        else:
                            ttc_score = (ttc - TTC_DANGER_THRESHOLD) / ttc_range

                        if ttc_score < step_score:
                            at_fault = True
                            if ego_v < STOPPED_SPEED_THRESHOLD:
                                at_fault = False
                            dx, dy = ax - states['x'][t], ay - states['y'][t]
                            longitudinal_dist = dx * ego_hx + dy * ego_hy
                            if longitudinal_dist < -1.0:
                                at_fault = False

                            if at_fault:
                                step_score = min(step_score, ttc_score)
                            else:
                                step_score = min(step_score, max(ttc_score, 0.5))


                col_den += w
                col_num += w * step_score
            metrics['col'] = col_num / col_den if col_den > 0 else 1.0

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

        # 4. TLC
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

        # 7. EC
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
        metrics['ep'] = min(dist / 3.0, 1.0)

        return metrics

    # ---------------------------------------------------------------
    # Score computation helper
    # ---------------------------------------------------------------

    @staticmethod
    def _metrics_to_score(metrics: dict) -> float:
        """Convert metrics dict to scalar score."""
        multi_prod = metrics['col'] * metrics['dac'] * metrics['ddc'] * metrics['tlc']
        weighted_sum = (W_PROGRESS * metrics['ep'] +
                        W_LANE_KEEPING * metrics['lk'] +
                        W_HISTORY_COMFORT * metrics['hc'] +
                        W_EXTENDED_COMFORT * metrics['ec'])
        return multi_prod * weighted_sum / W_TOTAL

    def _score_single_world_traj(
        self,
        traj_world: np.ndarray,
        frame_idx: int,
        ego_state: Dict[str, Any],
        n_execute: Optional[int],
        agents_per_t: list = None,
    ) -> float:
        """
        Score one candidate trajectory in world frame at current replan time.

        Args:
            traj_world: (T, 2) future waypoints in world frame (no current point).
            frame_idx: Current scenario frame index.
            ego_state: Current ego state dict.
            n_execute: Planner steps to execute before next replan.
            agents_per_t: Pre-computed agent polygons per timestep for COL scoring.
        """
        horizon = len(traj_world)
        if horizon <= 0:
            return 0.0

        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        current_pos = sdc_track['state']['position'][frame_idx][:2]

        ego_pos_tiled = np.broadcast_to(current_pos[None, :], (1, 1, 2))
        traj_world_batched = traj_world[None, :, :]
        all_paths_world = np.concatenate([ego_pos_tiled, traj_world_batched], axis=1)
        all_paths_sim = all_paths_world + self.world_to_sim_offset

        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)
        states_0 = {k: all_states[k][0] for k in all_states}
        red_lanes_per_t = self._precompute_red_lanes(frame_idx, horizon)

        # Trim agents_per_t to match this trajectory's horizon if needed
        trimmed_agents = None
        if agents_per_t is not None:
            trimmed_agents = agents_per_t[:horizon]

        metrics = self._calculate_metrics(
            states_0,
            horizon,
            frame_idx,
            ego_state=ego_state,
            n_execute=n_execute,
            agents_per_t=trimmed_agents,
            red_lanes_per_t=red_lanes_per_t,
        )
        return self._metrics_to_score(metrics)

    # ---------------------------------------------------------------
    # Interpolation helper
    # ---------------------------------------------------------------

    def _interpolate_raw_to_sim(self, raw: np.ndarray) -> np.ndarray:
        """
        Interpolate raw planner waypoints (H, 3) at planner_dt intervals
        to sim_dt intervals -> (H * gt_stride, 3).
        """
        H = raw.shape[0]
        origin = np.zeros((1, 3))
        with_origin = np.concatenate([origin, raw], axis=0)

        t_orig = np.arange(H + 1) * self.planner_dt
        t_max = t_orig[-1]
        t_interp = np.arange(self.scenario_dt, t_max + self.scenario_dt / 2, self.scenario_dt)

        interp_x = np.interp(t_interp, t_orig, with_origin[:, 0])
        interp_y = np.interp(t_interp, t_orig, with_origin[:, 1])
        interp_h = np.interp(t_interp, t_orig, with_origin[:, 2])

        return np.stack([interp_x, interp_y, interp_h], axis=-1)

    # ---------------------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------------------

    def _ego_to_world(self, candidates_np: np.ndarray, ego_state: Dict[str, Any]) -> np.ndarray:
        """Vectorized ego->world transform. (N, T, 3) -> (N, T, 2)."""
        ego_x, ego_y = ego_state['position'][:2]
        ego_heading = ego_state['heading']
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
        x_fwd = candidates_np[:, :, 0]
        y_lat = candidates_np[:, :, 1]
        world_x = ego_x + x_fwd * cos_h - y_lat * sin_h
        world_y = ego_y + x_fwd * sin_h + y_lat * cos_h
        return np.stack([world_x, world_y], axis=-1)

    def _reproject_from_old_ego_to_current_ego(
        self,
        traj_old_ego: np.ndarray,
        old_ego_pos: np.ndarray,
        old_ego_heading: float,
        current_ego_pos: np.ndarray,
        current_ego_heading: float,
    ) -> np.ndarray:
        """
        Reproject trajectory from previous prediction ego frame to current ego frame.

        Input/Output trajectory convention follows model output: [:, 0]=forward, [:, 1]=lateral.
        """
        if traj_old_ego.size == 0:
            return traj_old_ego

        traj_curr_ego = traj_old_ego.copy()

        old_forward = traj_old_ego[:, 0]
        old_lateral = traj_old_ego[:, 1]
        old_cos, old_sin = math.cos(old_ego_heading), math.sin(old_ego_heading)

        # old ego -> world
        world_x = old_ego_pos[0] + old_forward * old_cos - old_lateral * old_sin
        world_y = old_ego_pos[1] + old_forward * old_sin + old_lateral * old_cos

        # world -> current ego
        dx = world_x - current_ego_pos[0]
        dy = world_y - current_ego_pos[1]
        cur_cos, cur_sin = math.cos(current_ego_heading), math.sin(current_ego_heading)
        traj_curr_ego[:, 0] = dx * cur_cos + dy * cur_sin
        traj_curr_ego[:, 1] = -dx * cur_sin + dy * cur_cos

        if traj_curr_ego.shape[1] > 2:
            heading_delta = old_ego_heading - current_ego_heading
            traj_curr_ego[:, 2] = np.arctan2(
                np.sin(traj_old_ego[:, 2] + heading_delta),
                np.cos(traj_old_ego[:, 2] + heading_delta),
            )

        return traj_curr_ego

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("TTAScorer not initialized.")

        t_start = time.time()

        ego_state = kwargs['ego_state']
        frame_idx = kwargs['frame_idx']

        # Compute replan interval in sim frames and planner steps
        if self.prev_frame_idx is not None:
            replan_sim_frames = frame_idx - self.prev_frame_idx
            replan_interval_s = replan_sim_frames * self.scenario_dt
            n_execute = max(1, int(round(replan_interval_s / self.planner_dt)))
        else:
            replan_sim_frames = None
            n_execute = None

        all_candidates = model_output["all_candidates"]  # (B, N, 8, 3) tensor
        candidates_np = all_candidates[0].cpu().numpy()   # (N, 8, 3)
        N = candidates_np.shape[0]
        horizon = candidates_np.shape[1]  # 8

        # --- Pre-compute shared data ---
        red_lanes_per_t = self._precompute_red_lanes(frame_idx, horizon)
        agents_per_t = self._precompute_agent_futures_linear(frame_idx, horizon)
        n_agents_check = sum(len(agents_per_t[t]) for t in range(horizon))

        # --- Transform new candidates to world frame ---
        candidates_world = self._ego_to_world(candidates_np, ego_state)  # (N, 8, 2)

        # --- Validate current frame ---
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if not sdc_track['state']['valid'][frame_idx]:
            best_traj = all_candidates[0, 0]
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.zeros(1, N),
                "best_idx": torch.tensor([0]),
            }

        current_pos = sdc_track['state']['position'][frame_idx][:2]

        # --- Score all new candidates (full horizon) ---
        ego_pos_tiled = np.broadcast_to(current_pos[None, None, :], (N, 1, 2))
        all_paths_world = np.concatenate([ego_pos_tiled, candidates_world], axis=1)
        all_paths_sim = all_paths_world + self.world_to_sim_offset

        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)

        scores = np.zeros(N)
        for i in range(N):
            states_i = {k: all_states[k][i] for k in all_states}
            metrics = self._calculate_metrics(
                states_i, horizon, frame_idx,
                ego_state=ego_state, n_execute=n_execute,
                agents_per_t=agents_per_t,
                red_lanes_per_t=red_lanes_per_t,
            )
            scores[i] = self._metrics_to_score(metrics)

        # Step 1: Select top-K new candidates by full-horizon score
        top_k = min(5, N)
        top_new_indices = np.argsort(scores)[::-1][:top_k]
        best_new_idx = top_new_indices[0]
        best_new_score = scores[best_new_idx]

        # --- Compare against previous trajectory (re-score remaining segment) ---
        keep_prev = False
        shifted_planner = None
        prev_remaining_score = None
        best_new_trimmed_score = None

        if (self._prev_best_interp is not None
                and self._prev_best_ego_pos is not None
                and self._prev_best_ego_heading is not None
                and replan_sim_frames is not None):
            new_consumed = self._prev_best_consumed + replan_sim_frames
            total_sim = len(self._prev_best_interp)
            remaining_sim = total_sim - new_consumed

            if remaining_sim > 0:
                shifted_interp = self._prev_best_interp[new_consumed:]

                planner_indices = np.arange(self.gt_stride - 1, len(shifted_interp), self.gt_stride)

                if len(planner_indices) > 0:
                    shifted_planner = shifted_interp[planner_indices]
                    n_planner = len(shifted_planner)

                    if n_planner >= 2 * n_execute:
                        shifted_planner_current = self._reproject_from_old_ego_to_current_ego(
                            shifted_planner,
                            self._prev_best_ego_pos,
                            self._prev_best_ego_heading,
                            np.array(ego_state['position'][:2]),
                            float(ego_state['heading']),
                        )
                        shifted_world = self._ego_to_world(
                            shifted_planner_current[None, :, :], ego_state
                        )[0]
                        prev_remaining_score = self._score_single_world_traj(
                            shifted_world,
                            frame_idx,
                            ego_state,
                            n_execute,
                            agents_per_t=agents_per_t,
                        )

                        # Step 2: Re-score top-K new candidates trimmed to same
                        # horizon as old trajectory for fair comparison
                        best_new_trimmed_score = -1.0
                        for idx in top_new_indices:
                            trimmed_world = candidates_world[idx][:n_planner]
                            trimmed_score = self._score_single_world_traj(
                                trimmed_world,
                                frame_idx,
                                ego_state,
                                n_execute,
                                agents_per_t=agents_per_t,
                            )
                            if trimmed_score > best_new_trimmed_score:
                                best_new_trimmed_score = trimmed_score
                                best_new_idx = idx

                        if prev_remaining_score >= best_new_trimmed_score:
                            keep_prev = True

        # --- Build result ---
        if keep_prev:
            # shifted_planner_current already computed above
            best_traj = torch.from_numpy(shifted_planner_current).float()

            self._prev_best_consumed = self._prev_best_consumed + replan_sim_frames

            elapsed = time.time() - t_start
            n_wp = len(shifted_planner_current)
            print(f"[EPDMS AdaptiveCol] Frame {frame_idx}: KEPT prev_best, "
                  f"prev_remaining={prev_remaining_score:.4f}, "
                  f"best_new_trimmed={best_new_trimmed_score:.4f}, "
                  f"best_new_full={best_new_score:.4f}, "
                  f"remaining_wp={n_wp}, consumed_sim={self._prev_best_consumed}, "
                  f"replan_frames={replan_sim_frames}, time={elapsed:.2f}s")

            self.prev_frame_idx = frame_idx
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.from_numpy(scores).unsqueeze(0).float(),
                "best_idx": torch.tensor([-1]),
            }
        else:
            best_traj = all_candidates[0, best_new_idx]

            raw = candidates_np[best_new_idx]
            self._prev_best_interp = self._interpolate_raw_to_sim(raw)
            self._prev_best_consumed = 0
            self._prev_best_ego_pos = np.array(ego_state['position'][:2], dtype=np.float64).copy()
            self._prev_best_ego_heading = float(ego_state['heading'])

            elapsed = time.time() - t_start
            print(f"[EPDMS AdaptiveCol] Frame {frame_idx}: SWITCH to new idx={best_new_idx}/{N}, "
                  f"score={scores[best_new_idx]:.4f}, "
                  f"K={n_execute}, replan_frames={replan_sim_frames}, time={elapsed:.2f}s")

            self.prev_frame_idx = frame_idx
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.from_numpy(scores).unsqueeze(0).float(),
                "best_idx": torch.tensor([best_new_idx]),
            }

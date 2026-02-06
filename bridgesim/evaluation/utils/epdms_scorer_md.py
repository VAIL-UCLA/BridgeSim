import numpy as np
import math
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely import affinity
from typing import Dict, List, Optional, Tuple
from scipy.signal import savgol_filter

# --- Constants ---
VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH = 1.852

VEHICLE_POLYGON_COORDS = np.array([
    [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2],
    [VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2]
])

# Weights & Thresholds
W_PROGRESS, W_TTC, W_LANE_KEEPING, W_HISTORY_COMFORT, W_EXTENDED_COMFORT = 5.0, 5.0, 2.0, 2.0, 2.0
LANE_DEVIATION_LIMIT = 0.5  
LANE_KEEPING_WINDOW = 2.0   
TTC_HORIZON = 1.0
STOPPED_SPEED_THRESHOLD = 5e-2 # 0.05 m/s
MAX_ACCEL, MAX_JERK, MAX_YAW_RATE = 4.89, 8.37, 0.95
EC_ACCEL_THRESH, EC_JERK_THRESH, EC_YAW_RATE_THRESH = 0.7, 0.5, 0.1

class EPDMSScorer:
    def __init__(self, scenario_data: dict, env):
        self.scenario_data = scenario_data
        self.env = env
        self.sdc_id = scenario_data['metadata']['sdc_id']
        
        # --- Time Step Alignment ---
        self.scenario_dt = 0.1
        if 'metadata' in scenario_data and 'timestep' in scenario_data['metadata']:
            timestep = scenario_data['metadata']['timestep']
            # Handle numpy array or scalar
            import numpy as np
            if isinstance(timestep, np.ndarray):
                if timestep.size == 1:
                    self.scenario_dt = float(timestep.item())
                elif timestep.size > 1:
                    # Take first element or compute mean difference
                    self.scenario_dt = float(timestep.flat[0]) if timestep.flat[0] > 0 else 0.1
            else:
                self.scenario_dt = float(timestep)
        self.planner_dt = 0.5
        # Ensure we don't divide by zero
        if self.scenario_dt > 0:
            self.gt_stride = int(round(self.planner_dt / self.scenario_dt))
        else:
            self.gt_stride = 5  # Default fallback
        
        # --- Map Extraction ---
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
        
        # --- COORDINATE CALIBRATION (World -> Sim) ---
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if sdc_track['state']['valid'][0]:
            start_pos_world = sdc_track['state']['position'][0][:2]
            start_pos_sim = self.env.agent.position
            self.world_to_sim_offset = np.array(start_pos_sim) - np.array(start_pos_world)
            print(f"[EPDMS INIT] Calibrated World->Sim Offset: {self.world_to_sim_offset}")
        else:
            self.world_to_sim_offset = np.array([0.0, 0.0])
            print("[EPDMS WARN] Could not calibrate coordinates (Frame 0 invalid).")

        self.prev_plan_states = None
        self.prev_human_states = None

        # --- Live Scoring State (for closed-loop evaluation) ---
        # Motion history for comfort metrics
        self.prev_position = None
        self.prev_velocity = None
        self.prev_heading = None
        self.prev_acceleration = None
        self.prev_jerk = None
        self.prev_yaw_rate = None

        # Lane keeping state (tracks consecutive deviations across frames)
        self.consecutive_lane_deviation = 0

    def _to_sim_frame(self, points):
        """Transforms points from World Frame to Simulation Frame."""
        return points + self.world_to_sim_offset

    def _get_trajectory_states(self, path: np.ndarray, dt: float) -> Optional[Dict[str, np.ndarray]]:
        if path.shape[0] < 2: return None 
        
        # --- RESTORED: Savitzky-Golay Smoothing ---
        # Smooth position data to reduce noise in derivatives
        window_size = min(7, path.shape[0] if path.shape[0] % 2 != 0 else path.shape[0] - 1)
        if window_size >= 3:
            try:
                # Apply filter to X and Y independently
                path[:, 0] = savgol_filter(path[:, 0], window_length=window_size, polyorder=2)
                path[:, 1] = savgol_filter(path[:, 1], window_length=window_size, polyorder=2)
            except Exception:
                pass # Fallback to raw data if filter fails (e.g. too few points)

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
        
        # --- NOISE FILTERING (Hard Clamp for Stops) ---
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

    def _get_ego_polygon(self, x, y, heading) -> Polygon:
        base_poly = Polygon(VEHICLE_POLYGON_COORDS)
        rotated_poly = affinity.rotate(base_poly, heading, origin=(0, 0), use_radians=True)
        return affinity.translate(rotated_poly, xoff=x, yoff=y)

    def _query_nearby_lanes(self, x, y, radius=50.0):
        # x, y are already in Sim Frame (calibrated)
        nearby = []
        for lane, poly in self.all_lanes:
            minx, miny, maxx, maxy = poly.bounds
            if (minx - radius < x < maxx + radius) and (miny - radius < y < maxy + radius):
                nearby.append(lane)
        return nearby

    def _get_best_lane(self, x, y, heading, nearby_lanes, speed=None):
        if not nearby_lanes: return None
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

    def _calculate_metrics(self, states, horizon, frame_idx, role="Agent", prev_states=None):
        metrics = {
            "nc": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
            "ep": 0.0, "ttc": 1.0, "lk": 1.0, "hc": 1.0, "ec": 1.0
        }
        
        # 1. DAC
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
            if corners_valid < 4:
                metrics['dac'] = 0.0
                break

        # 2. NC & TTC
        tracks = self.scenario_data['tracks']
        for t in range(horizon):
            current_sim_frame = frame_idx + (t * self.gt_stride)
            if current_sim_frame >= self.scenario_data['length']: break
            
            ego_poly = self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
            ego_v = states['speed'][t]
            
            for obj_id, track in tracks.items():
                if obj_id == self.sdc_id: continue
                if track['type'] not in ['VEHICLE', 'CYCLIST']: continue
                
                pos_arr = track['state']['position']
                valid_arr = track['state']['valid']
                
                if current_sim_frame >= len(pos_arr) or not valid_arr[current_sim_frame]: continue
                
                # Transform Object to Sim Frame
                obj_pos_world = pos_arr[current_sim_frame][:2]
                ax, ay = self._to_sim_frame(obj_pos_world)
                ah = track['state']['heading'][current_sim_frame]
                agent_poly = self._get_ego_polygon(ax, ay, ah)
                
                if ego_poly.intersects(agent_poly):
                    is_at_fault = True
                    if ego_v < STOPPED_SPEED_THRESHOLD: is_at_fault = False
                    
                    dx, dy = ax - states['x'][t], ay - states['y'][t]
                    ego_hx, ego_hy = np.cos(states['heading'][t]), np.sin(states['heading'][t])
                    longitudinal_dist = dx * ego_hx + dy * ego_hy
                    if longitudinal_dist < -1.0: is_at_fault = False 
                    
                    if is_at_fault:
                        metrics['nc'] = 0.0
                        if t * self.planner_dt <= TTC_HORIZON: metrics['ttc'] = 0.0
                        break
            if metrics['nc'] == 0.0: break

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
            if current_sim_frame >= self.scenario_data['length']: break
            
            # Use a Set for O(1) lookup and exact string matching
            red_lane_ids = set()
            for lane_id, tl_data in self.traffic_lights.items():
                s_list = tl_data['state']['object_state']
                if current_sim_frame < len(s_list) and s_list[current_sim_frame] == "LANE_STATE_STOP":
                    red_lane_ids.add(str(lane_id))
            if not red_lane_ids: continue
            
            ego_poly = self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
            # Smaller radius to avoid picking up adjacent road segments
            nearby = self._query_nearby_lanes(states['x'][t], states['y'][t], radius=5.0)
            
            for lane in nearby:
                # Handle MetaDrive tuple indices
                curr_id = str(lane.index)
                if isinstance(lane.index, (tuple, list)) and len(lane.index) > 2:
                    curr_id = str(lane.index[2])
                
                # STRICT ID CHECK
                is_red = curr_id in red_lane_ids
                
                # LOGIC CHECK: Are we actually running the light?
                if is_red and states['speed'][t] > 1.0 and lane.shapely_polygon.intersects(ego_poly):
                    
                    # FIX: Calculate Longitudinal Position (s)
                    s, _ = lane.local_coordinates(np.array([states['x'][t], states['y'][t]]))
                    dist_to_end = lane.length - s
                    
                    # Heuristic: Only fail if we are NEAR THE END (crossing stop line)
                    # or if the lane is very short (likely the intersection interior itself).
                    # This prevents failing when driving AWAY from an intersection (start of lane).
                    if (lane.length < 10.0) or (dist_to_end < 5.0):
                        metrics['tlc'] = 0.0
                        break
                    else:
                        # Debug: We are on a red lane but safe (far from end)
                        # pass 
                        pass

            if metrics['tlc'] == 0.0: break

        # 6. Lane Keeping
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

        # 7. HC
        max_acc = np.max(states['acceleration'])
        max_jerk = np.max(states['jerk'])
        max_yr = np.max(np.abs(states['yaw_rate']))
        if (max_acc > MAX_ACCEL or max_jerk > MAX_JERK or max_yr > MAX_YAW_RATE):
            metrics['hc'] = 0.0

        # 8. EC
        if prev_states is not None:
            n_overlap = min(len(states['acceleration']), len(prev_states['acceleration']))
            # Use Index-aligned Comparison (No Shift) for 10Hz Sim Step
            diff_acc = np.sqrt(np.mean((states['acceleration'][:n_overlap] - prev_states['acceleration'][:n_overlap])**2))
            diff_jerk = np.sqrt(np.mean((states['jerk'][:n_overlap] - prev_states['jerk'][:n_overlap])**2))
            diff_yr = np.sqrt(np.mean((states['yaw_rate'][:n_overlap] - prev_states['yaw_rate'][:n_overlap])**2))
            
            if (diff_acc > EC_ACCEL_THRESH or diff_jerk > EC_JERK_THRESH or diff_yr > EC_YAW_RATE_THRESH):
                metrics['ec'] = 0.0
        
        # 5. Progress
        dist = np.linalg.norm(np.array([states['x'][-1], states['y'][-1]]) - np.array([states['x'][0], states['y'][0]]))
        metrics['ep'] = min(dist / 30.0, 1.0)
        
        return metrics

    def score_frame(self, plan_traj: np.ndarray, frame_idx: int) -> Dict[str, float]:
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if not sdc_track['state']['valid'][frame_idx]:
             return {"score": 0.0, "valid": False}
        current_pos = sdc_track['state']['position'][frame_idx][:2]
        
        # 1. Agent (Transform World -> Sim)
        full_agent_path_world = np.vstack([current_pos, plan_traj])
        full_agent_path_sim = self._to_sim_frame(full_agent_path_world)
        
        agent_states = self._get_trajectory_states(full_agent_path_sim, self.planner_dt)
        if agent_states is None: return {"score": 0.0, "valid": False}
        
        horizon = len(plan_traj) 
        agent_metrics = self._calculate_metrics(agent_states, len(plan_traj), frame_idx, "Agent", self.prev_plan_states)
        
        # 2. Human (Transform World -> Sim)
        human_traj = [current_pos]
        for t in range(1, len(plan_traj) + 1): 
            idx = frame_idx + (t * self.gt_stride) 
            if idx < self.scenario_data['length'] and sdc_track['state']['valid'][idx]:
                pos = sdc_track['state']['position'][idx][:2]
                human_traj.append(pos)
            else:
                break
        
        human_metrics = None
        if len(human_traj) >= 2:
            human_traj_sim = self._to_sim_frame(np.array(human_traj))
            human_states = self._get_trajectory_states(human_traj_sim, self.planner_dt)
            if human_states:
                human_metrics = self._calculate_metrics(human_states, len(human_traj), frame_idx, "Human", self.prev_human_states)
                self.prev_human_states = human_states
        
        # 3. Filter
        final_metrics = agent_metrics.copy()
        
        print(f"\n[METRIC REPORT] Frame {frame_idx}")
        print(f"{'METRIC':<10} | {'AGENT':<5} | {'HUMAN':<5} | {'FINAL':<10}")
        print("-" * 45)
        
        keys = ['nc', 'dac', 'ddc', 'tlc', 'ep', 'ttc', 'lk', 'hc', 'ec']
        for m in keys:
            agent_val = agent_metrics.get(m, 0.0)
            human_val = human_metrics.get(m, 1.0) if human_metrics else "N/A"
            final_val = agent_val
            if human_metrics and human_metrics.get(m) == 0.0:
                final_val = 1.0
            final_metrics[m] = final_val
            h_str = f"{human_val:.1f}" if isinstance(human_val, float) else human_val
            print(f"{m.upper():<10} | {agent_val:.1f}   | {h_str:<5} | {final_val:.1f}")

        multi_prod = (final_metrics['nc'] * final_metrics['dac'] * final_metrics['ddc'] * final_metrics['tlc'])
        weighted_sum = (W_PROGRESS * final_metrics['ep'] + W_TTC * final_metrics['ttc'] + 
                        W_LANE_KEEPING * final_metrics['lk'] + W_HISTORY_COMFORT * final_metrics['hc'] + 
                        W_EXTENDED_COMFORT * final_metrics['ec'])
        weighted_avg = weighted_sum / (W_PROGRESS + W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT)
        score = multi_prod * weighted_avg
        
        self.prev_plan_states = agent_states
        print(f"SCORE: {score:.4f}")
        
        # Return full set of keys for CSV compatibility
        return {
            "no_at_fault_collisions": final_metrics['nc'],
            "drivable_area_compliance": final_metrics['dac'],
            "driving_direction_compliance": final_metrics['ddc'],
            "traffic_light_compliance": final_metrics['tlc'],
            "ego_progress": final_metrics['ep'],
            "time_to_collision_within_bound": final_metrics['ttc'],
            "lane_keeping": final_metrics['lk'],
            "history_comfort": final_metrics['hc'],
            "extended_comfort": final_metrics['ec'],
            "score": score,
            "valid": True
        }

    def score_frame_no_ep(self, plan_traj: np.ndarray, frame_idx: int) -> Dict[str, float]:
        """
        Score frame without ego_progress (EP) for closed-loop evaluation.

        Formula: (NC × DAC × DDC × TLC) × (5×TTC + 2×LK + 2×HC + 2×EC) / 11

        This is used for closed-loop mode where route completion is computed
        separately from actual execution and multiplied at the end.
        """
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if not sdc_track['state']['valid'][frame_idx]:
             return {"score": 0.0, "valid": False}
        current_pos = sdc_track['state']['position'][frame_idx][:2]

        # 1. Agent (Transform World -> Sim)
        full_agent_path_world = np.vstack([current_pos, plan_traj])
        full_agent_path_sim = self._to_sim_frame(full_agent_path_world)

        agent_states = self._get_trajectory_states(full_agent_path_sim, self.planner_dt)
        if agent_states is None: return {"score": 0.0, "valid": False}

        horizon = len(plan_traj)
        agent_metrics = self._calculate_metrics(agent_states, len(plan_traj), frame_idx, "Agent", self.prev_plan_states)

        # 2. Human (Transform World -> Sim)
        human_traj = [current_pos]
        for t in range(1, len(plan_traj) + 1):
            idx = frame_idx + (t * self.gt_stride)
            if idx < self.scenario_data['length'] and sdc_track['state']['valid'][idx]:
                pos = sdc_track['state']['position'][idx][:2]
                human_traj.append(pos)
            else:
                break

        human_metrics = None
        if len(human_traj) >= 2:
            human_traj_sim = self._to_sim_frame(np.array(human_traj))
            human_states = self._get_trajectory_states(human_traj_sim, self.planner_dt)
            if human_states:
                human_metrics = self._calculate_metrics(human_states, len(human_traj), frame_idx, "Human", self.prev_human_states)
                self.prev_human_states = human_states

        # 3. Filter (same logic as score_frame)
        final_metrics = agent_metrics.copy()

        keys = ['nc', 'dac', 'ddc', 'tlc', 'ttc', 'lk', 'hc', 'ec']  # No 'ep'
        for m in keys:
            agent_val = agent_metrics.get(m, 0.0)
            if human_metrics and human_metrics.get(m) == 0.0:
                final_metrics[m] = 1.0

        # 4. Compute score WITHOUT ego_progress (EP)
        # Formula: (NC × DAC × DDC × TLC) × (5×TTC + 2×LK + 2×HC + 2×EC) / 11
        multi_prod = (final_metrics['nc'] * final_metrics['dac'] * final_metrics['ddc'] * final_metrics['tlc'])
        weighted_sum = (W_TTC * final_metrics['ttc'] +
                        W_LANE_KEEPING * final_metrics['lk'] +
                        W_HISTORY_COMFORT * final_metrics['hc'] +
                        W_EXTENDED_COMFORT * final_metrics['ec'])
        weighted_avg = weighted_sum / (W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT)
        score = multi_prod * weighted_avg

        self.prev_plan_states = agent_states

        return {
            "no_at_fault_collisions": final_metrics['nc'],
            "drivable_area_compliance": final_metrics['dac'],
            "driving_direction_compliance": final_metrics['ddc'],
            "traffic_light_compliance": final_metrics['tlc'],
            "time_to_collision_within_bound": final_metrics['ttc'],
            "lane_keeping": final_metrics['lk'],
            "history_comfort": final_metrics['hc'],
            "extended_comfort": final_metrics['ec'],
            "score": score,
            "valid": True
        }

    # ==================== LIVE SCORING METHODS ====================
    # These methods use current state from the simulation environment
    # instead of predicted trajectories or logged data for agents.

    def _get_current_agents_from_sim(self) -> List[dict]:
        """
        Query current agent states from simulation environment.
        Returns list of dicts with position, heading, and bounding box info.
        """
        agents = []
        ego_name = self.env.agent.name

        for obj in self.env.engine.get_objects().values():
            if obj.name == ego_name:
                continue
            # Only consider objects with position and heading (vehicles, cyclists, etc.)
            if not hasattr(obj, 'position') or not hasattr(obj, 'heading_theta'):
                continue

            # Get bounding box dimensions
            length = getattr(obj, 'top_down_length', getattr(obj, 'LENGTH', VEHICLE_LENGTH))
            width = getattr(obj, 'top_down_width', getattr(obj, 'WIDTH', VEHICLE_WIDTH))

            agents.append({
                'name': obj.name,
                'position': np.array(obj.position[:2]),  # [x, y]
                'heading': obj.heading_theta,
                'length': length,
                'width': width,
            })

        return agents

    def _get_agent_polygon(self, agent: dict) -> Polygon:
        """Create a polygon for an agent given its state dict."""
        half_l = agent['length'] / 2
        half_w = agent['width'] / 2
        coords = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])
        base_poly = Polygon(coords)
        rotated_poly = affinity.rotate(base_poly, agent['heading'], origin=(0, 0), use_radians=True)
        return affinity.translate(rotated_poly, xoff=agent['position'][0], yoff=agent['position'][1])

    def _check_dac_live(self, ego_pos: np.ndarray, ego_heading: float) -> float:
        """
        Check Drivable Area Compliance using current ego state.
        Returns 1.0 if all corners are in drivable area, 0.0 otherwise.
        """
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=15.0)
        ego_poly = self._get_ego_polygon(ego_pos[0], ego_pos[1], ego_heading)
        corners = [Point(c) for c in ego_poly.exterior.coords[:-1]]

        corners_valid = 0
        for corner in corners:
            for lane in nearby:
                if lane.shapely_polygon.contains(corner):
                    corners_valid += 1
                    break

        return 1.0 if corners_valid >= 4 else 0.0

    def _check_nc_ttc_live(self, ego_pos: np.ndarray, ego_heading: float,
                           ego_speed: float, current_agents: List[dict]) -> Tuple[float, float]:
        """
        Check No Collision and Time to Collision using current simulation state.
        Only checks current frame (t=0), not future predictions.

        Returns (nc, ttc) tuple.
        """
        nc, ttc = 1.0, 1.0

        ego_poly = self._get_ego_polygon(ego_pos[0], ego_pos[1], ego_heading)

        for agent in current_agents:
            agent_poly = self._get_agent_polygon(agent)

            if ego_poly.intersects(agent_poly):
                # At-fault logic (same as original)
                is_at_fault = True

                # If ego is stopped, not at fault
                if ego_speed < STOPPED_SPEED_THRESHOLD:
                    is_at_fault = False

                # If agent is behind ego, not at fault (rear-ended)
                dx = agent['position'][0] - ego_pos[0]
                dy = agent['position'][1] - ego_pos[1]
                ego_hx = np.cos(ego_heading)
                ego_hy = np.sin(ego_heading)
                longitudinal_dist = dx * ego_hx + dy * ego_hy
                if longitudinal_dist < -1.0:
                    is_at_fault = False

                if is_at_fault:
                    nc = 0.0
                    ttc = 0.0  # Collision at current frame means TTC = 0
                    break

        return nc, ttc

    def _check_ddc_live(self, ego_pos: np.ndarray, ego_heading: float, ego_speed: float) -> float:
        """
        Check Driving Direction Compliance using current ego state.
        Returns 1.0 if heading aligns with lane direction, 0.0 otherwise.
        """
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=15.0)
        best_lane = self._get_best_lane(ego_pos[0], ego_pos[1], ego_heading, nearby, speed=ego_speed)

        if best_lane and ego_speed > 1.0:
            s, r = best_lane.local_coordinates(np.array([ego_pos[0], ego_pos[1]]))
            s_clamped = max(0, min(s, best_lane.length))
            lane_heading_vec = best_lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(ego_heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi

            if abs(diff) > np.pi / 2:
                return 0.0

        return 1.0

    def _check_tlc_live(self, ego_pos: np.ndarray, ego_heading: float,
                        ego_speed: float, frame_idx: int) -> float:
        """
        Check Traffic Light Compliance using current ego state.
        Uses logged traffic light states at current frame_idx.
        Returns 1.0 if compliant, 0.0 if running a red light.
        """
        # Get red lane IDs from logged traffic light states at current frame
        red_lane_ids = set()
        for lane_id, tl_data in self.traffic_lights.items():
            s_list = tl_data['state']['object_state']
            if frame_idx < len(s_list) and s_list[frame_idx] == "LANE_STATE_STOP":
                red_lane_ids.add(str(lane_id))

        if not red_lane_ids:
            return 1.0

        ego_poly = self._get_ego_polygon(ego_pos[0], ego_pos[1], ego_heading)
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=5.0)

        for lane in nearby:
            # Handle MetaDrive tuple indices
            curr_id = str(lane.index)
            if isinstance(lane.index, (tuple, list)) and len(lane.index) > 2:
                curr_id = str(lane.index[2])

            is_red = curr_id in red_lane_ids

            if is_red and ego_speed > 1.0 and lane.shapely_polygon.intersects(ego_poly):
                s, _ = lane.local_coordinates(np.array([ego_pos[0], ego_pos[1]]))
                dist_to_end = lane.length - s

                # Only fail if near the end (crossing stop line) or lane is short
                if (lane.length < 10.0) or (dist_to_end < 5.0):
                    return 0.0

        return 1.0

    def _check_lk_live(self, ego_pos: np.ndarray, ego_heading: float, ego_speed: float) -> float:
        """
        Check Lane Keeping using current ego state.
        Tracks consecutive deviations across frames using self.consecutive_lane_deviation.
        Returns 1.0 if lane keeping is good, 0.0 if consistently deviating.
        """
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=10.0)
        best_lane = self._get_best_lane(ego_pos[0], ego_pos[1], ego_heading, nearby, speed=ego_speed)

        if best_lane:
            _, lat = best_lane.local_coordinates(np.array([ego_pos[0], ego_pos[1]]))
            if abs(lat) > LANE_DEVIATION_LIMIT:
                self.consecutive_lane_deviation += 1
            else:
                self.consecutive_lane_deviation = 0

        # Check if consecutive deviation exceeds threshold
        # Using scenario_dt as the time step between frames
        if self.consecutive_lane_deviation * self.scenario_dt > LANE_KEEPING_WINDOW:
            return 0.0

        return 1.0

    def _check_comfort_live(self, ego_pos: np.ndarray, ego_velocity: np.ndarray,
                            ego_heading: float) -> Tuple[float, float]:
        """
        Check History Comfort (HC) and Extended Comfort (EC) using actual motion.
        Computes acceleration, jerk, yaw_rate from motion history.

        Returns (hc, ec) tuple.
        """
        hc, ec = 1.0, 1.0
        dt = self.scenario_dt

        # Current speed
        speed = np.linalg.norm(ego_velocity)
        is_stopped = speed < STOPPED_SPEED_THRESHOLD

        # Compute current acceleration
        if self.prev_velocity is not None:
            acceleration_vec = (ego_velocity - self.prev_velocity) / dt
            acceleration = np.linalg.norm(acceleration_vec)
        else:
            acceleration = 0.0

        # Compute current jerk
        if self.prev_acceleration is not None:
            jerk = abs(acceleration - self.prev_acceleration) / dt
        else:
            jerk = 0.0

        # Compute current yaw rate
        if self.prev_heading is not None:
            heading_diff = ego_heading - self.prev_heading
            # Wrap to [-pi, pi]
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = abs(heading_diff) / dt
        else:
            yaw_rate = 0.0

        # Filter out noise when stopped
        if is_stopped:
            acceleration = 0.0
            jerk = 0.0
            yaw_rate = 0.0

        # HC: Check if current values exceed thresholds
        if acceleration > MAX_ACCEL or jerk > MAX_JERK or yaw_rate > MAX_YAW_RATE:
            hc = 0.0

        # EC: Check consistency with previous values
        if self.prev_acceleration is not None and self.prev_jerk is not None and self.prev_yaw_rate is not None:
            diff_acc = abs(acceleration - self.prev_acceleration)
            diff_jerk = abs(jerk - self.prev_jerk)
            diff_yr = abs(yaw_rate - self.prev_yaw_rate)

            if diff_acc > EC_ACCEL_THRESH or diff_jerk > EC_JERK_THRESH or diff_yr > EC_YAW_RATE_THRESH:
                ec = 0.0

        # Update history for next frame
        self.prev_position = ego_pos.copy()
        self.prev_velocity = ego_velocity.copy()
        self.prev_heading = ego_heading
        self.prev_acceleration = acceleration
        self.prev_jerk = jerk
        self.prev_yaw_rate = yaw_rate

        return hc, ec

    def score_frame_live(self, frame_idx: int) -> Dict[str, float]:
        """
        Score current frame using LIVE simulation state for ALL metrics.

        This method queries the current ego state and other agents directly from
        the simulation environment, rather than using predicted trajectories or
        logged agent data.

        Use this for closed-loop evaluation where actual execution matters.

        Args:
            frame_idx: Current frame index (used for traffic light state lookup)

        Returns:
            Dict with all metric scores and overall score.
        """
        # 1. Get CURRENT ego state from simulation
        ego_pos = np.array(self.env.agent.position[:2])
        ego_heading = self.env.agent.heading_theta
        ego_velocity = np.array(self.env.agent.velocity[:2]) if hasattr(self.env.agent, 'velocity') else np.array([0.0, 0.0])
        ego_speed = np.linalg.norm(ego_velocity)

        # 2. Get CURRENT other agents from simulation
        current_agents = self._get_current_agents_from_sim()

        # 3. Compute all metrics using live state
        # NC & TTC: Collision check with current agents
        nc, ttc = self._check_nc_ttc_live(ego_pos, ego_heading, ego_speed, current_agents)

        # DAC: Drivable area compliance
        dac = self._check_dac_live(ego_pos, ego_heading)

        # DDC: Driving direction compliance
        ddc = self._check_ddc_live(ego_pos, ego_heading, ego_speed) if dac > 0 else 1.0

        # TLC: Traffic light compliance
        tlc = self._check_tlc_live(ego_pos, ego_heading, ego_speed, frame_idx)

        # LK: Lane keeping
        lk = self._check_lk_live(ego_pos, ego_heading, ego_speed) if dac > 0 else 1.0

        # HC & EC: Comfort metrics from actual motion
        hc, ec = self._check_comfort_live(ego_pos, ego_velocity, ego_heading)

        # 4. Compute final score (same formula as score_frame_no_ep)
        multi_prod = nc * dac * ddc * tlc
        weighted_sum = (W_TTC * ttc + W_LANE_KEEPING * lk +
                        W_HISTORY_COMFORT * hc + W_EXTENDED_COMFORT * ec)
        weighted_avg = weighted_sum / (W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT)
        score = multi_prod * weighted_avg

        return {
            "no_at_fault_collisions": nc,
            "drivable_area_compliance": dac,
            "driving_direction_compliance": ddc,
            "traffic_light_compliance": tlc,
            "time_to_collision_within_bound": ttc,
            "lane_keeping": lk,
            "history_comfort": hc,
            "extended_comfort": ec,
            "score": score,
            "valid": True
        }

    def reset_live_state(self):
        """
        Reset the live scoring state. Call this at the start of each episode.
        """
        self.prev_position = None
        self.prev_velocity = None
        self.prev_heading = None
        self.prev_acceleration = None
        self.prev_jerk = None
        self.prev_yaw_rate = None
        self.consecutive_lane_deviation = 0
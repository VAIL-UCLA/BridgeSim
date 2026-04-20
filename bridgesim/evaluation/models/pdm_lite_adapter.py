"""
PDM-lite model adapter.

Rule-based privileged-state planner. No checkpoint required.
Pass --checkpoint none (or any path) when running.

Algorithm per frame:
  1. Extract GT centerline look-ahead (privileged)
  2. Generate 7 lateral-offset candidate paths
  3. IDM longitudinal control per candidate
  4. Kinematic bicycle forward simulation
  5. Score candidates (COL, DAC, DDC, TLC, EP, LK, HC)
  6. Return best trajectory in ego frame

Centerline source: GT future ego positions from scenario_data.
This is intentional — PDM-lite is a privileged-state planner.
"""

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from scipy.ndimage import uniform_filter1d
from shapely.geometry import Point, Polygon

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter

VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH  = 1.852


class _IDMController:
    """
    Intelligent Driver Model.
    Computes desired longitudinal acceleration given current speed,
    gap to leading vehicle, and relative speed.
    """

    def __init__(
        self,
        v_desired: float = 8.33,
        a_max: float = 2.0,
        b: float = 3.0,
        T: float = 1.5,
        s_min: float = 2.0,
        corridor_half_width: float = 1.5,
    ):
        self.v_desired = v_desired
        self.a_max = a_max
        self.b = b
        self.T = T
        self.s_min = s_min
        self.corridor_half_width = corridor_half_width

    def compute_acceleration(self, ego_speed: float, gap: float, rel_speed: float) -> float:
        """IDM formula. gap: bumper-to-bumper gap (m); rel_speed: ego - leader (positive = closing)."""
        if ego_speed < 0.0:
            ego_speed = 0.0
        v_ratio = ego_speed / max(self.v_desired, 1e-3)
        s_star = self.s_min + max(
            0.0,
            ego_speed * self.T + ego_speed * rel_speed / (2.0 * math.sqrt(self.a_max * self.b)),
        )
        gap = max(gap, 1e-3)
        a = self.a_max * (1.0 - v_ratio ** 4 - (s_star / gap) ** 2)
        return float(np.clip(a, -self.b, self.a_max))

    def find_leading_vehicle(
        self,
        x: float,
        y: float,
        heading: float,
        ego_speed: float,
        agents: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        """Scan agents; return (gap_m, rel_speed) for closest agent in corridor ahead."""
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        fwd = np.array([cos_h, sin_h])
        lat = np.array([-sin_h, cos_h])

        best_gap = math.inf
        best_rel_speed = 0.0

        for agent in agents:
            delta = agent["position"][:2] - np.array([x, y])
            lon = float(np.dot(delta, fwd))
            if lon <= -VEHICLE_LENGTH:
                continue
            if abs(float(np.dot(delta, lat))) > self.corridor_half_width:
                continue
            gap = max(0.0, lon - VEHICLE_LENGTH)
            agent_lon_speed = float(np.dot(agent["velocity"][:2], fwd))
            rel = ego_speed - agent_lon_speed
            if gap < best_gap:
                best_gap = gap
                best_rel_speed = rel

        return best_gap, best_rel_speed


class _KinematicBicycleSimulator:
    """
    Forward-integrates a kinematic bicycle model along a reference path.
    Control: pure-pursuit steering + IDM longitudinal.
    Integration at sim_dt=0.1 s; outputs every output_dt=0.5 s.
    """

    MAX_STEER = 0.6
    LOOKAHEAD_BASE = 5.0
    LOOKAHEAD_SPEED_GAIN = 2.5

    def __init__(
        self,
        wheelbase: float = 2.8,
        sim_dt: float = 0.1,
        output_dt: float = 0.5,
        n_output_steps: int = 8,
    ):
        self.wheelbase = wheelbase
        self.sim_dt = sim_dt
        self.output_dt = output_dt
        self.n_output_steps = n_output_steps
        self.steps_per_output = int(round(output_dt / sim_dt))

    def simulate(
        self,
        path: np.ndarray,
        ego_state: Dict[str, Any],
        agents: List[Dict[str, Any]],
        idm: "_IDMController",
    ) -> np.ndarray:
        """Simulate for n_output_steps; returns (n_output_steps, 2) world-frame trajectory."""
        x = float(ego_state["position"][0])
        y = float(ego_state["position"][1])
        heading = float(ego_state["heading"])
        speed = float(ego_state["speed"])

        total_steps = self.n_output_steps * self.steps_per_output
        output_waypoints: List[List[float]] = []

        for step in range(total_steps):
            gap, rel_speed = idm.find_leading_vehicle(x, y, heading, speed, agents)
            a = idm.compute_acceleration(speed, gap, rel_speed)
            delta = self._pure_pursuit(x, y, heading, speed, path)

            x       += speed * math.cos(heading) * self.sim_dt
            y       += speed * math.sin(heading) * self.sim_dt
            heading += speed / self.wheelbase * math.tan(delta) * self.sim_dt
            speed    = float(np.clip(speed + a * self.sim_dt, 0.0, idm.v_desired))

            if (step + 1) % self.steps_per_output == 0:
                output_waypoints.append([x, y])

        return np.array(output_waypoints)

    def _pure_pursuit(self, x: float, y: float, heading: float, speed: float, path: np.ndarray) -> float:
        if len(path) == 0:
            return 0.0
        ld = self.LOOKAHEAD_BASE + self.LOOKAHEAD_SPEED_GAIN * speed
        ego = np.array([x, y])
        dists = np.linalg.norm(path - ego, axis=1)
        closest_idx = int(np.argmin(dists))

        target = path[-1]
        cumulative = 0.0
        for i in range(closest_idx, len(path) - 1):
            seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
            if cumulative + seg_len >= ld:
                t = (ld - cumulative) / max(seg_len, 1e-9)
                target = path[i] + t * (path[i + 1] - path[i])
                break
            cumulative += seg_len

        dx = target[0] - x
        dy = target[1] - y
        alpha = math.atan2(dy, dx) - heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        return float(np.clip(delta, -self.MAX_STEER, self.MAX_STEER))


class _CenterlineExtractor:
    """
    Extracts a smoothed, uniformly-spaced centerline from scenario GT ego
    track, then generates laterally-offset copies as planning candidates.
    Using GT future positions is correct here: PDM-lite is a privileged planner
    and its value lies in lateral candidate generation + safety scoring.
    """

    def __init__(
        self,
        lookahead_m: float = 50.0,
        smooth_window: int = 5,
        resample_spacing: float = 0.5,
        lateral_offsets: Tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
    ):
        self.lookahead_m = lookahead_m
        self.smooth_window = smooth_window
        self.resample_spacing = resample_spacing
        self.lateral_offsets = lateral_offsets

    def extract_centerline(
        self,
        scenario_data: Dict[str, Any],
        sdc_id: str,
        ego_pos: np.ndarray,
        frame_id: int,
    ) -> Optional[np.ndarray]:
        """Return (N, 2) centerline in world frame, or None if too short."""
        track = scenario_data["tracks"][sdc_id]
        positions = track["state"]["position"]
        valid    = track["state"]["valid"]

        future_pts: List[np.ndarray] = []
        cumulative_dist: float = 0.0
        for i in range(frame_id, len(positions)):
            if not valid[i]:
                continue
            pt = positions[i][:2].copy()
            if future_pts:
                cumulative_dist += float(np.linalg.norm(pt - future_pts[-1]))
                if cumulative_dist >= self.lookahead_m:
                    future_pts.append(pt)
                    break
            future_pts.append(pt)

        if len(future_pts) < 2:
            return None

        pts = np.array(future_pts)
        if len(pts) >= self.smooth_window:
            pts[:, 0] = uniform_filter1d(pts[:, 0], size=self.smooth_window, mode='nearest')
            pts[:, 1] = uniform_filter1d(pts[:, 1], size=self.smooth_window, mode='nearest')

        return self._resample(pts)

    def _resample(self, pts: np.ndarray) -> np.ndarray:
        dists = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        total = dists[-1]
        if total < 1e-6:
            return pts
        n = max(2, int(total / self.resample_spacing) + 1)
        t_uniform = np.linspace(0.0, total, n)
        x = np.interp(t_uniform, dists, pts[:, 0])
        y = np.interp(t_uniform, dists, pts[:, 1])
        return np.stack([x, y], axis=1)

    def generate_candidates(self, centerline: Optional[np.ndarray]) -> List[np.ndarray]:
        """Return list of (N, 2) candidate paths — one per lateral offset."""
        if centerline is None or len(centerline) < 2:
            return []
        tangents = np.diff(centerline, axis=0)
        tangents = np.vstack([tangents, tangents[-1:]])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        tangents = tangents / norms
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        candidates: List[np.ndarray] = []
        for offset in self.lateral_offsets:
            if abs(offset) < 1e-9:
                candidates.append(centerline.copy())
            else:
                candidates.append(centerline + offset * normals)
        return candidates


class _PDMLiteScorer:
    """
    Scores a simulated world-frame trajectory using EPDMS-compatible metrics.
    Metrics: COL, DAC, DDC, TLC, EP, LK, HC.
    Combined: (COL × DAC × DDC × TLC) × (5·EP + 2·LK + 2·HC) / 9
    """

    PLANNER_DT      = 0.5
    DEVIATION_LIMIT = 0.5
    LK_WINDOW_S     = 2.0
    MAX_ACCEL       = 4.89

    def __init__(
        self,
        all_lanes: list,
        all_lane_bounds: np.ndarray,
        world_to_sim_offset: np.ndarray,
    ):
        self.all_lanes = all_lanes
        self.all_lane_bounds = all_lane_bounds
        self.world_to_sim_offset = world_to_sim_offset

    def score(
        self,
        traj_world: np.ndarray,
        tl_states: set,
        agents: List[Dict[str, Any]],
    ) -> float:
        T = len(traj_world)
        if T == 0:
            return 0.0
        traj_sim = traj_world + self.world_to_sim_offset
        headings = self._compute_headings(traj_sim)

        col = self._score_col(traj_sim, headings, agents)
        dac = self._score_dac(traj_sim)
        ddc = self._score_ddc(traj_sim, headings, dac)
        tlc = self._score_tlc(traj_sim, headings, tl_states)
        ep  = self._score_ep(traj_sim)
        lk  = self._score_lk(traj_sim, headings)
        hc  = self._score_hc(traj_sim)

        return float(col * dac * ddc * tlc * (5.0 * ep + 2.0 * lk + 2.0 * hc) / 9.0)

    def _compute_headings(self, traj: np.ndarray) -> np.ndarray:
        if len(traj) < 2:
            return np.zeros(len(traj))
        diffs = np.diff(traj, axis=0)
        h = np.arctan2(diffs[:, 1], diffs[:, 0])
        return np.append(h, h[-1])

    def _ego_polygon(self, x: float, y: float, heading: float) -> Polygon:
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        hl, hw = VEHICLE_LENGTH / 2.0, VEHICLE_WIDTH / 2.0
        corners = [
            (x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h),
            (x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h),
            (x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h),
            (x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h),
        ]
        return Polygon(corners)

    def _nearby_lane_indices(self, x: float, y: float, radius: float = 15.0) -> List[int]:
        if len(self.all_lane_bounds) == 0:
            return []
        b = self.all_lane_bounds
        mask = (
            (b[:, 0] - radius < x) & (x < b[:, 2] + radius) &
            (b[:, 1] - radius < y) & (y < b[:, 3] + radius)
        )
        return list(np.where(mask)[0])

    def _best_lane(self, x: float, y: float, heading: float, nearby: List[int]):
        pos = np.array([x, y])
        best, best_dist = None, math.inf
        for idx in nearby:
            lane = self.all_lanes[idx][0]
            dist = lane.distance(pos)
            s, _ = lane.local_coordinates(pos)
            s_c = max(0.0, min(s, lane.length))
            lh_vec = lane.heading_at(s_c)
            lh = math.atan2(lh_vec[1], lh_vec[0])
            diff = abs(heading - lh)
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < math.pi / 2 and dist < best_dist:
                best_dist = dist
                best = lane
        return best

    def _score_dac(self, traj: np.ndarray) -> float:
        if not self.all_lanes:
            return 1.0
        valid = 0
        for pt in traj:
            nearby = self._nearby_lane_indices(pt[0], pt[1])
            p = Point(pt[0], pt[1])
            if any(self.all_lanes[i][1].contains(p) for i in nearby):
                valid += 1
        return valid / len(traj)

    def _score_ddc(self, traj: np.ndarray, headings: np.ndarray, dac: float) -> float:
        if dac == 0.0 or not self.all_lanes:
            return 1.0
        for t in range(0, len(traj), 5):
            x, y = traj[t, 0], traj[t, 1]
            nearby = self._nearby_lane_indices(x, y)
            best_dist = math.inf
            nearest_lane = None
            for idx in nearby:
                lane = self.all_lanes[idx][0]
                dist = lane.distance(np.array([x, y]))
                if dist < best_dist:
                    best_dist = dist
                    nearest_lane = lane
            if nearest_lane is not None:
                s, _ = nearest_lane.local_coordinates(np.array([x, y]))
                s_c = max(0.0, min(s, nearest_lane.length))
                lh_vec = nearest_lane.heading_at(s_c)
                lh = math.atan2(lh_vec[1], lh_vec[0])
                diff = abs(headings[t] - lh)
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                if abs(diff) > math.pi / 2:
                    return 0.0
        return 1.0

    def _score_tlc(self, traj: np.ndarray, headings: np.ndarray, tl_states: set) -> float:
        if not tl_states or not self.all_lanes:
            return 1.0
        for t, pt in enumerate(traj):
            nearby = self._nearby_lane_indices(pt[0], pt[1], 5.0)
            ego_poly = self._ego_polygon(pt[0], pt[1], headings[t])
            for idx in nearby:
                lane = self.all_lanes[idx][0]
                raw_idx = lane.index
                lane_id = str(raw_idx[2]) if isinstance(raw_idx, (tuple, list)) and len(raw_idx) > 2 else str(raw_idx)
                if lane_id not in tl_states:
                    continue
                if self.all_lanes[idx][1].intersects(ego_poly):
                    s, _ = lane.local_coordinates(np.array([pt[0], pt[1]]))
                    if lane.length - s < 5.0:
                        return 0.0
        return 1.0

    def _score_col(self, traj: np.ndarray, headings: np.ndarray, agents: List[Dict[str, Any]]) -> float:
        if not agents:
            return 1.0
        for t, pt in enumerate(traj):
            ego_poly = self._ego_polygon(pt[0], pt[1], headings[t])
            dt_future = t * self.PLANNER_DT
            for agent in agents:
                a_pos = agent["position"][:2] + agent["velocity"][:2] * dt_future + self.world_to_sim_offset
                a_poly = self._ego_polygon(a_pos[0], a_pos[1], agent["heading"])
                if ego_poly.intersects(a_poly):
                    return 0.0
        return 1.0

    def _score_ep(self, traj: np.ndarray) -> float:
        if len(traj) < 2:
            return 0.0
        dist = float(np.linalg.norm(traj[-1] - traj[0]))
        return min(dist / (3.0 * len(traj)), 1.0)

    def _score_lk(self, traj: np.ndarray, headings: np.ndarray) -> float:
        if not self.all_lanes:
            return 1.0
        consecutive_bad = 0
        for t, pt in enumerate(traj):
            nearby = self._nearby_lane_indices(pt[0], pt[1], 10.0)
            best = self._best_lane(pt[0], pt[1], headings[t], nearby)
            if best is not None:
                _, lat = best.local_coordinates(np.array([pt[0], pt[1]]))
                if abs(lat) > self.DEVIATION_LIMIT:
                    consecutive_bad += 1
                else:
                    consecutive_bad = 0
            else:
                consecutive_bad = 0
            if consecutive_bad * self.PLANNER_DT > self.LK_WINDOW_S:
                return 0.0
        return 1.0

    def _score_hc(self, traj: np.ndarray) -> float:
        if len(traj) < 3:
            return 1.0
        speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.PLANNER_DT
        accels = np.abs(np.diff(speeds)) / self.PLANNER_DT
        if len(accels) > 0 and float(np.max(accels)) > self.MAX_ACCEL:
            return 0.0
        return 1.0


class PDMLiteAdapter(BaseModelAdapter):
    """
    BridgeSim adapter for PDM-lite (rule-based, no checkpoint required).

    Usage:
        python bridgesim/evaluation/unified_evaluator.py \\
            --model-type pdm_lite --checkpoint none \\
            --scenario-path /path/to/scenario --output-dir outputs/
    """

    def __init__(
        self,
        checkpoint_path: str = "none",
        config_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self._env = None
        self._extractor: Optional[_CenterlineExtractor] = None
        self._idm: Optional[_IDMController] = None
        self._simulator: Optional[_KinematicBicycleSimulator] = None
        self._sdc_id: Optional[str] = None
        self._all_lanes: list = []
        self._all_lane_bounds: np.ndarray = np.empty((0, 4))
        self._world_to_sim_offset: np.ndarray = np.zeros(2)
        self._lanes_initialized: bool = False

    def load_model(self) -> None:
        if self.checkpoint_path.lower() not in ("none", "", "null"):
            print(f"[PDM-lite] Checkpoint path ignored (rule-based model): {self.checkpoint_path}")

        cfg = self._load_config()

        self._extractor = _CenterlineExtractor(
            lookahead_m=cfg.get("lookahead_m", 50.0),
            smooth_window=cfg.get("smooth_window", 5),
            resample_spacing=cfg.get("resample_spacing", 0.5),
            lateral_offsets=tuple(cfg.get("lateral_offsets", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])),
        )
        self._idm = _IDMController(
            v_desired=cfg.get("v_desired", 8.33),
            a_max=cfg.get("a_max", 2.0),
            b=cfg.get("b", 3.0),
            T=cfg.get("T", 1.5),
            s_min=cfg.get("s_min", 2.0),
        )
        self._simulator = _KinematicBicycleSimulator(
            wheelbase=cfg.get("wheelbase", 2.8),
            sim_dt=0.1,
            output_dt=0.5,
            n_output_steps=8,
        )
        print("[PDM-lite] Loaded successfully (rule-based, no checkpoint).")

    def _load_config(self) -> dict:
        if self.config_path and self.config_path.lower() not in ("none", ""):
            try:
                import yaml
                with open(self.config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[PDM-lite] Could not load config {self.config_path}: {e}")
        return {}

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {}

    def perceive(self, env, frame_id: int):
        """Cache env reference; extract lane geometry once on first call."""
        self._env = env
        if not self._lanes_initialized:
            self._init_lane_geometry(env)
        return None

    def _init_lane_geometry(self, env) -> None:
        """Extract MetaDrive lane polygons for DAC/DDC/TLC/LK scoring."""
        try:
            road_network = env.engine.map_manager.current_map.road_network
            if hasattr(road_network, "get_all_lanes"):
                lanes_iter = road_network.get_all_lanes()
            else:
                lanes_iter = (
                    lane
                    for start, end_dict in road_network.graph.items()
                    for end, lanes in end_dict.items()
                    for lane in lanes
                )
            for lane in lanes_iter:
                if hasattr(lane, "shapely_polygon"):
                    self._all_lanes.append((lane, lane.shapely_polygon))
        except Exception as e:
            print(f"[PDM-lite] Warning: could not extract lane geometry: {e}")

        if self._all_lanes:
            self._all_lane_bounds = np.array([poly.bounds for _, poly in self._all_lanes])
        else:
            self._all_lane_bounds = np.empty((0, 4))
        self._lanes_initialized = True

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        self._sdc_id = scenario_data["metadata"]["sdc_id"]

        if self._env is not None:
            sdc_track = scenario_data["tracks"][self._sdc_id]
            if sdc_track["state"]["valid"][0]:
                world_pos_0 = sdc_track["state"]["position"][0][:2]
                sim_pos_0 = np.array(self._env.agent.position)[:2]
                self._world_to_sim_offset = sim_pos_0 - world_pos_0

        agents = self._extract_agents(scenario_data, frame_id)
        tl_states = self._extract_tl_states(scenario_data, frame_id)
        centerline = self._extractor.extract_centerline(
            scenario_data, self._sdc_id, ego_state["position"], frame_id
        )

        return {
            "ego_state": ego_state,
            "agents": agents,
            "tl_states": tl_states,
            "centerline": centerline,
            "frame_id": frame_id,
        }

    def run_inference(self, model_input: Any) -> Any:
        ego_state  = model_input["ego_state"]
        agents     = model_input["agents"]
        tl_states  = model_input["tl_states"]
        centerline = model_input["centerline"]

        candidates = self._extractor.generate_candidates(centerline)
        if not candidates:
            return self._fallback_straight(ego_state)

        scorer = _PDMLiteScorer(
            all_lanes=self._all_lanes,
            all_lane_bounds=self._all_lane_bounds,
            world_to_sim_offset=self._world_to_sim_offset,
        )

        simulated: List[np.ndarray] = []
        scores: List[float] = []
        for path in candidates:
            traj = self._simulator.simulate(path, ego_state, agents, self._idm)
            simulated.append(traj)
            scores.append(scorer.score(traj, tl_states, agents))

        best_idx = int(np.argmax(scores))
        return {
            "trajectory_world": simulated[best_idx],
            "all_scores": scores,
            "best_idx": best_idx,
        }

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert world-frame trajectory to BridgeSim ego frame (lateral, forward)."""
        traj_world = np.asarray(model_output["trajectory_world"])
        cos_h = math.cos(-ego_state["heading"])
        sin_h = math.sin(-ego_state["heading"])
        delta = traj_world - ego_state["position"][:2]
        lateral = sin_h * delta[:, 0] + cos_h * delta[:, 1]
        forward = cos_h * delta[:, 0] - sin_h * delta[:, 1]
        return {"trajectory": np.stack([lateral, forward], axis=1)}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0

    def _extract_agents(self, scenario_data: Dict[str, Any], frame_id: int) -> List[Dict[str, Any]]:
        agents: List[Dict[str, Any]] = []
        sdc_id = self._sdc_id
        sdc_pos = scenario_data["tracks"][sdc_id]["state"]["position"]
        if frame_id >= len(sdc_pos):
            return agents
        ego_xy = sdc_pos[frame_id][:2]

        for obj_id, track in scenario_data["tracks"].items():
            if obj_id == sdc_id:
                continue
            if track["type"] not in ("VEHICLE", "CYCLIST"):
                continue
            pos_arr   = track["state"]["position"]
            valid_arr = track["state"]["valid"]
            if frame_id >= len(pos_arr) or not valid_arr[frame_id]:
                continue
            pos = pos_arr[frame_id][:2]
            if float(np.linalg.norm(pos - ego_xy)) > 60.0:
                continue

            if "velocity" in track["state"] and frame_id < len(track["state"]["velocity"]):
                vel = track["state"]["velocity"][frame_id][:2].astype(float)
            elif frame_id > 0 and valid_arr[frame_id - 1]:
                vel = (pos - pos_arr[frame_id - 1][:2]) / 0.1
            else:
                vel = np.zeros(2)

            agents.append({
                "position": pos.astype(float),
                "velocity": vel,
                "heading":  float(track["state"]["heading"][frame_id]),
                "type":     track["type"],
            })
        return agents

    def _extract_tl_states(self, scenario_data: Dict[str, Any], frame_id: int) -> set:
        red: set = set()
        for lane_id, tl_data in scenario_data.get("dynamic_map_states", {}).items():
            s_list = tl_data["state"]["object_state"]
            if frame_id < len(s_list) and s_list[frame_id] == "LANE_STATE_STOP":
                red.add(str(lane_id))
        return red

    def _fallback_straight(self, ego_state: Dict[str, Any]) -> Dict[str, Any]:
        """Straight-line fallback when no centerline can be extracted."""
        heading = float(ego_state["heading"])
        speed   = max(float(ego_state["speed"]), 1.0)
        pos_xy  = ego_state["position"][:2]
        dt      = 0.5
        waypoints = [
            [pos_xy[0] + speed * math.cos(heading) * (i * dt),
             pos_xy[1] + speed * math.sin(heading) * (i * dt)]
            for i in range(1, 9)
        ]
        return {"trajectory_world": np.array(waypoints), "all_scores": [0.0], "best_idx": 0}

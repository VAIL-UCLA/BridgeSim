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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.ndimage import uniform_filter1d
from shapely.geometry import Point, Polygon

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS

VEHICLE_LENGTH = 4.515
VEHICLE_WIDTH  = 1.852
CYCLIST_LENGTH = 2.0
CYCLIST_WIDTH  = 0.8
PEDESTRIAN_LENGTH = 0.8
PEDESTRIAN_WIDTH  = 0.8


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _infer_box_size(agent_type: str) -> Tuple[float, float]:
    t = agent_type.upper()
    if "PEDESTRIAN" in t or "HUMAN" in t:
        return PEDESTRIAN_LENGTH, PEDESTRIAN_WIDTH
    if "CYCLIST" in t or "BICYCLE" in t:
        return CYCLIST_LENGTH, CYCLIST_WIDTH
    return VEHICLE_LENGTH, VEHICLE_WIDTH


class _LiveRoutePlanner:
    def __init__(
        self,
        lookahead_m: float = 60.0,
        sample_spacing: float = 0.5,
        lateral_offsets: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    ):
        self.lookahead_m = lookahead_m
        self.sample_spacing = sample_spacing
        self.lateral_offsets = tuple(lateral_offsets)

    def build_candidate_paths(
        self,
        env: Any,
        ego_state: Dict[str, Any],
        ego_position_sim: np.ndarray,
        sim_to_world_offset: np.ndarray,
        target_waypoint_world: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        route_lanes = self._get_route_lanes(env, ego_position_sim, target_waypoint_world, sim_to_world_offset)
        if not route_lanes:
            return self._fallback_paths(ego_state, target_waypoint_world)

        base_paths = []
        for lane_chain in route_lanes:
            base_sim = self._sample_lane_chain(lane_chain, ego_position_sim)
            if len(base_sim) >= 2:
                base_paths.append(base_sim + sim_to_world_offset)

        if not base_paths:
            return self._fallback_paths(ego_state, target_waypoint_world)

        candidates: List[np.ndarray] = []
        for i, base in enumerate(base_paths):
            offsets = (0.0,) if i > 0 else self.lateral_offsets
            candidates.extend(self._offset_path(base, offsets))

        return candidates[:7]

    def _get_route_lanes(
        self,
        env: Any,
        ego_position_sim: np.ndarray,
        target_waypoint_world: Optional[np.ndarray],
        sim_to_world_offset: np.ndarray,
    ) -> List[List[Any]]:
        agent = getattr(env, "agent", None)
        navigation = getattr(agent, "navigation", None)
        if navigation is None:
            lane = getattr(agent, "lane", None)
            return [[lane]] if lane is not None else []

        current_lanes = list(getattr(navigation, "current_ref_lanes", None) or [])
        next_lanes = list(getattr(navigation, "next_ref_lanes", None) or [])
        if not current_lanes:
            lane = getattr(agent, "lane", None)
            current_lanes = [lane] if lane is not None else []
        if not current_lanes:
            return []

        selected_idx = self._pick_best_lane_index(current_lanes, target_waypoint_world, sim_to_world_offset)
        lane_chains: List[List[Any]] = []
        for idx, lane in enumerate(current_lanes):
            chain = [lane]
            next_lane = self._pick_following_lane(next_lanes, idx, selected_idx)
            if next_lane is not None and next_lane is not lane:
                chain.append(next_lane)
            lane_chains.append(chain)
        return lane_chains

    def _pick_best_lane_index(
        self,
        current_lanes: List[Any],
        target_waypoint_world: Optional[np.ndarray],
        sim_to_world_offset: np.ndarray,
    ) -> int:
        if target_waypoint_world is None or len(current_lanes) == 1:
            return len(current_lanes) // 2

        target_sim = np.asarray(target_waypoint_world[:2], dtype=float) - sim_to_world_offset
        best_idx = 0
        best_dist = math.inf
        for idx, lane in enumerate(current_lanes):
            long, _ = lane.local_coordinates(target_sim)
            long = float(np.clip(long, 0.0, getattr(lane, "length", 0.0)))
            center = np.asarray(lane.position(long, 0.0), dtype=float)
            dist = float(np.linalg.norm(center - target_sim))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _pick_following_lane(self, next_lanes: List[Any], idx: int, selected_idx: int) -> Optional[Any]:
        if not next_lanes:
            return None
        if len(next_lanes) == 1:
            return next_lanes[0]
        mapped = min(len(next_lanes) - 1, max(0, idx))
        return next_lanes[mapped]

    def _sample_lane_chain(self, lane_chain: List[Any], ego_position_sim: np.ndarray) -> np.ndarray:
        points: List[np.ndarray] = []
        remaining = self.lookahead_m
        for chain_idx, lane in enumerate(lane_chain):
            if lane is None:
                continue
            start_long = 0.0
            if chain_idx == 0:
                start_long = max(0.0, lane.local_coordinates(ego_position_sim)[0])
            lane_length = float(getattr(lane, "length", 0.0))
            if lane_length <= start_long:
                continue

            sample_count = max(2, int(min(remaining, lane_length - start_long) / self.sample_spacing) + 1)
            longs = np.linspace(start_long, min(lane_length, start_long + remaining), sample_count)
            for s in longs:
                p = np.asarray(lane.position(float(s), 0.0), dtype=float)
                if not points or np.linalg.norm(p - points[-1]) > 1e-3:
                    points.append(p)

            if len(points) >= 2:
                remaining = self.lookahead_m - float(
                    np.sum(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1))
                )
            if remaining <= self.sample_spacing:
                break

        return np.asarray(points, dtype=float)

    def _offset_path(self, base_path_world: np.ndarray, offsets: Sequence[float]) -> List[np.ndarray]:
        if len(base_path_world) < 2:
            return [base_path_world]

        tangents = np.diff(base_path_world, axis=0)
        tangents = np.vstack([tangents, tangents[-1:]])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, 1e-6)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        out = []
        for offset in offsets:
            if abs(offset) < 1e-9:
                out.append(base_path_world.copy())
            else:
                out.append(base_path_world + offset * normals)
        return out

    def _fallback_paths(self, ego_state: Dict[str, Any], target_waypoint_world: Optional[np.ndarray]) -> List[np.ndarray]:
        heading = float(ego_state["heading"])
        start = np.asarray(ego_state["position"][:2], dtype=float)
        if target_waypoint_world is not None:
            vec = np.asarray(target_waypoint_world[:2], dtype=float) - start
            if np.linalg.norm(vec) > 1e-3:
                heading = math.atan2(vec[1], vec[0])

        lengths = np.linspace(2.0, 20.0, 40)
        base = np.stack(
            [start[0] + lengths * math.cos(heading), start[1] + lengths * math.sin(heading)],
            axis=1,
        )
        return self._offset_path(base, self.lateral_offsets)


class _CandidateScorer:
    PLANNER_DT = 0.5
    TTC_SAFE = 3.0
    TTC_DANGER = 0.5
    LK_LIMIT = 0.75
    MAX_HARD_ACCEL = 4.89
    MAX_EXT_ACCEL_DELTA = 0.7
    MAX_EXT_YAW_RATE_DELTA = 0.1

    def __init__(self, all_lanes: List[Tuple[Any, "Polygon"]], all_lane_bounds: np.ndarray, world_to_sim_offset: np.ndarray):
        self.all_lanes = all_lanes
        self.all_lane_bounds = all_lane_bounds
        self.world_to_sim_offset = world_to_sim_offset

    def score(
        self,
        traj_world: np.ndarray,
        ego_state: Dict[str, Any],
        agents: List[Dict[str, Any]],
        red_light_lanes: set,
    ) -> float:
        if len(traj_world) == 0:
            return 0.0

        traj_sim = traj_world + self.world_to_sim_offset
        headings = self._compute_headings(traj_sim, ego_state["heading"])

        col = self._score_collision(traj_sim, headings, agents)
        dac = self._score_dac(traj_sim)
        ddc = self._score_ddc(traj_sim, headings)
        tlc = self._score_tlc(traj_sim, headings, red_light_lanes)
        ttc = self._score_ttc(traj_sim, headings, agents)
        lk = self._score_lane_keeping(traj_sim, headings)
        hc = self._score_hard_comfort(traj_world, ego_state)
        ec = self._score_extended_comfort(traj_world, ego_state)
        progress = self._score_progress(traj_sim)

        gated = col * dac * ddc * tlc
        comfort = (hc + ec) / 2.0
        weighted = (5.0 * progress + 5.0 * ttc + 2.0 * comfort) / 12.0
        return float(gated * weighted)

    def _compute_headings(self, traj_sim: np.ndarray, initial_heading: float) -> np.ndarray:
        if len(traj_sim) < 2:
            return np.full(len(traj_sim), float(initial_heading))
        diffs = np.diff(traj_sim, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        return np.append(headings, headings[-1])

    def _ego_polygon(self, x: float, y: float, heading: float, length: float = VEHICLE_LENGTH, width: float = VEHICLE_WIDTH) -> "Polygon":
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        hl = 0.5 * length
        hw = 0.5 * width
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

    def _score_dac(self, traj_sim: np.ndarray) -> float:
        if not self.all_lanes:
            return 1.0
        valid = 0
        for pt in traj_sim:
            p = Point(float(pt[0]), float(pt[1]))
            nearby = self._nearby_lane_indices(pt[0], pt[1])
            if any(self.all_lanes[idx][1].contains(p) for idx in nearby):
                valid += 1
        return valid / max(len(traj_sim), 1)

    def _score_ddc(self, traj_sim: np.ndarray, headings: np.ndarray) -> float:
        if not self.all_lanes:
            return 1.0
        for t in range(0, len(traj_sim), 2):
            lane = self._nearest_lane(traj_sim[t])
            if lane is None:
                continue
            long, _ = lane.local_coordinates(traj_sim[t])
            lane_heading = lane.heading_theta_at(float(np.clip(long, 0.0, lane.length)))
            if abs(_wrap_to_pi(headings[t] - lane_heading)) > math.pi / 2:
                return 0.0
        return 1.0

    def _score_tlc(self, traj_sim: np.ndarray, headings: np.ndarray, red_light_lanes: set) -> float:
        if not red_light_lanes or not self.all_lanes:
            return 1.0
        for t, pt in enumerate(traj_sim):
            ego_poly = self._ego_polygon(pt[0], pt[1], headings[t])
            for idx in self._nearby_lane_indices(pt[0], pt[1], 6.0):
                lane = self.all_lanes[idx][0]
                lane_id = str(getattr(lane, "index", ""))
                if lane_id not in red_light_lanes:
                    continue
                if self.all_lanes[idx][1].intersects(ego_poly):
                    long, _ = lane.local_coordinates(pt)
                    if lane.length - long < 5.0:
                        return 0.0
        return 1.0

    def _score_collision(self, traj_sim: np.ndarray, headings: np.ndarray, agents: List[Dict[str, Any]]) -> float:
        for t, pt in enumerate(traj_sim):
            ego_poly = self._ego_polygon(pt[0], pt[1], headings[t])
            dt_future = t * self.PLANNER_DT
            for agent in agents:
                pos = np.asarray(agent["position"][:2], dtype=float) + dt_future * np.asarray(agent["velocity"][:2], dtype=float)
                poly = self._ego_polygon(pos[0], pos[1], float(agent["heading"]), agent["length"], agent["width"])
                if ego_poly.intersects(poly):
                    return 0.0
        return 1.0

    def _score_ttc(self, traj_sim: np.ndarray, headings: np.ndarray, agents: List[Dict[str, Any]]) -> float:
        if not agents:
            return 1.0
        best_ttc = math.inf
        for t, pt in enumerate(traj_sim):
            ego_heading = headings[t]
            ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
            ego_speed = 0.0
            if t > 0:
                ego_speed = float(np.linalg.norm(traj_sim[t] - traj_sim[t - 1]) / self.PLANNER_DT)

            for agent in agents:
                agent_pos = np.asarray(agent["position"][:2], dtype=float) + t * self.PLANNER_DT * np.asarray(agent["velocity"][:2], dtype=float)
                rel = agent_pos - pt
                dist = float(np.linalg.norm(rel))
                if dist < 1e-3:
                    best_ttc = 0.0
                    continue
                rel_speed = ego_speed - float(np.dot(np.asarray(agent["velocity"][:2], dtype=float), ego_dir))
                if rel_speed <= 1e-3:
                    continue
                along = float(np.dot(rel, ego_dir))
                if along <= 0.0:
                    continue
                ttc = max(0.0, along / rel_speed)
                best_ttc = min(best_ttc, ttc)

        if best_ttc >= self.TTC_SAFE:
            return 1.0
        if best_ttc <= self.TTC_DANGER:
            return 0.0
        return (best_ttc - self.TTC_DANGER) / (self.TTC_SAFE - self.TTC_DANGER)

    def _score_lane_keeping(self, traj_sim: np.ndarray, headings: np.ndarray) -> float:
        if not self.all_lanes:
            return 1.0
        bad = 0
        for t, pt in enumerate(traj_sim):
            lane = self._best_lane(pt, headings[t])
            if lane is None:
                continue
            _, lateral = lane.local_coordinates(pt)
            if abs(lateral) > self.LK_LIMIT:
                bad += 1
            else:
                bad = 0
            if bad * self.PLANNER_DT > 2.0:
                return 0.0
        return 1.0

    def _score_hard_comfort(self, traj_world: np.ndarray, ego_state: Dict[str, Any]) -> float:
        speeds = [float(np.linalg.norm(np.asarray(ego_state["velocity"][:2], dtype=float)))]
        if len(traj_world) > 1:
            speeds.extend(np.linalg.norm(np.diff(traj_world, axis=0), axis=1) / self.PLANNER_DT)
        if len(speeds) < 3:
            return 1.0
        accels = np.diff(np.asarray(speeds)) / self.PLANNER_DT
        return 0.0 if float(np.max(np.abs(accels))) > self.MAX_HARD_ACCEL else 1.0

    def _score_extended_comfort(self, traj_world: np.ndarray, ego_state: Dict[str, Any]) -> float:
        initial_speed = float(np.linalg.norm(np.asarray(ego_state["velocity"][:2], dtype=float)))
        prev_acc = float(np.linalg.norm(np.asarray(ego_state["acceleration"][:2], dtype=float)))
        if len(traj_world) < 2:
            return 1.0

        speeds = [initial_speed]
        speeds.extend(np.linalg.norm(np.diff(traj_world, axis=0), axis=1) / self.PLANNER_DT)
        accels = np.diff(np.asarray(speeds)) / self.PLANNER_DT
        accel_delta = 0.0 if len(accels) == 0 else float(abs(accels[0] - prev_acc))

        initial_yaw_rate = float(abs(np.asarray(ego_state["angular_velocity"])[2])) if "angular_velocity" in ego_state else 0.0
        if len(traj_world) < 3:
            yaw_delta = initial_yaw_rate
        else:
            headings = np.arctan2(np.diff(traj_world[:, 1]), np.diff(traj_world[:, 0]))
            yaw_rates = np.abs(np.diff(headings) / self.PLANNER_DT)
            yaw_delta = float(abs(yaw_rates[0] - initial_yaw_rate)) if len(yaw_rates) else initial_yaw_rate

        return 1.0 if accel_delta <= self.MAX_EXT_ACCEL_DELTA and yaw_delta <= self.MAX_EXT_YAW_RATE_DELTA else 0.0

    def _score_progress(self, traj_sim: np.ndarray) -> float:
        if len(traj_sim) < 2:
            return 0.0
        dist = float(np.sum(np.linalg.norm(np.diff(traj_sim, axis=0), axis=1)))
        return min(1.0, dist / 20.0)

    def _nearest_lane(self, pt: np.ndarray) -> Optional[Any]:
        nearby = self._nearby_lane_indices(pt[0], pt[1])
        best_lane = None
        best_dist = math.inf
        for idx in nearby:
            lane = self.all_lanes[idx][0]
            dist = lane.distance(pt)
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
        return best_lane

    def _best_lane(self, pt: np.ndarray, heading: float) -> Optional[Any]:
        nearby = self._nearby_lane_indices(pt[0], pt[1], 10.0)
        best_lane = None
        best_score = math.inf
        for idx in nearby:
            lane = self.all_lanes[idx][0]
            long, _ = lane.local_coordinates(pt)
            lane_heading = lane.heading_theta_at(float(np.clip(long, 0.0, lane.length)))
            heading_err = abs(_wrap_to_pi(heading - lane_heading))
            score = lane.distance(pt) + 2.0 * heading_err
            if score < best_score:
                best_score = score
                best_lane = lane
        return best_lane


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

    def compute_acceleration(
        self,
        ego_speed: float,
        gap: float,
        rel_speed: float,
        target_speed: Optional[float] = None,
    ) -> float:
        """IDM formula. gap: bumper-to-bumper gap (m); rel_speed: ego - leader (positive = closing)."""
        if ego_speed < 0.0:
            ego_speed = 0.0
        v_desired = self.v_desired if target_speed is None else max(float(target_speed), 1e-3)
        v_ratio = ego_speed / max(v_desired, 1e-3)
        s_star = self.s_min + max(
            0.0,
            ego_speed * self.T + ego_speed * rel_speed / (2.0 * math.sqrt(self.a_max * self.b)),
        )
        gap = max(gap, 1e-3)
        a = self.a_max * (1.0 - v_ratio ** 10 - (s_star / gap) ** 2)
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
    LOOKAHEAD_BASE = 4.0
    LOOKAHEAD_SPEED_GAIN = 0.3

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
        speed_limit_mps: Optional[float] = None,
        stop_distance_m: Optional[float] = None,
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
            target_speed = self._compute_target_speed(
                path=path,
                x=x,
                y=y,
                speed=speed,
                step_idx=step,
                speed_limit_mps=speed_limit_mps,
                stop_distance_m=stop_distance_m,
                idm=idm,
            )
            a = idm.compute_acceleration(speed, gap, rel_speed, target_speed=target_speed)
            delta = self._pure_pursuit(x, y, heading, speed, path)

            x       += speed * math.cos(heading) * self.sim_dt
            y       += speed * math.sin(heading) * self.sim_dt
            heading += speed / self.wheelbase * math.tan(delta) * self.sim_dt
            speed    = float(np.clip(speed + a * self.sim_dt, 0.0, max(target_speed, 0.0)))

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

    def _compute_target_speed(
        self,
        path: np.ndarray,
        x: float,
        y: float,
        speed: float,
        step_idx: int,
        speed_limit_mps: Optional[float],
        stop_distance_m: Optional[float],
        idm: "_IDMController",
    ) -> float:
        target_speed = idm.v_desired
        if speed_limit_mps is not None and math.isfinite(speed_limit_mps):
            target_speed = min(target_speed, float(speed_limit_mps))

        # Slow down for curvature using a crude lateral-acceleration bound.
        curvature_speed = self._curvature_limited_speed(path, np.array([x, y], dtype=float))
        if curvature_speed is not None:
            target_speed = min(target_speed, curvature_speed)

        # Respect an upcoming stop line with a smooth braking profile.
        if stop_distance_m is not None and math.isfinite(stop_distance_m):
            travelled = step_idx * max(speed, 0.0) * self.sim_dt
            remaining = max(0.0, stop_distance_m - travelled)
            if remaining < 30.0:
                stop_speed = min(target_speed, math.sqrt(max(0.0, 2.0 * idm.b * max(remaining - 2.0, 0.0))))
                target_speed = min(target_speed, stop_speed)

        return max(0.0, target_speed)

    def _curvature_limited_speed(self, path: np.ndarray, position: np.ndarray) -> Optional[float]:
        if len(path) < 3:
            return None
        dists = np.linalg.norm(path - position[None, :], axis=1)
        idx = int(np.argmin(dists))
        if idx >= len(path) - 2:
            idx = max(0, len(path) - 3)
        pts = path[idx:idx + 3]
        if len(pts) < 3:
            return None

        a = float(np.linalg.norm(pts[1] - pts[0]))
        b = float(np.linalg.norm(pts[2] - pts[1]))
        c = float(np.linalg.norm(pts[2] - pts[0]))
        if min(a, b, c) < 1e-3:
            return None
        s = 0.5 * (a + b + c)
        area_sq = max(s * (s - a) * (s - b) * (s - c), 0.0)
        if area_sq <= 1e-8:
            return None
        area = math.sqrt(area_sq)
        radius = a * b * c / max(4.0 * area, 1e-6)
        if radius <= 1e-3:
            return None

        # 2.5 m/s^2 lateral acceleration target for conservative urban driving.
        return min(15.0, math.sqrt(2.5 * radius))


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
    Metrics: COL, DAC, DDC, TLC, EP, TTC, HC.
    Combined: (COL × DAC × DDC × TLC) × (5·EP + 5·TTC + 2·HC) / 12
    """

    PLANNER_DT      = 0.5
    DEVIATION_LIMIT = 0.5
    LK_WINDOW_S     = 2.0
    MAX_ACCEL       = 4.89
    TTC_SAFE        = 3.0
    TTC_DANGER      = 0.5

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
        ttc = self._score_ttc(traj_sim, headings, agents)
        hc  = self._score_hc(traj_sim)

        return float(col * dac * ddc * tlc * (5.0 * ep + 5.0 * ttc + 2.0 * hc) / 12.0)

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

    def _score_ttc(self, traj_sim: np.ndarray, headings: np.ndarray, agents: List[Dict[str, Any]]) -> float:
        if not agents:
            return 1.0
        best_ttc = math.inf
        for t, pt in enumerate(traj_sim):
            ego_heading = headings[t]
            ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
            ego_speed = 0.0
            if t > 0:
                ego_speed = float(np.linalg.norm(traj_sim[t] - traj_sim[t - 1]) / self.PLANNER_DT)
            for agent in agents:
                agent_pos = (np.asarray(agent["position"][:2], dtype=float)
                             + t * self.PLANNER_DT * np.asarray(agent["velocity"][:2], dtype=float))
                rel = agent_pos - pt
                dist = float(np.linalg.norm(rel))
                if dist < 1e-3:
                    best_ttc = 0.0
                    continue
                rel_speed = ego_speed - float(np.dot(np.asarray(agent["velocity"][:2], dtype=float), ego_dir))
                if rel_speed <= 1e-3:
                    continue
                along = float(np.dot(rel, ego_dir))
                if along <= 0.0:
                    continue
                best_ttc = min(best_ttc, max(0.0, along / rel_speed))
        if best_ttc >= self.TTC_SAFE:
            return 1.0
        if best_ttc <= self.TTC_DANGER:
            return 0.0
        return (best_ttc - self.TTC_DANGER) / (self.TTC_SAFE - self.TTC_DANGER)

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


class _RichCandidateScorer(_CandidateScorer):
    """
    Closed-loop scorer used when live route candidates are available.

    This extends the simpler candidate scoring with:
    - rollout-based collision checking
    - uncertainty inflation over time
    - explicit obstacle-detour preference for blocked-route cases
    """

    def score(
        self,
        traj_world: np.ndarray,
        ego_state: Dict[str, Any],
        agents: List[Dict[str, Any]],
        red_light_lanes: set,
        predicted_actors: Optional[List[Dict[str, Any]]] = None,
        scenario_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(traj_world) == 0:
            return 0.0

        traj_sim = traj_world + self.world_to_sim_offset
        headings = self._compute_headings(traj_sim, ego_state["heading"])

        col = self._score_collision_rollout(traj_sim, headings, predicted_actors or [])
        dac = self._score_dac(traj_sim)
        ddc = self._score_ddc(traj_sim, headings)
        tlc = self._score_tlc(traj_sim, headings, red_light_lanes)
        ttc = self._score_ttc_rollout(traj_sim, headings, predicted_actors or [])
        lk = self._score_lane_keeping(traj_sim, headings)
        hc = self._score_hard_comfort(traj_world, ego_state)
        ec = self._score_extended_comfort(traj_world, ego_state)
        progress = self._score_progress(traj_sim)

        gated = col * dac * ddc * tlc
        comfort = (hc + ec) / 2.0
        weighted = (5.0 * progress + 5.0 * ttc + 2.0 * comfort) / 12.0
        score = float(gated * weighted)

        if scenario_context:
            score += self._detour_bonus(traj_world, scenario_context)
            score += self._pedestrian_penalty(traj_world, predicted_actors or [], scenario_context)

        return float(np.clip(score, 0.0, 1.0))

    def _score_collision_rollout(self, traj_sim: np.ndarray, headings: np.ndarray, predicted_actors: List[Dict[str, Any]]) -> float:
        if not predicted_actors:
            return 1.0
        for t, pt in enumerate(traj_sim):
            ego_poly = self._ego_polygon(pt[0], pt[1], headings[t])
            actor_t = min(t, max((len(a["positions"]) - 1 for a in predicted_actors), default=0))
            for actor in predicted_actors:
                idx = min(actor_t, len(actor["positions"]) - 1)
                pos = actor["positions"][idx]
                heading = actor["headings"][idx]
                length = actor["lengths"][idx]
                width = actor["widths"][idx]
                poly = self._ego_polygon(pos[0] + self.world_to_sim_offset[0], pos[1] + self.world_to_sim_offset[1], heading, length, width)
                if ego_poly.intersects(poly):
                    return 0.0
        return 1.0

    def _score_ttc_rollout(self, traj_sim: np.ndarray, headings: np.ndarray, predicted_actors: List[Dict[str, Any]]) -> float:
        if not predicted_actors:
            return 1.0
        best_ttc = math.inf
        for t, pt in enumerate(traj_sim):
            ego_heading = headings[t]
            ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
            ego_speed = 0.0
            if t > 0:
                ego_speed = float(np.linalg.norm(traj_sim[t] - traj_sim[t - 1]) / self.PLANNER_DT)
            for actor in predicted_actors:
                idx = min(t, len(actor["positions"]) - 1)
                pos = np.asarray(actor["positions"][idx], dtype=float)
                rel = pos - (pt - self.world_to_sim_offset)
                if float(np.dot(rel, ego_dir)) <= 0.0:
                    continue
                dist = max(1e-3, float(np.linalg.norm(rel)) - 0.5 * actor["lengths"][idx])
                rel_speed = ego_speed - float(np.dot(np.asarray(actor["velocities"][idx], dtype=float), ego_dir))
                if rel_speed <= 1e-3:
                    continue
                best_ttc = min(best_ttc, dist / rel_speed)
        if best_ttc >= self.TTC_SAFE:
            return 1.0
        if best_ttc <= self.TTC_DANGER:
            return 0.0
        return (best_ttc - self.TTC_DANGER) / (self.TTC_SAFE - self.TTC_DANGER)

    def _detour_bonus(self, traj_world: np.ndarray, scenario_context: Dict[str, Any]) -> float:
        blocker = scenario_context.get("blocked_actor")
        if blocker is None:
            return 0.0
        blocker_pos = np.asarray(blocker["position"][:2], dtype=float)
        dists = np.linalg.norm(traj_world - blocker_pos[None, :], axis=1)
        lateral_clearance = float(np.min(dists))
        if lateral_clearance > 2.0 and float(blocker.get("distance", 1e9)) < 35.0:
            return 0.05
        return 0.0

    def _pedestrian_penalty(
        self,
        traj_world: np.ndarray,
        predicted_actors: List[Dict[str, Any]],
        scenario_context: Dict[str, Any],
    ) -> float:
        if not scenario_context.get("pedestrian_ahead", False):
            return 0.0
        worst = math.inf
        for actor in predicted_actors:
            if "PEDESTRIAN" not in str(actor.get("type", "")).upper():
                continue
            n = min(len(traj_world), len(actor["positions"]))
            if n == 0:
                continue
            d = np.linalg.norm(traj_world[:n] - actor["positions"][:n], axis=1)
            worst = min(worst, float(np.min(d)))
        if worst < 2.5:
            return -0.10
        if worst < 4.0:
            return -0.05
        return 0.0


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
        self._planner: Optional[_LiveRoutePlanner] = None
        self._idm: Optional[_IDMController] = None
        self._simulator: Optional[_KinematicBicycleSimulator] = None
        self._sdc_id: Optional[str] = None
        self._all_lanes: list = []
        self._all_lane_bounds: np.ndarray = np.empty((0, 4))
        self._world_to_sim_offset: np.ndarray = np.zeros(2)
        self._sim_to_world_offset: np.ndarray = np.zeros(2)
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
        self._planner = _LiveRoutePlanner(
            lookahead_m=cfg.get("lookahead_m", 60.0),
            sample_spacing=cfg.get("resample_spacing", 0.5),
            lateral_offsets=tuple(cfg.get("lateral_offsets", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])),
        )
        self._idm = _IDMController(
            v_desired=cfg.get("v_desired", 8.33),
            a_max=cfg.get("a_max", 2.0),
            b=cfg.get("b", 3.0),
            T=cfg.get("T", 1.0),
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
        # Planning is state-based, but visualization needs at least a front view.
        return {
            "CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"],
            "CAM_THIRD_PERSON": NAVSIM_CAM_CONFIGS["CAM_THIRD_PERSON"],
        }

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
            ego_position_sim = np.asarray(self._env.agent.position, dtype=float)
            self._sim_to_world_offset = np.asarray(ego_state["position"][:2], dtype=float) - ego_position_sim[:2]
            self._world_to_sim_offset = -self._sim_to_world_offset
        else:
            ego_position_sim = np.asarray(ego_state["position"][:2], dtype=float)
            self._sim_to_world_offset = np.zeros(2, dtype=float)
            self._world_to_sim_offset = np.zeros(2, dtype=float)

        target_waypoint_world = None
        if ego_state.get("waypoint") is not None:
            target_waypoint_world = np.asarray(ego_state["waypoint"][:2], dtype=float)

        candidate_paths = []
        if self._env is not None and self._planner is not None:
            candidate_paths = self._planner.build_candidate_paths(
                env=self._env,
                ego_state=ego_state,
                ego_position_sim=ego_position_sim[:2],
                sim_to_world_offset=self._sim_to_world_offset,
                target_waypoint_world=target_waypoint_world,
            )

        agents = self._extract_agents_live(ego_state)
        if not agents:
            agents = self._extract_agents(scenario_data, frame_id)

        red_light_lanes = self._extract_red_light_lanes_live()
        if not red_light_lanes:
            red_light_lanes = self._extract_tl_states(scenario_data, frame_id)

        centerline = self._extractor.extract_centerline(
            scenario_data, self._sdc_id, ego_state["position"], frame_id
        )

        return {
            "ego_state": ego_state,
            "agents": agents,
            "red_light_lanes": red_light_lanes,
            "candidate_paths": candidate_paths,
            "centerline": centerline,
            "frame_id": frame_id,
        }

    def run_inference(self, model_input: Any) -> Any:
        ego_state = model_input["ego_state"]
        agents = model_input["agents"]
        red_light_lanes = model_input["red_light_lanes"]
        centerline = model_input["centerline"]

        candidates = model_input.get("candidate_paths") or []
        if not candidates:
            candidates = self._extractor.generate_candidates(centerline)
        if candidates:
            candidates = self._augment_with_obstacle_detours(candidates, ego_state, agents)
        if not candidates:
            return self._fallback_straight(ego_state)

        use_live_scorer = model_input.get("candidate_paths") not in (None, []) and len(model_input.get("candidate_paths") or []) > 0
        if use_live_scorer:
            scorer = _RichCandidateScorer(
                all_lanes=self._all_lanes,
                all_lane_bounds=self._all_lane_bounds,
                world_to_sim_offset=self._world_to_sim_offset,
            )
        else:
            scorer = _PDMLiteScorer(
                all_lanes=self._all_lanes,
                all_lane_bounds=self._all_lane_bounds,
                world_to_sim_offset=self._world_to_sim_offset,
            )

        simulated: List[np.ndarray] = []
        scores: List[float] = []
        predicted_actors = self._predict_actor_rollouts(agents)
        scenario_context = self._build_scenario_context(ego_state, agents, candidates)
        speed_limit_mps = self._estimate_speed_limit_mps()
        if speed_limit_mps is not None and speed_limit_mps > 1.0:
            self._idm.v_desired = 0.95 * speed_limit_mps
        for path in candidates:
            stop_distance_m = self._estimate_stop_distance(path, red_light_lanes)
            traj = self._simulator.simulate(
                path,
                ego_state,
                agents,
                self._idm,
                speed_limit_mps=speed_limit_mps,
                stop_distance_m=stop_distance_m,
            )
            simulated.append(traj)
            if use_live_scorer:
                scores.append(
                    scorer.score(
                        traj,
                        ego_state,
                        agents,
                        red_light_lanes,
                        predicted_actors=predicted_actors,
                        scenario_context=scenario_context,
                    )
                )
            else:
                scores.append(scorer.score(traj, red_light_lanes, agents))

        best_idx = int(np.argmax(scores))
        print(f"[PDM-Lite] scorer={'rich' if use_live_scorer else 'pdm_lite'} "
              f"candidates={len(candidates)} best_idx={best_idx} "
              f"best_score={scores[best_idx]:.4f}")
        return {
            "trajectory_world": simulated[best_idx],
            "all_simulated_world": simulated,
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
        result = {"trajectory": np.stack([lateral, forward], axis=1)}

        # Emit explicit target speed for the longitudinal PID, mirroring CARLA PDM-Lite.
        # Computed from the planner's first 0.5 s of intended motion (waypoint_dt = 0.5).
        if len(traj_world) > 0:
            first_disp = float(np.linalg.norm(traj_world[0] - ego_state["position"][:2]))
            result["target_speed"] = first_disp / 0.5

        all_sim = model_output.get("all_simulated_world")
        if all_sim and len(all_sim) > 0:
            ego_xy = ego_state["position"][:2]
            candidates_ego = []
            for cand_world in all_sim:
                cand_world = np.asarray(cand_world)
                d = cand_world - ego_xy
                lat = sin_h * d[:, 0] + cos_h * d[:, 1]
                fwd = cos_h * d[:, 0] - sin_h * d[:, 1]
                candidates_ego.append(np.stack([lat, fwd], axis=1))
            result["trajectory_topk"] = np.array(candidates_ego)
            result["topk_scores"] = np.array(model_output["all_scores"])

        return result

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0

    def get_controller_config(self) -> Dict[str, Any]:
        # Approximate the reported "velocity-scaled PID" behavior using the
        # PID-based controller path with tuned gains instead of the generic
        # evaluator defaults.
        return {
            "type": "pid",
            "params": {
                "turn_KP": 1.7,
                "turn_KI": 0.01,
                "turn_KD": 1.4,
                "speed_KP": 0.3,
                "speed_KI": 0.002,
                "speed_KD": 0.05,
                "aim_dist": 5.5,
            },
        }

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
            if track["type"] not in ("VEHICLE", "CYCLIST", "PEDESTRIAN"):
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

            length, width = _infer_box_size(str(track["type"]))

            agents.append({
                "position": pos.astype(float),
                "velocity": vel,
                "heading":  float(track["state"]["heading"][frame_id]),
                "length": length,
                "width": width,
                "type":     track["type"],
            })
        return agents

    def _extract_agents_live(self, ego_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._env is None:
            return []

        ego_obj = getattr(self._env, "agent", None)
        ego_xy_world = np.asarray(ego_state["position"][:2], dtype=float)
        agents: List[Dict[str, Any]] = []
        for obj in self._env.engine.get_objects().values():
            if obj is ego_obj:
                continue

            cls_name = obj.__class__.__name__.lower()
            if "trafficlight" in cls_name or "lane" in cls_name or "map" in cls_name:
                continue
            if not hasattr(obj, "position"):
                continue

            pos_sim = np.asarray(obj.position, dtype=float)
            pos_world = pos_sim[:2] + self._sim_to_world_offset
            if float(np.linalg.norm(pos_world - ego_xy_world)) > 60.0:
                continue

            if hasattr(obj, "velocity"):
                vel = np.asarray(obj.velocity, dtype=float)[:2]
            else:
                vel = np.zeros(2, dtype=float)

            heading = float(getattr(obj, "heading_theta", 0.0))
            length = float(getattr(obj, "LENGTH", VEHICLE_LENGTH))
            width = float(getattr(obj, "WIDTH", VEHICLE_WIDTH))
            if length <= 0.0 or width <= 0.0:
                length, width = _infer_box_size(cls_name)

            agents.append({
                "position": pos_world,
                "velocity": vel,
                "heading": heading,
                "length": length,
                "width": width,
                "type": cls_name.upper(),
            })
        return agents

    def _extract_tl_states(self, scenario_data: Dict[str, Any], frame_id: int) -> set:
        red: set = set()
        for lane_id, tl_data in scenario_data.get("dynamic_map_states", {}).items():
            s_list = tl_data["state"]["object_state"]
            if frame_id < len(s_list) and s_list[frame_id] == "LANE_STATE_STOP":
                red.add(str(lane_id))
        return red

    def _extract_red_light_lanes_live(self) -> set:
        if self._env is None:
            return set()

        red_lanes = set()
        for obj in self._env.engine.get_objects().values():
            cls_name = obj.__class__.__name__.lower()
            if "trafficlight" not in cls_name:
                continue
            status = str(getattr(obj, "status", "")).lower()
            if "red" not in status:
                continue
            lane = getattr(obj, "lane", None)
            if lane is not None:
                red_lanes.add(str(getattr(lane, "index", "")))
        return red_lanes

    def _augment_with_obstacle_detours(
        self,
        candidates: List[np.ndarray],
        ego_state: Dict[str, Any],
        agents: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        if not candidates or not agents:
            return candidates

        augmented: List[np.ndarray] = list(candidates)
        blockers = []
        ego_pos = np.asarray(ego_state["position"][:2], dtype=float)
        ego_heading = float(ego_state.get("heading", 0.0))
        ego_fwd = np.array([math.cos(ego_heading), math.sin(ego_heading)])
        for agent in agents:
            rel = np.asarray(agent["position"][:2], dtype=float) - ego_pos
            distance = float(np.linalg.norm(rel))
            lon = float(np.dot(rel, ego_fwd))
            speed = float(np.linalg.norm(np.asarray(agent["velocity"][:2], dtype=float)))
            if distance < 35.0 and speed < 3.0 and lon > 0.0:
                blockers.append((distance, agent))
        blockers.sort(key=lambda x: x[0])

        for _, blocker in blockers[:2]:
            for path in candidates[:4]:
                detours = self._generate_detour_paths(path, blocker)
                augmented.extend(detours)

        # Keep candidates diverse but bounded.
        return augmented[:16]

    def _generate_detour_paths(self, path: np.ndarray, blocker: Dict[str, Any]) -> List[np.ndarray]:
        if len(path) < 8:
            return []
        blocker_pos = np.asarray(blocker["position"][:2], dtype=float)
        dists = np.linalg.norm(path - blocker_pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > 4.0 or idx < 2 or idx > len(path) - 3:
            return []

        tangents = np.diff(path, axis=0)
        tangents = np.vstack([tangents, tangents[-1:]])
        tangents = tangents / np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

        local_normal = normals[idx]
        blocker_side = float(np.dot(blocker_pos - path[idx], local_normal))
        preferred_sign = -1.0 if blocker_side >= 0.0 else 1.0
        length = float(blocker.get("length", VEHICLE_LENGTH))
        width = float(blocker.get("width", VEHICLE_WIDTH))
        clearance = max(2.0, 0.5 * width + 1.5)
        max_offset = min(4.0, clearance + 0.5 * length / 4.0)

        start = max(0, idx - 6)
        peak = idx
        end = min(len(path) - 1, idx + 8)
        offsets = [preferred_sign * max_offset, preferred_sign * (max_offset + 1.2)]
        out = []
        for offset in offsets:
            shifted = path.copy()
            for i in range(start, end + 1):
                if i <= peak:
                    phase = (i - start) / max(peak - start, 1)
                    weight = 0.5 - 0.5 * math.cos(math.pi * phase)
                else:
                    phase = (i - peak) / max(end - peak, 1)
                    weight = 0.5 + 0.5 * math.cos(math.pi * phase)
                shifted[i] = shifted[i] + normals[i] * (offset * weight)
            out.append(shifted)
        return out

    def _predict_actor_rollouts(self, agents: List[Dict[str, Any]], horizon_s: float = 2.0, dt: float = 0.5) -> List[Dict[str, Any]]:
        num_steps = max(1, int(round(horizon_s / dt)))
        predictions: List[Dict[str, Any]] = []
        for agent in agents:
            pos0 = np.asarray(agent["position"][:2], dtype=float)
            vel = np.asarray(agent["velocity"][:2], dtype=float)
            speed = float(np.linalg.norm(vel))
            heading = float(agent.get("heading", math.atan2(vel[1], vel[0]) if speed > 1e-3 else 0.0))
            positions = []
            headings = []
            velocities = []
            lengths = []
            widths = []
            for step in range(num_steps):
                t = step * dt
                if "PEDESTRIAN" in str(agent.get("type", "")).upper():
                    pos = pos0 + vel * t
                    scale = 1.0 + 0.5 * (t / max(horizon_s, 1e-3))
                else:
                    pos = pos0 + vel * t
                    scale = 1.0 + 1.0 * (t / max(horizon_s, 1e-3))
                positions.append(pos)
                headings.append(heading)
                velocities.append(vel)
                lengths.append(float(agent.get("length", VEHICLE_LENGTH)) * scale)
                widths.append(float(agent.get("width", VEHICLE_WIDTH)) * scale)
            predictions.append({
                "type": agent.get("type", "UNKNOWN"),
                "positions": np.asarray(positions, dtype=float),
                "headings": np.asarray(headings, dtype=float),
                "velocities": np.asarray(velocities, dtype=float),
                "lengths": np.asarray(lengths, dtype=float),
                "widths": np.asarray(widths, dtype=float),
            })
        return predictions

    def _build_scenario_context(
        self,
        ego_state: Dict[str, Any],
        agents: List[Dict[str, Any]],
        candidates: List[np.ndarray],
    ) -> Dict[str, Any]:
        ego_pos = np.asarray(ego_state["position"][:2], dtype=float)
        blocked_actor = None
        best_dist = math.inf
        pedestrian_ahead = False
        ego_heading = float(ego_state.get("heading", 0.0))
        ego_fwd = np.array([math.cos(ego_heading), math.sin(ego_heading)])
        for agent in agents:
            rel = np.asarray(agent["position"][:2], dtype=float) - ego_pos
            dist = float(np.linalg.norm(rel))
            if "PEDESTRIAN" in str(agent.get("type", "")).upper() and dist < 20.0:
                pedestrian_ahead = True
            lon = float(np.dot(rel, ego_fwd))
            if lon <= 0.0:
                continue
            if dist < best_dist and float(np.linalg.norm(np.asarray(agent["velocity"][:2], dtype=float))) < 3.0:
                best_dist = dist
                blocked_actor = dict(agent)
                blocked_actor["distance"] = dist
        return {
            "blocked_actor": blocked_actor,
            "pedestrian_ahead": pedestrian_ahead,
            "num_candidates": len(candidates),
        }

    def _estimate_speed_limit_mps(self) -> Optional[float]:
        if self._env is None:
            return None
        lane = getattr(getattr(self._env, "agent", None), "lane", None)
        if lane is None:
            return None
        speed_limit = getattr(lane, "speed_limit", None)
        if speed_limit is None:
            return None
        # MetaDrive lanes store speed limits in km/h.
        return float(speed_limit) / 3.6

    def _estimate_stop_distance(self, path: np.ndarray, red_light_lanes: set) -> Optional[float]:
        if not red_light_lanes or not self._all_lanes or len(path) == 0:
            return None
        cumulative = 0.0
        for i in range(len(path)):
            if i > 0:
                cumulative += float(np.linalg.norm(path[i] - path[i - 1]))
            nearby = self._nearby_lane_indices_world(path[i][0], path[i][1], radius=4.0)
            for idx in nearby:
                lane = self._all_lanes[idx][0]
                lane_id = str(getattr(lane, "index", ""))
                if lane_id in red_light_lanes:
                    return cumulative
        return None

    def _nearby_lane_indices_world(self, x: float, y: float, radius: float = 15.0) -> List[int]:
        if len(self._all_lane_bounds) == 0:
            return []
        b = self._all_lane_bounds
        mask = (
            (b[:, 0] - radius < x + self._world_to_sim_offset[0]) &
            (x + self._world_to_sim_offset[0] < b[:, 2] + radius) &
            (b[:, 1] - radius < y + self._world_to_sim_offset[1]) &
            (y + self._world_to_sim_offset[1] < b[:, 3] + radius)
        )
        return list(np.where(mask)[0])

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

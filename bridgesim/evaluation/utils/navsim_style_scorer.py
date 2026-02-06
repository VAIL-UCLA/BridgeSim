"""
NavSim-Style EPDMS Scorer for Closed-Loop Evaluation.

This module implements a stateful scorer that computes EPDMS metrics
incrementally during simulation, matching NavSim's metric calculation logic.

Key Features:
- Per-frame scoring during simulation (not batch at end)
- Detailed collision classification
- History comfort with past trajectory padding
- Extended comfort for plan consistency
- Progress computed relative to GT trajectory

References:
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
- navsim/planning/simulation/planner/pdm_planner/scoring/scene_aggregator.py
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from shapely.geometry import Polygon, Point

from .comfort_metrics import (
    StateIndex,
    ego_state_to_state_array,
    compute_history_comfort,
    compute_extended_comfort,
    STOPPED_SPEED_THRESHOLD,
)
from .collision_classifier import (
    CollisionType,
    TrackedObject,
    create_vehicle_polygon,
    check_collision_with_agents,
    is_at_fault_collision,
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
)
from .metric_cache_placeholder import (
    MetricCache,
    GTTrajectoryCache,
)


# === Metric Weights (from NavSim pdm_scorer.py:44-48) ===
W_PROGRESS = 5.0
W_TTC = 5.0
W_LANE_KEEPING = 2.0
W_HISTORY_COMFORT = 2.0
W_EXTENDED_COMFORT = 2.0

# === Metric Thresholds ===
# Driving direction (NavSim pdm_scorer.py:52-54)
DRIVING_DIRECTION_HORIZON = 1.0  # seconds
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # meters
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # meters

# TTC (NavSim pdm_scorer.py:56-57)
TTC_HORIZON = 1.0  # seconds

# Progress (NavSim pdm_scorer.py:58)
PROGRESS_DISTANCE_THRESHOLD = 5.0  # meters

# Lane keeping (NavSim pdm_scorer.py:59-60)
LANE_KEEPING_DEVIATION_LIMIT = 0.5  # meters
LANE_KEEPING_HORIZON_WINDOW = 2.0  # seconds


@dataclass
class ScorerConfig:
    """Configuration for the NavSim-style scorer."""
    # Metric weights
    progress_weight: float = W_PROGRESS
    ttc_weight: float = W_TTC
    lane_keeping_weight: float = W_LANE_KEEPING
    history_comfort_weight: float = W_HISTORY_COMFORT
    extended_comfort_weight: float = W_EXTENDED_COMFORT

    # Thresholds
    driving_direction_horizon: float = DRIVING_DIRECTION_HORIZON
    driving_direction_compliance_threshold: float = DRIVING_DIRECTION_COMPLIANCE_THRESHOLD
    driving_direction_violation_threshold: float = DRIVING_DIRECTION_VIOLATION_THRESHOLD
    ttc_horizon: float = TTC_HORIZON
    progress_distance_threshold: float = PROGRESS_DISTANCE_THRESHOLD
    lane_keeping_deviation_limit: float = LANE_KEEPING_DEVIATION_LIMIT
    lane_keeping_horizon_window: float = LANE_KEEPING_HORIZON_WINDOW

    # History settings
    history_seconds: float = 2.0  # Seconds of history to keep for comfort

    # Simulation settings
    dt: float = 0.1  # Simulation timestep

    @property
    def history_frames(self) -> int:
        return int(self.history_seconds / self.dt)

    @property
    def weighted_metrics_array(self) -> np.ndarray:
        return np.array([
            self.progress_weight,
            self.ttc_weight,
            self.lane_keeping_weight,
            self.history_comfort_weight,
            self.extended_comfort_weight,
        ])


@dataclass
class FrameMetrics:
    """Metrics computed for a single frame."""
    frame_idx: int

    # Binary metrics (0 or 1)
    no_collision: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0

    # Weighted metrics (0 to 1)
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    extended_comfort: float = 1.0

    # Additional details
    collision_type: Optional[CollisionType] = None
    collision_details: Optional[Dict] = None
    lateral_deviation: float = 0.0
    progress_distance: float = 0.0


@dataclass
class AggregatedMetrics:
    """Final aggregated metrics across all frames."""
    # Binary metrics (product across frames)
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0

    # Weighted metrics (aggregated values)
    ego_progress: float = 0.0
    time_to_collision_within_bound: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    extended_comfort: float = 1.0

    # Final score
    score: float = 0.0

    # Metadata
    num_frames: int = 0
    num_collision_frames: int = 0
    valid: bool = True


class NavSimStyleScorer:
    """
    Stateful scorer for closed-loop evaluation.

    This scorer maintains state across frames and computes metrics
    incrementally during simulation.
    """

    def __init__(
        self,
        scenario_data: dict,
        env: Any,
        config: Optional[ScorerConfig] = None,
    ):
        """
        Initialize the scorer.

        Args:
            scenario_data: Loaded scenario pickle data
            env: MetaDrive environment instance
            config: Scorer configuration (uses defaults if None)
        """
        self.scenario_data = scenario_data
        self.env = env
        self.config = config or ScorerConfig()

        self.sdc_id = scenario_data['metadata']['sdc_id']
        self.scenario_length = scenario_data['length']

        # Build metric cache (GT trajectory, etc.)
        self.metric_cache = MetricCache.from_scenario_data(
            scenario_data,
            scenario_name=scenario_data.get('scenario_id', 'unknown'),
        )

        # === Accumulated State ===
        # History of ego states for comfort calculation
        self.ego_state_history: deque = deque(maxlen=self.config.history_frames)

        # History of plan trajectories for extended comfort
        self.plan_trajectory_history: deque = deque(maxlen=2)

        # Previous plan states (for extended comfort comparison)
        self.prev_plan_states: Optional[np.ndarray] = None

        # Accumulated progress along GT trajectory
        self.start_progress: Optional[float] = None
        self.current_progress: float = 0.0

        # === Per-Frame Violation Tracking ===
        self.collision_occurred: bool = False
        self.collision_frames: List[int] = []
        self.collided_track_ids: List[str] = []

        self.dac_violated: bool = False
        self.ddc_violation_progress: float = 0.0
        self.tlc_violated: bool = False
        self.ttc_violated: bool = False

        self.lk_consecutive_violation: int = 0
        self.lk_violated: bool = False

        self.hc_violated: bool = False
        self.ec_violated: bool = False

        # === Frame Results ===
        self.frame_metrics: List[FrameMetrics] = []

        # === Map/Lane Data (from MetaDrive) ===
        self._all_lanes = []
        self._init_map_data()

        print(f"[NavSimStyleScorer] Initialized for scenario with {self.scenario_length} frames")
        print(f"  History frames: {self.config.history_frames}")
        print(f"  GT trajectory length: {self.metric_cache.gt_trajectory.total_length:.1f}m")

    def _init_map_data(self):
        """Initialize map data from MetaDrive environment."""
        if self.env and self.env.engine.map_manager.current_map:
            road_network = self.env.engine.map_manager.current_map.road_network
            if hasattr(road_network, 'get_all_lanes'):
                for lane in road_network.get_all_lanes():
                    if hasattr(lane, 'shapely_polygon'):
                        self._all_lanes.append((lane, lane.shapely_polygon))
            else:
                for start_node, end_dict in road_network.graph.items():
                    for end_node, lanes in end_dict.items():
                        for lane in lanes:
                            if hasattr(lane, 'shapely_polygon'):
                                self._all_lanes.append((lane, lane.shapely_polygon))

    def _get_nearby_lanes(self, x: float, y: float, radius: float = 50.0) -> List:
        """Query lanes near a position."""
        nearby = []
        for lane, poly in self._all_lanes:
            minx, miny, maxx, maxy = poly.bounds
            if (minx - radius < x < maxx + radius) and (miny - radius < y < maxy + radius):
                nearby.append(lane)
        return nearby

    def _get_agents_at_frame(self, frame_idx: int) -> List[TrackedObject]:
        """
        Get tracked objects at a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            List of TrackedObject instances
        """
        agents = []
        tracks = self.scenario_data['tracks']

        for track_id, track in tracks.items():
            if track_id == self.sdc_id:
                continue

            # Check if track is valid at this frame
            if frame_idx >= len(track['state']['valid']):
                continue
            if not track['state']['valid'][frame_idx]:
                continue
            # TODO: pedestrian?
            obj_type = track.get('type', 'VEHICLE')
            position = track['state']['position'][frame_idx][:2]
            heading = track['state']['heading'][frame_idx]

            # Get velocity if available
            if 'velocity' in track['state']:
                velocity = track['state']['velocity'][frame_idx][:2]
            else:
                velocity = np.zeros(2)

            # Get dimensions if available
            length = track.get('length', 4.5)
            width = track.get('width', 1.8)

            agents.append(TrackedObject(
                track_id=track_id,
                object_type=obj_type,
                position=position,
                heading=heading,
                velocity=velocity,
                length=length,
                width=width,
            ))

        return agents

    def _check_drivable_area(
        self,
        ego_polygon: Polygon,
        nearby_lanes: List,
    ) -> bool:
        """
        Check if ego is within drivable area.

        Reference: NavSim pdm_scorer.py:422-429

        Returns:
            True if compliant (all corners in drivable area)
        """
        if not nearby_lanes:
            return True  # No lanes to check against

        corners = list(ego_polygon.exterior.coords)[:-1]  # Exclude closing point

        corners_in_drivable = 0
        for corner in corners:
            corner_point = Point(corner)
            for lane in nearby_lanes:
                if lane.shapely_polygon.contains(corner_point):
                    corners_in_drivable += 1
                    break

        return corners_in_drivable >= 4  # All corners must be in drivable area

    def _check_driving_direction(
        self,
        ego_position: np.ndarray,
        ego_heading: float,
        nearby_lanes: List,
        frame_idx: int,
    ) -> Tuple[float, float]:
        """
        Check driving direction compliance.

        Reference: NavSim pdm_scorer.py:431-474

        Returns:
            Tuple of (compliance_score, violation_progress)
        """
        # Get GT heading at this frame for reference
        gt_heading = self.metric_cache.gt_trajectory.headings[
            min(frame_idx, len(self.metric_cache.gt_trajectory.headings) - 1)
        ]

        # Compute heading difference
        heading_diff = np.abs(ego_heading - gt_heading)
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # If heading difference > 90 degrees, consider it oncoming traffic
        if np.abs(heading_diff) > np.pi / 2:
            # TODO: Accumulate distance traveled in wrong direction
            # For now, use simplified check
            return 0.0, np.abs(heading_diff)

        return 1.0, 0.0

    def _check_traffic_light(
        self,
        ego_polygon: Polygon,
        frame_idx: int,
        ego_speed: float,
    ) -> bool:
        """
        Check traffic light compliance.

        Reference: NavSim pdm_scorer.py:582-613

        Returns:
            True if compliant (not running red light)
        """
        traffic_lights = self.scenario_data.get('dynamic_map_states', {})
        if not traffic_lights:
            return True

        # Get red light lane IDs at current frame
        red_lane_ids = set()
        for lane_id, tl_data in traffic_lights.items():
            s_list = tl_data['state']['object_state']
            if frame_idx < len(s_list) and s_list[frame_idx] == "LANE_STATE_STOP":
                red_lane_ids.add(str(lane_id))

        if not red_lane_ids:
            return True

        # Check if ego intersects any red light lane while moving
        if ego_speed < 1.0:
            return True  # Stopped, so OK

        # TODO: More sophisticated check using lane polygons
        # For now, simplified check
        return True

    def _compute_ttc(
        self,
        ego_position: np.ndarray,
        ego_heading: float,
        ego_velocity: np.ndarray,
        agents: List[TrackedObject],
    ) -> bool:
        """
        Compute time-to-collision metric.

        Reference: NavSim pdm_scorer.py:492-580

        Returns:
            True if TTC is within safe bounds (no imminent collision)
        """
        ego_speed = np.linalg.norm(ego_velocity)
        if ego_speed < STOPPED_SPEED_THRESHOLD:
            return True  # Stopped, no TTC concern

        # Project ego forward over TTC horizon
        ttc_steps = int(self.config.ttc_horizon / self.config.dt)

        for step in range(1, ttc_steps + 1):
            # Project ego position
            t = step * self.config.dt
            projected_position = ego_position + ego_velocity * t

            # Create projected ego polygon
            projected_polygon = create_vehicle_polygon(
                projected_position[0], projected_position[1],
                ego_heading, VEHICLE_LENGTH, VEHICLE_WIDTH
            )

            # Check intersection with agents
            for agent in agents:
                # Simple: assume agent stays in place (could use agent velocity for better estimate)
                agent_polygon = agent.get_polygon()

                if projected_polygon.intersects(agent_polygon):
                    # Potential collision - but only if agent is ahead
                    delta = agent.position - ego_position
                    forward_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])
                    if np.dot(delta, forward_dir) > 0:  # Agent is ahead
                        return False  # TTC violation

        return True

    def _ego_state_to_array(self, ego_state: Dict) -> np.ndarray:
        """Convert ego state dict to state array."""
        return ego_state_to_state_array(
            position=ego_state['position'],
            heading=ego_state['heading'],
            velocity=ego_state['velocity'],
            acceleration=ego_state.get('acceleration'),
        )

    def update_history(self, ego_state: Dict) -> None:
        """
        Update ego state history buffer.

        Args:
            ego_state: Current ego state dict with position, heading, velocity, etc.
        """
        state_array = self._ego_state_to_array(ego_state)
        self.ego_state_history.append(state_array)

    def score_frame(
        self,
        ego_state: Dict,
        plan_traj: np.ndarray,
        frame_idx: int,
    ) -> FrameMetrics:
        """
        Score a single frame during simulation.

        Args:
            ego_state: Current ego state dict with:
                - position: (3,) [x, y, z]
                - heading: float
                - velocity: (3,) [vx, vy, vz]
                - acceleration: (3,) [ax, ay, az]
                - speed: float
            plan_traj: Model's predicted trajectory (N, 2) in ego frame
            frame_idx: Current frame index

        Returns:
            FrameMetrics for this frame
        """
        metrics = FrameMetrics(frame_idx=frame_idx)

        # Extract ego state components
        ego_position = ego_state['position'][:2]
        ego_heading = ego_state['heading']
        ego_velocity = ego_state['velocity'][:2]
        ego_speed = ego_state.get('speed', np.linalg.norm(ego_velocity))

        # Create ego polygon
        ego_polygon = create_vehicle_polygon(
            ego_position[0], ego_position[1],
            ego_heading, VEHICLE_LENGTH, VEHICLE_WIDTH
        )

        # Get agents at this frame
        agents = self._get_agents_at_frame(frame_idx)

        # Get nearby lanes
        nearby_lanes = self._get_nearby_lanes(ego_position[0], ego_position[1])

        # === 1. No Collision (NC) ===
        collision_result = check_collision_with_agents(
            ego_position, ego_heading, ego_velocity,
            ego_polygon, agents,
            already_collided_ids=self.collided_track_ids,
            ego_in_wrong_area=self.dac_violated,
        )

        if collision_result.has_collision:
            metrics.no_collision = collision_result.collision_score
            metrics.collision_type = collision_result.collision_type
            metrics.collision_details = collision_result.details

            if collision_result.is_at_fault:
                self.collision_occurred = True
                self.collision_frames.append(frame_idx)

            if collision_result.collided_object_id:
                self.collided_track_ids.append(collision_result.collided_object_id)

        # === 2. Drivable Area Compliance (DAC) ===
        dac_ok = self._check_drivable_area(ego_polygon, nearby_lanes)
        if not dac_ok:
            metrics.drivable_area_compliance = 0.0
            self.dac_violated = True

        # === 3. Driving Direction Compliance (DDC) ===
        ddc_score, ddc_violation = self._check_driving_direction(
            ego_position, ego_heading, nearby_lanes, frame_idx
        )
        metrics.driving_direction_compliance = ddc_score
        self.ddc_violation_progress += ddc_violation

        # === 4. Traffic Light Compliance (TLC) ===
        tlc_ok = self._check_traffic_light(ego_polygon, frame_idx, ego_speed)
        if not tlc_ok:
            metrics.traffic_light_compliance = 0.0
            self.tlc_violated = True

        # === 5. Progress (EP) ===
        current_progress = self.metric_cache.gt_trajectory.get_progress_at_position(ego_position)
        if self.start_progress is None:
            self.start_progress = current_progress
        self.current_progress = current_progress
        metrics.progress_distance = current_progress - self.start_progress

        # === 6. Time-to-Collision (TTC) ===
        ttc_ok = self._compute_ttc(ego_position, ego_heading, ego_velocity, agents)
        if not ttc_ok:
            metrics.time_to_collision = 0.0
            self.ttc_violated = True

        # === 7. Lane Keeping (LK) ===
        lateral_dev = self.metric_cache.gt_trajectory.get_lateral_deviation(ego_position, frame_idx)
        metrics.lateral_deviation = lateral_dev

        if lateral_dev > self.config.lane_keeping_deviation_limit:
            self.lk_consecutive_violation += 1
        else:
            self.lk_consecutive_violation = 0

        lk_violation_frames = int(self.config.lane_keeping_horizon_window / self.config.dt)
        if self.lk_consecutive_violation >= lk_violation_frames:
            metrics.lane_keeping = 0.0
            self.lk_violated = True

        # === 8. History Comfort (HC) ===
        self.update_history(ego_state)

        if len(self.ego_state_history) >= 2:
            history_array = np.array(list(self.ego_state_history))
            current_array = self._ego_state_to_array(ego_state)[None, :]

            hc_ok, hc_details = compute_history_comfort(
                current_array, history_array[None, :], self.config.dt
            )
            if not hc_ok:
                metrics.history_comfort = 0.0
                self.hc_violated = True

        # === 9. Extended Comfort (EC) ===
        # Convert plan trajectory to state array for comparison
        # TODO: This needs plan trajectory with velocities/accelerations
        # For now, skip extended comfort or use simplified version
        if self.prev_plan_states is not None and len(plan_traj) > 0:
            # Simplified: just check if we have previous plan
            # Full implementation would convert plan_traj to states and compare
            pass

        # Store current plan for next frame comparison
        # self.prev_plan_states = ...

        # Store frame metrics
        self.frame_metrics.append(metrics)

        return metrics

    def finalize(self) -> AggregatedMetrics:
        """
        Finalize scoring and compute aggregated metrics.

        Returns:
            AggregatedMetrics with final scores
        """
        result = AggregatedMetrics()
        result.num_frames = len(self.frame_metrics)

        if result.num_frames == 0:
            result.valid = False
            return result

        # === Aggregate Binary Metrics ===
        # These are 0 if violated at ANY frame
        result.no_at_fault_collisions = 0.0 if self.collision_occurred else 1.0
        result.drivable_area_compliance = 0.0 if self.dac_violated else 1.0
        result.traffic_light_compliance = 0.0 if self.tlc_violated else 1.0

        # DDC: Based on accumulated violation progress
        if self.ddc_violation_progress < self.config.driving_direction_compliance_threshold:
            result.driving_direction_compliance = 1.0
        elif self.ddc_violation_progress < self.config.driving_direction_violation_threshold:
            result.driving_direction_compliance = 0.5
        else:
            result.driving_direction_compliance = 0.0

        # === Aggregate Weighted Metrics ===
        # Progress: Normalize by GT trajectory length or threshold
        total_progress = self.current_progress - (self.start_progress or 0)
        max_progress = max(self.metric_cache.gt_trajectory.total_length, self.config.progress_distance_threshold)
        result.ego_progress = min(total_progress / max_progress, 1.0)

        # TTC: 0 if violated at any frame
        result.time_to_collision_within_bound = 0.0 if self.ttc_violated else 1.0

        # Lane Keeping: 0 if violated
        result.lane_keeping = 0.0 if self.lk_violated else 1.0

        # History Comfort: 0 if violated at any frame
        result.history_comfort = 0.0 if self.hc_violated else 1.0

        # Extended Comfort: 0 if violated at any frame
        result.extended_comfort = 0.0 if self.ec_violated else 1.0

        # === Compute Final Score ===
        # Multiplicative metrics (gate)
        multi_prod = (
            result.no_at_fault_collisions *
            result.drivable_area_compliance *
            result.driving_direction_compliance *
            result.traffic_light_compliance
        )

        # Weighted metrics
        weights = self.config.weighted_metrics_array
        weighted_values = np.array([
            result.ego_progress,
            result.time_to_collision_within_bound,
            result.lane_keeping,
            result.history_comfort,
            result.extended_comfort,
        ])

        weighted_avg = np.sum(weights * weighted_values) / np.sum(weights)

        # Final score
        result.score = multi_prod * weighted_avg
        result.num_collision_frames = len(self.collision_frames)
        result.valid = True

        return result

    def get_results_dict(self) -> Dict:
        """
        Get results as a dictionary for CSV export.

        Returns:
            Dictionary with all metric values
        """
        final = self.finalize()

        return {
            'no_at_fault_collisions': final.no_at_fault_collisions,
            'drivable_area_compliance': final.drivable_area_compliance,
            'driving_direction_compliance': final.driving_direction_compliance,
            'traffic_light_compliance': final.traffic_light_compliance,
            'ego_progress': final.ego_progress,
            'time_to_collision_within_bound': final.time_to_collision_within_bound,
            'lane_keeping': final.lane_keeping,
            'history_comfort': final.history_comfort,
            'extended_comfort': final.extended_comfort,
            'score': final.score,
            'num_frames': final.num_frames,
            'num_collision_frames': final.num_collision_frames,
            'valid': final.valid,
        }

    def reset(self) -> None:
        """Reset scorer state for a new episode."""
        self.ego_state_history.clear()
        self.plan_trajectory_history.clear()
        self.prev_plan_states = None

        self.start_progress = None
        self.current_progress = 0.0

        self.collision_occurred = False
        self.collision_frames = []
        self.collided_track_ids = []

        self.dac_violated = False
        self.ddc_violation_progress = 0.0
        self.tlc_violated = False
        self.ttc_violated = False

        self.lk_consecutive_violation = 0
        self.lk_violated = False

        self.hc_violated = False
        self.ec_violated = False

        self.frame_metrics = []
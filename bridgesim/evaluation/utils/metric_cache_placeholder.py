"""
Metric Cache Placeholder for future pre-computation infrastructure.

This module provides dataclasses and placeholders for caching metric-related
data that can be pre-computed before evaluation.

TODO: Implement full caching infrastructure in future iterations:
- Pre-compute GT trajectory arc lengths
- Cache lane polygons and drivable area maps
- Pre-compute agent trajectories and polygons
- Store traffic light states in efficient format

Reference: navsim/planning/metric_caching/metric_cache.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class GTTrajectoryCache:
    """
    Pre-computed ground truth trajectory data.

    This caches data derived from the GT trajectory that would otherwise
    be computed repeatedly during scoring.
    """
    # GT trajectory positions: (N, 2) [x, y]
    positions: np.ndarray

    # GT trajectory headings: (N,)
    headings: np.ndarray

    # Cumulative arc length along GT trajectory: (N,)
    # arc_lengths[i] = total distance from start to waypoint i
    arc_lengths: np.ndarray

    # Total trajectory length in meters
    total_length: float

    # Frame indices corresponding to each waypoint
    frame_indices: np.ndarray

    @classmethod
    def from_scenario_data(cls, scenario_data: dict, sdc_id: str) -> 'GTTrajectoryCache':
        """
        Build cache from scenario data.

        Args:
            scenario_data: Loaded scenario pickle data
            sdc_id: ID of the self-driving car (ego)

        Returns:
            GTTrajectoryCache instance
        """
        ego_track = scenario_data['tracks'][sdc_id]
        positions = ego_track['state']['position'][:, :2]  # (N, 2)
        headings = ego_track['state']['heading']  # (N,)

        # Compute arc lengths
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.zeros(len(positions))
        arc_lengths[1:] = np.cumsum(segment_lengths)

        return cls(
            positions=positions.astype(np.float64),
            headings=headings.astype(np.float64),
            arc_lengths=arc_lengths.astype(np.float64),
            total_length=float(arc_lengths[-1]),
            frame_indices=np.arange(len(positions)),
        )

    def get_progress_at_position(self, position: np.ndarray) -> float:
        """
        Get progress along GT trajectory for a given position.

        Args:
            position: (2,) [x, y] position

        Returns:
            Progress in meters (arc length to closest point on GT)
        """
        # Find closest point on GT trajectory
        distances = np.linalg.norm(self.positions - position[:2], axis=1)
        closest_idx = np.argmin(distances)
        return float(self.arc_lengths[closest_idx])

    def get_lateral_deviation(self, position: np.ndarray, frame_idx: int) -> float:
        """
        Get lateral deviation from GT trajectory.

        Args:
            position: (2,) [x, y] position
            frame_idx: Current frame index (for reference point)

        Returns:
            Lateral deviation in meters
        """
        # Clamp frame_idx to valid range
        frame_idx = min(frame_idx, len(self.positions) - 1)
        gt_position = self.positions[frame_idx]
        return float(np.linalg.norm(position[:2] - gt_position))


@dataclass
class MapCache:
    """
    Pre-computed map data cache.

    TODO: Implement in future iteration
    """
    # Placeholder for lane polygons
    # lane_polygons: List[Polygon]

    # Placeholder for drivable area polygons
    # drivable_area_polygons: List[Polygon]

    # Placeholder for route lane IDs
    route_lane_ids: List[str] = field(default_factory=list)

    # Placeholder flag
    is_initialized: bool = False


@dataclass
class AgentCache:
    """
    Pre-computed agent trajectory cache.

    TODO: Implement in future iteration
    """
    # Agent ID -> trajectory positions at each frame
    # trajectories: Dict[str, np.ndarray]  # {agent_id: (N_frames, 2)}

    # Agent ID -> object type
    agent_types: Dict[str, str] = field(default_factory=dict)

    # Placeholder flag
    is_initialized: bool = False


@dataclass
class TrafficLightCache:
    """
    Pre-computed traffic light state cache.

    TODO: Implement in future iteration
    """
    # Lane ID -> list of states per frame
    # states: Dict[str, List[str]]  # {lane_id: ['GREEN', 'RED', ...]}

    # Placeholder flag
    is_initialized: bool = False


@dataclass
class MetricCache:
    """
    Complete metric cache containing all pre-computed data.

    This is the main interface for accessing cached metric data.

    Reference structure inspired by:
    navsim/planning/metric_caching/metric_cache.py
    """
    # Scenario metadata
    scenario_name: str
    scenario_length: int  # Number of frames

    # Pre-computed caches
    gt_trajectory: Optional[GTTrajectoryCache] = None
    map_cache: Optional[MapCache] = None
    agent_cache: Optional[AgentCache] = None
    traffic_light_cache: Optional[TrafficLightCache] = None

    # Cache file path (if loaded from disk)
    file_path: Optional[Path] = None

    @classmethod
    def from_scenario_data(
        cls,
        scenario_data: dict,
        scenario_name: str,
    ) -> 'MetricCache':
        """
        Build metric cache from scenario data.

        Currently only implements GT trajectory cache.
        Other caches are placeholders for future implementation.

        Args:
            scenario_data: Loaded scenario pickle data
            scenario_name: Name of the scenario

        Returns:
            MetricCache instance
        """
        sdc_id = scenario_data['metadata']['sdc_id']
        scenario_length = scenario_data['length']

        # Build GT trajectory cache
        gt_trajectory = GTTrajectoryCache.from_scenario_data(scenario_data, sdc_id)

        # Placeholder caches (not implemented yet)
        map_cache = MapCache(is_initialized=False)
        agent_cache = AgentCache(is_initialized=False)
        traffic_light_cache = TrafficLightCache(is_initialized=False)

        return cls(
            scenario_name=scenario_name,
            scenario_length=scenario_length,
            gt_trajectory=gt_trajectory,
            map_cache=map_cache,
            agent_cache=agent_cache,
            traffic_light_cache=traffic_light_cache,
        )

    def dump(self, file_path: Path) -> None:
        """
        Save metric cache to disk.

        TODO: Implement serialization
        """
        # import pickle
        # with open(file_path, 'wb') as f:
        #     pickle.dump(self, f)
        raise NotImplementedError("MetricCache serialization not yet implemented")

    @classmethod
    def load(cls, file_path: Path) -> 'MetricCache':
        """
        Load metric cache from disk.

        TODO: Implement deserialization
        """
        # import pickle
        # with open(file_path, 'rb') as f:
        #     return pickle.load(f)
        raise NotImplementedError("MetricCache deserialization not yet implemented")


# === Helper Functions ===

def compute_arc_length(positions: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc length along a trajectory.

    Args:
        positions: (N, 2) array of [x, y] positions

    Returns:
        (N,) array of cumulative arc lengths
    """
    if len(positions) < 2:
        return np.zeros(len(positions))

    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.zeros(len(positions))
    arc_lengths[1:] = np.cumsum(segment_lengths)

    return arc_lengths


def project_point_to_trajectory(
    point: np.ndarray,
    trajectory_positions: np.ndarray,
    trajectory_arc_lengths: np.ndarray,
) -> tuple:
    """
    Project a point onto a trajectory and get progress + lateral offset.

    Args:
        point: (2,) [x, y] point to project
        trajectory_positions: (N, 2) trajectory positions
        trajectory_arc_lengths: (N,) cumulative arc lengths

    Returns:
        Tuple of (progress, lateral_offset)
    """
    # Find closest point on trajectory
    distances = np.linalg.norm(trajectory_positions - point[:2], axis=1)
    closest_idx = np.argmin(distances)

    progress = trajectory_arc_lengths[closest_idx]
    lateral_offset = distances[closest_idx]

    return progress, lateral_offset

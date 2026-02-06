"""
Detailed collision classification matching NavSim's implementation.

References:
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:363-420
"""

import numpy as np
from enum import IntEnum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity


class CollisionType(IntEnum):
    """
    Collision type classification matching nuPlan's CollisionType.

    Reference: nuplan/planning/metrics/utils/collision_utils.py
    """
    NO_COLLISION = 0
    STOPPED_EGO_COLLISION = 1      # Ego stopped when hit
    STOPPED_TRACK_COLLISION = 2    # Other agent stopped when hit (at-fault)
    ACTIVE_FRONT_COLLISION = 3     # Ego front bumper collision (at-fault)
    ACTIVE_REAR_COLLISION = 4      # Ego rear collision (not at-fault)
    ACTIVE_LATERAL_COLLISION = 5   # Side collision (at-fault if ego in wrong area)


# Agent types that count as full collision (vs static objects)
AGENT_TYPES = ['VEHICLE', 'CYCLIST', 'PEDESTRIAN']

# Stopped speed threshold
STOPPED_SPEED_THRESHOLD = 5e-2  # m/s


# === Vehicle Dimensions ===
VEHICLE_LENGTH = 4.515  # meters
VEHICLE_WIDTH = 1.852   # meters


@dataclass
class TrackedObject:
    """
    Representation of a tracked object (vehicle, cyclist, pedestrian).
    """
    track_id: str
    object_type: str  # 'VEHICLE', 'CYCLIST', 'PEDESTRIAN', etc.
    position: np.ndarray  # (2,) [x, y]
    heading: float  # radians
    velocity: np.ndarray  # (2,) [vx, vy]
    length: float
    width: float

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)

    @property
    def is_stopped(self) -> bool:
        return self.speed <= STOPPED_SPEED_THRESHOLD

    def get_polygon(self) -> Polygon:
        """Get shapely polygon for this object."""
        return create_vehicle_polygon(
            self.position[0], self.position[1],
            self.heading, self.length, self.width
        )


def create_vehicle_polygon(
    x: float,
    y: float,
    heading: float,
    length: float = VEHICLE_LENGTH,
    width: float = VEHICLE_WIDTH,
) -> Polygon:
    """
    Create a vehicle polygon at the given position and heading.

    Args:
        x: X position (center)
        y: Y position (center)
        heading: Heading in radians
        length: Vehicle length
        width: Vehicle width

    Returns:
        Shapely Polygon representing the vehicle footprint
    """
    # Create polygon centered at origin
    half_length = length / 2
    half_width = width / 2

    corners = np.array([
        [half_length, half_width],    # Front-left
        [half_length, -half_width],   # Front-right
        [-half_length, -half_width],  # Rear-right
        [-half_length, half_width],   # Rear-left
    ])

    base_poly = Polygon(corners)

    # Rotate and translate
    rotated_poly = affinity.rotate(base_poly, np.degrees(heading), origin=(0, 0))
    translated_poly = affinity.translate(rotated_poly, xoff=x, yoff=y)

    return translated_poly


def is_agent_behind(
    ego_position: np.ndarray,
    ego_heading: float,
    agent_position: np.ndarray,
) -> bool:
    """
    Check if agent is behind ego vehicle.

    Reference: nuplan/planning/simulation/observation/idm/utils.py

    Args:
        ego_position: (2,) ego position [x, y]
        ego_heading: Ego heading in radians
        agent_position: (2,) agent position [x, y]

    Returns:
        True if agent is behind ego
    """
    # Vector from ego to agent
    delta = agent_position[:2] - ego_position[:2]

    # Ego forward direction
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    # Dot product: positive if agent is in front, negative if behind
    dot = np.dot(delta, ego_forward)

    return dot < 0


def is_agent_ahead(
    ego_position: np.ndarray,
    ego_heading: float,
    agent_position: np.ndarray,
) -> bool:
    """
    Check if agent is ahead of ego vehicle.

    Args:
        ego_position: (2,) ego position [x, y]
        ego_heading: Ego heading in radians
        agent_position: (2,) agent position [x, y]

    Returns:
        True if agent is ahead of ego
    """
    return not is_agent_behind(ego_position, ego_heading, agent_position)


def get_collision_type(
    ego_position: np.ndarray,
    ego_heading: float,
    ego_velocity: np.ndarray,
    ego_polygon: Polygon,
    tracked_object: TrackedObject,
    stopped_speed_threshold: float = STOPPED_SPEED_THRESHOLD,
) -> CollisionType:
    """
    Classify collision between ego and tracked object.

    Reference: NavSim pdm_scorer_utils.py:12-64

    Args:
        ego_position: (2,) ego position [x, y]
        ego_heading: Ego heading in radians
        ego_velocity: (2,) ego velocity [vx, vy]
        ego_polygon: Shapely Polygon for ego vehicle
        tracked_object: TrackedObject instance
        stopped_speed_threshold: Speed threshold for "stopped"

    Returns:
        CollisionType enum value
    """
    ego_speed = np.linalg.norm(ego_velocity)
    is_ego_stopped = ego_speed <= stopped_speed_threshold

    tracked_object_polygon = tracked_object.get_polygon()
    agent_position = tracked_object.position[:2]

    # Case 1: Ego is stopped
    if is_ego_stopped:
        return CollisionType.STOPPED_EGO_COLLISION

    # Case 2: Tracked object is stopped (at-fault for ego)
    if tracked_object.is_stopped:
        return CollisionType.STOPPED_TRACK_COLLISION

    # Case 3: Rear collision (agent behind ego)
    if is_agent_behind(ego_position, ego_heading, agent_position):
        return CollisionType.ACTIVE_REAR_COLLISION

    # Case 4: Front bumper collision
    # Check if front edge of ego intersects with agent
    coords = list(ego_polygon.exterior.coords)
    # Front edge is between front-left (index 0) and front-right (index 1) for our polygon
    # But NavSim uses indices 0 and 3, which depends on polygon construction
    # We'll use the front edge based on our polygon construction
    front_edge = LineString([coords[0], coords[1]])
    if front_edge.intersects(tracked_object_polygon):
        return CollisionType.ACTIVE_FRONT_COLLISION

    # Case 5: Lateral collision (default)
    return CollisionType.ACTIVE_LATERAL_COLLISION


def is_at_fault_collision(
    collision_type: CollisionType,
    ego_in_wrong_area: bool = False,
) -> bool:
    """
    Determine if ego is at fault for the collision.

    Reference: NavSim pdm_scorer.py:400-418

    At-fault conditions:
    1. STOPPED_TRACK_COLLISION: Ego hit a stopped vehicle
    2. ACTIVE_FRONT_COLLISION: Ego hit something with front bumper
    3. ACTIVE_LATERAL_COLLISION + ego in wrong area (multiple lanes or non-drivable)

    Args:
        collision_type: Type of collision
        ego_in_wrong_area: Whether ego is in multiple lanes or non-drivable area

    Returns:
        True if ego is at fault
    """
    # Always at-fault
    if collision_type in [
        CollisionType.STOPPED_TRACK_COLLISION,
        CollisionType.ACTIVE_FRONT_COLLISION,
    ]:
        return True

    # Lateral collision is at-fault only if ego is in wrong area
    if collision_type == CollisionType.ACTIVE_LATERAL_COLLISION and ego_in_wrong_area:
        return True

    # Not at-fault: STOPPED_EGO_COLLISION, ACTIVE_REAR_COLLISION,
    # or ACTIVE_LATERAL_COLLISION when ego is in correct area
    return False


def get_collision_score(
    collision_type: CollisionType,
    object_type: str,
    ego_in_wrong_area: bool = False,
) -> float:
    """
    Get collision score based on collision type and object type.

    Reference: NavSim pdm_scorer.py:410-414

    Args:
        collision_type: Type of collision
        object_type: Type of object hit ('VEHICLE', 'CYCLIST', etc.)
        ego_in_wrong_area: Whether ego is in wrong area

    Returns:
        Score: 0.0 for at-fault with agent, 0.5 for at-fault with static, 1.0 for not at-fault
    """
    if not is_at_fault_collision(collision_type, ego_in_wrong_area):
        return 1.0

    # At-fault collision
    if object_type in AGENT_TYPES:
        return 0.0  # Full penalty for hitting agents
    else:
        return 0.5  # Partial penalty for hitting static objects


@dataclass
class CollisionResult:
    """
    Result of collision check for a single frame.
    """
    has_collision: bool
    collision_type: CollisionType
    is_at_fault: bool
    collision_score: float
    collided_object_id: Optional[str]
    collided_object_type: Optional[str]
    details: Dict


def check_collision_with_agents(
    ego_position: np.ndarray,
    ego_heading: float,
    ego_velocity: np.ndarray,
    ego_polygon: Polygon,
    agents: List[TrackedObject],
    already_collided_ids: List[str],
    ego_in_wrong_area: bool = False,
) -> CollisionResult:
    """
    Check for collisions between ego and all agents.

    Reference: NavSim pdm_scorer.py:363-420

    Args:
        ego_position: (2,) ego position
        ego_heading: Ego heading in radians
        ego_velocity: (2,) ego velocity
        ego_polygon: Shapely Polygon for ego
        agents: List of TrackedObject instances
        already_collided_ids: IDs of agents already collided with (to ignore)
        ego_in_wrong_area: Whether ego is in wrong driving area

    Returns:
        CollisionResult with details of any collision found
    """
    for agent in agents:
        # Skip already-collided agents
        if agent.track_id in already_collided_ids:
            continue

        agent_polygon = agent.get_polygon()

        # Check intersection
        if not ego_polygon.intersects(agent_polygon):
            continue

        # Collision detected - classify it
        collision_type = get_collision_type(
            ego_position, ego_heading, ego_velocity,
            ego_polygon, agent
        )

        is_at_fault = is_at_fault_collision(collision_type, ego_in_wrong_area)
        score = get_collision_score(collision_type, agent.object_type, ego_in_wrong_area)

        return CollisionResult(
            has_collision=True,
            collision_type=collision_type,
            is_at_fault=is_at_fault,
            collision_score=score,
            collided_object_id=agent.track_id,
            collided_object_type=agent.object_type,
            details={
                'ego_speed': np.linalg.norm(ego_velocity),
                'agent_speed': agent.speed,
                'ego_in_wrong_area': ego_in_wrong_area,
            }
        )

    # No collision
    return CollisionResult(
        has_collision=False,
        collision_type=CollisionType.NO_COLLISION,
        is_at_fault=False,
        collision_score=1.0,
        collided_object_id=None,
        collided_object_type=None,
        details={}
    )
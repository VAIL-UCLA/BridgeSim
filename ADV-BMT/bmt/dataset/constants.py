"""
Define a lot of constants. It should be totally removed as most of them should be defined by MetaDrive / ScenarioNet.
"""
from metadrive.scenario.scenario_description import MetaDriveType

# NUM_TYPES = 3
NUM_TYPES = 5

MAP_FEATURE_STATE_DIM = 27
TRAFFIC_LIGHT_STATE_DIM = 7

AGENT_STATE_DIM = 16

# ACTOR_PREDICT_DIM = 6 + 2 + 4 + 5  # 3 for position, 1 for heading, 2 for velocity, 5 for types
TRAFFIC_LIGHT_PREDICT_DIM = 9  # 9 original possible state

# TODO(pzh): Do we have to do the normalization? Shouldn't the layer norm solve this?
# POSITION_XY_RANGE = 100.
# LOCAL_POSITION_XY_RANGE = 5.
# HEADING_RANGE = np.pi
# VELOCITY_XY_RANGE = 10.
# SIZE_RANGE = 5.
# MAP_VECTOR_XY_RANGE = 50.

# TODO(pzh): Consider remove this.
object_type_to_int = {
    MetaDriveType.UNSET: 0,
    MetaDriveType.VEHICLE: 1,
    MetaDriveType.PEDESTRIAN: 2,
    MetaDriveType.CYCLIST: 3,
    MetaDriveType.OTHER: 4
}

# TODO(pzh): Consider remove this.
object_int_to_type = {
    -1: MetaDriveType.UNSET,
    0: MetaDriveType.UNSET,
    1: MetaDriveType.VEHICLE,
    2: MetaDriveType.PEDESTRIAN,
    3: MetaDriveType.CYCLIST,
    4: MetaDriveType.OTHER
}

HEADING_PLACEHOLDER = -100  # For the object that has no heading, set this.

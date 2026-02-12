"""
Command constants for evaluation pipeline.

RoadOption constants (1-indexed, used in route generation and evaluator):
    LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5, CHANGELANERIGHT=6

NAVSIM model command mapping (0-indexed, one-hot encoding):
    Left             [1, 0, 0, 0]
    Straight/Forward [0, 1, 0, 0]
    Right            [0, 0, 1, 0]
    Unknown          [0, 0, 0, 1]
"""

import numpy as np


# RoadOption Enum (1-indexed as defined in route generation)
VOID = -1
LEFT = 1
RIGHT = 2
STRAIGHT = 3
LANEFOLLOW = 4
CHANGELANELEFT = 5
CHANGELANERIGHT = 6

# NAVSIM model command mapping: 0-indexed command -> one-hot encoding
# Keys are (constant - 1) to convert from 1-indexed route options to 0-indexed model input
NAVSIM_CMD_MAPPING = {
    LEFT - 1:       np.array([1, 0, 0, 0], dtype=np.float32),  # Left
    STRAIGHT - 1:   np.array([0, 1, 0, 0], dtype=np.float32),  # Straight/Forward
    RIGHT - 1:      np.array([0, 0, 1, 0], dtype=np.float32),  # Right
    LANEFOLLOW - 1: np.array([0, 0, 0, 1], dtype=np.float32),  # Unknown
}
DEFAULT_CMD = np.array([0, 1, 0, 0], dtype=np.float32)  # Forward/Straight

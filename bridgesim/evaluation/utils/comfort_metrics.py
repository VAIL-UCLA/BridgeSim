"""
NavSim-identical comfort metrics calculation.

This module implements comfort metrics matching NavSim's implementation.

References:
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
- navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:656-700
- navsim/planning/simulation/planner/pdm_planner/scoring/scene_aggregator.py:49-77
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Dict
from scipy.signal import savgol_filter
from dataclasses import dataclass


# === Comfort Thresholds (from NavSim pdm_comfort_metrics.py:15-38) ===
# (1) ego_jerk_metric
MAX_ABS_MAG_JERK: float = 8.37  # [m/s^3]

# (2) ego_lat_acceleration_metric
MAX_ABS_LAT_ACCEL: float = 4.89  # [m/s^2]

# (3) ego_lon_acceleration_metric
MAX_LON_ACCEL: float = 2.40  # [m/s^2]
MIN_LON_ACCEL: float = -4.05

# (4) ego_yaw_acceleration_metric
MAX_ABS_YAW_ACCEL: float = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
MAX_ABS_LON_JERK: float = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
MAX_ABS_YAW_RATE: float = 0.95  # [rad/s]

# Extended Comfort thresholds (from NavSim pdm_comfort_metrics.py:36-39)
EC_ACCELERATION_THRESHOLD: float = 0.7  # [m/s^2]
EC_JERK_THRESHOLD: float = 0.5  # [m/s^3]
EC_YAW_RATE_THRESHOLD: float = 0.1  # [rad/s]
EC_YAW_ACCEL_THRESHOLD: float = 0.1  # [rad/s^2]

# Stopped speed threshold
STOPPED_SPEED_THRESHOLD: float = 5e-2  # [m/s]


@dataclass
class StateIndex:
    """
    Index mapping for state array representation.
    Matches NavSim's StateIndex enum for compatibility.
    """
    X: int = 0
    Y: int = 1
    HEADING: int = 2
    VELOCITY_X: int = 3
    VELOCITY_Y: int = 4
    ACCELERATION_X: int = 5
    ACCELERATION_Y: int = 6

    @classmethod
    def size(cls) -> int:
        return 7


def ego_state_to_state_array(
    position: np.ndarray,
    heading: float,
    velocity: np.ndarray,
    acceleration: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert ego state to state array representation.

    Args:
        position: (2,) or (3,) array [x, y, (z)]
        heading: Heading in radians
        velocity: (2,) or (3,) array [vx, vy, (vz)]
        acceleration: Optional (2,) or (3,) array [ax, ay, (az)]

    Returns:
        (7,) state array [x, y, heading, vx, vy, ax, ay]
    """
    state = np.zeros(StateIndex.size(), dtype=np.float64)
    state[StateIndex.X] = position[0]
    state[StateIndex.Y] = position[1]
    state[StateIndex.HEADING] = heading
    state[StateIndex.VELOCITY_X] = velocity[0]
    state[StateIndex.VELOCITY_Y] = velocity[1]

    if acceleration is not None:
        state[StateIndex.ACCELERATION_X] = acceleration[0]
        state[StateIndex.ACCELERATION_Y] = acceleration[1]

    return state


def _phase_unwrap(headings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Phase unwrap headings to avoid discontinuities.

    Reference: NavSim pdm_comfort_metrics.py:157-175
    """
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float64]:
    """
    Approximate derivatives using Savitzky-Golay filter.

    Reference: NavSim pdm_comfort_metrics.py:178-219
    """
    window_length = min(window_length, len(x))

    if window_length < 3:
        # Not enough points for savgol_filter, use simple diff
        if len(y.shape) == 1:
            derivative = np.zeros_like(y)
            if len(y) > 1:
                dx = np.mean(np.diff(x))
                derivative[:-1] = np.diff(y) / dx
                derivative[-1] = derivative[-2]
            return derivative
        else:
            # Batch case
            derivative = np.zeros_like(y)
            if y.shape[-1] > 1:
                dx = np.mean(np.diff(x))
                derivative[..., :-1] = np.diff(y, axis=-1) / dx
                derivative[..., -1] = derivative[..., -2]
            return derivative

    if not (poly_order < window_length):
        poly_order = window_length - 1

    dx = np.diff(x, axis=-1)
    dx = dx.mean()

    derivative: npt.NDArray[np.float64] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float64]:
    """
    Extract acceleration from state array.

    Reference: NavSim pdm_comfort_metrics.py:42-86
    """
    n_batch, n_time, n_states = states.shape

    if acceleration_coordinate == "x":
        acceleration = states[..., StateIndex.ACCELERATION_X]
    elif acceleration_coordinate == "y":
        acceleration = states[..., StateIndex.ACCELERATION_Y]
    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(f"Unknown acceleration_coordinate: {acceleration_coordinate}")

    # Apply smoothing
    window_length = min(window_length, n_time)
    if window_length >= 3:
        acceleration = savgol_filter(
            acceleration,
            polyorder=poly_order,
            window_length=window_length,
            axis=-1,
        )

    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    """
    Extract jerk (derivative of acceleration) from state array.

    Reference: NavSim pdm_comfort_metrics.py:89-125
    """
    n_batch, n_time, n_states = states.shape

    ego_acceleration = _extract_ego_acceleration(
        states,
        acceleration_coordinate=acceleration_coordinate,
    )

    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float64]:
    """
    Extract yaw rate from state array.

    Reference: NavSim pdm_comfort_metrics.py:128-154
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=window_length,
    )
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Check if metric values are within bounds.

    Reference: NavSim pdm_comfort_metrics.py:222-238
    """
    min_bound = min_bound if min_bound is not None else float(-np.inf)
    max_bound = max_bound if max_bound is not None else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_lon_acceleration(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check longitudinal acceleration within bounds."""
    lon_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="x")
    return _within_bound(lon_acceleration, min_bound=MIN_LON_ACCEL, max_bound=MAX_LON_ACCEL)


def _compute_lat_acceleration(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check lateral acceleration within bounds."""
    lat_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="y")
    return _within_bound(lat_acceleration, min_bound=-MAX_ABS_LAT_ACCEL, max_bound=MAX_ABS_LAT_ACCEL)


def _compute_jerk_metric(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check jerk magnitude within bounds."""
    jerk_metric = _extract_ego_jerk(states, acceleration_coordinate="magnitude", time_steps_s=time_steps_s)
    return _within_bound(jerk_metric, min_bound=-MAX_ABS_MAG_JERK, max_bound=MAX_ABS_MAG_JERK)


def _compute_lon_jerk_metric(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check longitudinal jerk within bounds."""
    lon_jerk_metric = _extract_ego_jerk(states, acceleration_coordinate="x", time_steps_s=time_steps_s)
    return _within_bound(lon_jerk_metric, min_bound=-MAX_ABS_LON_JERK, max_bound=MAX_ABS_LON_JERK)


def _compute_yaw_accel(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check yaw acceleration within bounds."""
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3)
    return _within_bound(yaw_accel_metric, min_bound=-MAX_ABS_YAW_ACCEL, max_bound=MAX_ABS_YAW_ACCEL)


def _compute_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Check yaw rate within bounds."""
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s)
    return _within_bound(yaw_rate_metric, min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE)


def ego_is_comfortable(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Check if ego trajectory is comfortable across all metrics.

    Reference: NavSim pdm_comfort_metrics.py:351-379

    Args:
        states: (n_batch, n_time, n_states) state array
        time_point_s: (n_time,) time steps in seconds

    Returns:
        (n_batch, n_metrics) boolean array indicating comfort for each metric
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lat_acceleration,
        _compute_jerk_metric,
        _compute_lon_jerk_metric,
        _compute_yaw_accel,
        _compute_yaw_rate,
    ]

    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, time_point_s)

    return results


def extract_features(
    states: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Extract features for Extended Comfort evaluation.

    Reference: NavSim pdm_comfort_metrics.py:410-427
    """
    return {
        "acceleration": _extract_ego_acceleration(states, "magnitude"),
        "jerk": _extract_ego_jerk(states, "magnitude", time_point_s),
        "yaw_rate": _extract_ego_yaw_rate(states, time_point_s),
        "yaw_accel": _extract_ego_yaw_rate(states, time_point_s, deriv_order=2),
    }


def calculate_rms(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute RMS of values along time axis.

    Reference: NavSim pdm_comfort_metrics.py:397-407
    """
    squared_values = values ** 2
    mean_squared = np.mean(squared_values, axis=-1)
    rms_values = np.sqrt(mean_squared)
    return rms_values


def ego_is_two_frame_extended_comfort(
    states_1: npt.NDArray[np.float64],
    states_2: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Evaluate Extended Comfort between two consecutive trajectories.

    Reference: NavSim pdm_comfort_metrics.py:430-467

    Args:
        states_1: First trajectory (n_batch, n_time, n_features)
        states_2: Second trajectory (n_batch, n_time, n_features)
        time_point_s: Time steps in seconds

    Returns:
        Boolean array indicating if difference meets Extended Comfort criteria
    """
    assert states_1.shape == states_2.shape, "Both trajectories must have the same shape"

    # Extract features for both trajectories
    features_1 = extract_features(states_1, time_point_s)
    features_2 = extract_features(states_2, time_point_s)

    # Compute differences between corresponding time steps
    diff_acceleration = features_1["acceleration"] - features_2["acceleration"]
    diff_jerk = features_1["jerk"] - features_2["jerk"]
    diff_yaw_rate = features_1["yaw_rate"] - features_2["yaw_rate"]
    diff_yaw_accel = features_1["yaw_accel"] - features_2["yaw_accel"]

    # Calculate RMS differences
    rms_acceleration = calculate_rms(diff_acceleration)
    rms_jerk = calculate_rms(diff_jerk)
    rms_yaw_rate = calculate_rms(diff_yaw_rate)
    rms_yaw_accel = calculate_rms(diff_yaw_accel)

    # Compare against thresholds
    meets_acceleration = rms_acceleration <= EC_ACCELERATION_THRESHOLD
    meets_jerk = rms_jerk <= EC_JERK_THRESHOLD
    meets_yaw_rate = rms_yaw_rate <= EC_YAW_RATE_THRESHOLD
    meets_yaw_accel = rms_yaw_accel <= EC_YAW_ACCEL_THRESHOLD

    return np.logical_and.reduce([meets_acceleration, meets_jerk, meets_yaw_rate, meets_yaw_accel])


def compute_history_comfort(
    current_states: npt.NDArray[np.float64],
    history_states: Optional[npt.NDArray[np.float64]],
    dt: float,
) -> Tuple[bool, Dict]:
    """
    Compute history comfort by padding current trajectory with past history.

    This matches NavSim's approach in pdm_scorer.py:656-700:
    1. Concatenate history + current trajectory
    2. Compute comfort over the full padded trajectory
    3. Return True if all comfort metrics pass

    Args:
        current_states: (n_time, n_states) current trajectory states
        history_states: (m_time, n_states) past trajectory states (or None)
        dt: Time step in seconds

    Returns:
        Tuple of (is_comfortable, details_dict)
    """
    # Ensure 3D shape for batch processing
    if current_states.ndim == 2:
        current_states = current_states[None, ...]  # Add batch dim

    if history_states is not None and len(history_states) > 0:
        if history_states.ndim == 2:
            history_states = history_states[None, ...]

        # Concatenate history with current (NavSim pdm_scorer.py:684-692)
        padded_states = np.concatenate([history_states, current_states], axis=1)
    else:
        padded_states = current_states

    n_batch, n_time, n_states = padded_states.shape

    # Create time points (NavSim pdm_scorer.py:695-696)
    time_point_s = np.arange(0, n_time).astype(np.float64) * dt

    # Compute comfort (NavSim pdm_scorer.py:698)
    comfort_results = ego_is_comfortable(padded_states, time_point_s)
    is_comfortable = comfort_results.all()

    details = {
        'history_length': history_states.shape[1] if history_states is not None else 0,
        'current_length': current_states.shape[1],
        'total_length': n_time,
        'comfort_per_metric': comfort_results[0].tolist() if n_batch > 0 else [],
    }

    return bool(is_comfortable), details


def compute_extended_comfort(
    current_states: npt.NDArray[np.float64],
    previous_states: Optional[npt.NDArray[np.float64]],
    dt: float,
    overlap_offset: int = 1,
) -> Tuple[bool, Dict]:
    """
    Compute extended comfort between current and previous plan.

    Reference: NavSim scene_aggregator.py:49-77

    This measures temporal consistency - how much the plan changes
    between consecutive planning steps.

    Args:
        current_states: (n_time, n_states) current plan states
        previous_states: (n_time, n_states) previous plan states (or None)
        dt: Time step in seconds
        overlap_offset: Number of time steps offset for overlap alignment

    Returns:
        Tuple of (is_consistent, details_dict)
    """
    if previous_states is None:
        return True, {'no_previous_plan': True}

    # Ensure 3D shape
    if current_states.ndim == 2:
        current_states = current_states[None, ...]
    if previous_states.ndim == 2:
        previous_states = previous_states[None, ...]

    # Align overlapping regions (NavSim scene_aggregator.py:64-66)
    # current_states[:-offset] overlaps with previous_states[offset:]
    if overlap_offset > 0 and overlap_offset < min(current_states.shape[1], previous_states.shape[1]):
        current_overlap = current_states[:, :-overlap_offset, :]
        previous_overlap = previous_states[:, overlap_offset:, :]

        # Ensure same length
        n_overlap = min(current_overlap.shape[1], previous_overlap.shape[1])
        current_overlap = current_overlap[:, :n_overlap, :]
        previous_overlap = previous_overlap[:, :n_overlap, :]
    else:
        # No offset, direct comparison
        n_overlap = min(current_states.shape[1], previous_states.shape[1])
        current_overlap = current_states[:, :n_overlap, :]
        previous_overlap = previous_states[:, :n_overlap, :]

    if n_overlap < 2:
        return True, {'insufficient_overlap': True, 'n_overlap': n_overlap}

    # Create time points
    time_point_s = np.arange(n_overlap) * dt

    # Compute two-frame extended comfort (NavSim scene_aggregator.py:71-75)
    is_consistent = ego_is_two_frame_extended_comfort(
        current_overlap,
        previous_overlap,
        time_point_s,
    )[0]

    details = {
        'n_overlap': n_overlap,
        'overlap_offset': overlap_offset,
        'is_consistent': bool(is_consistent),
    }

    return bool(is_consistent), details
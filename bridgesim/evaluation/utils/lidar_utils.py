"""Runtime lidar helpers for BridgeSim evaluation."""

from typing import Any, Dict, Optional

import numpy as np
import torch


def ray_lidar_to_ego_points(
    cloud_points,
    num_lasers: int,
    distance: float,
    height: float = 1.2,
    max_fraction: float = 0.999,
) -> np.ndarray:
    """Convert MetaDrive 2D ray-lidar hit fractions into ego-frame xyz points."""
    ranges = np.asarray(cloud_points, dtype=np.float32).reshape(-1)
    if ranges.size != num_lasers:
        num_lasers = ranges.size
    if num_lasers == 0:
        return np.zeros((0, 3), dtype=np.float32)

    valid = np.isfinite(ranges) & (ranges >= 0.0) & (ranges < max_fraction)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    angles = np.arange(num_lasers, dtype=np.float32) * (2.0 * np.pi / float(num_lasers))
    hit_dist = ranges[valid] * float(distance)
    hit_angles = angles[valid]
    points = np.stack(
        [
            hit_dist * np.cos(hit_angles),
            hit_dist * np.sin(hit_angles),
            np.full_like(hit_dist, float(height), dtype=np.float32),
        ],
        axis=1,
    )
    return points.astype(np.float32, copy=False)


def zero_lidar_bev(config: Any) -> torch.Tensor:
    """Create a zero lidar BEV tensor matching NavSim TransFuser feature layout."""
    return torch.zeros(
        int(config.lidar_seq_len),
        int(config.lidar_resolution_width),
        int(config.lidar_resolution_height),
        dtype=torch.float32,
    )


def lidar_points_to_bev(points: np.ndarray, config: Any) -> torch.Tensor:
    """Rasterize ego-frame xyz points into the TransFuser/DiffusionDrive lidar BEV format."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return zero_lidar_bev(config)
    points = points.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.size == 0:
        return zero_lidar_bev(config)

    max_height = float(getattr(config, "max_height_lidar", 100.0))
    split_height = float(getattr(config, "lidar_split_height", 0.2))
    use_ground_plane = bool(getattr(config, "use_ground_plane", False))
    hist_max = float(getattr(config, "hist_max_per_pixel", 5))

    points = points[points[:, 2] < max_height]
    if points.size == 0:
        return zero_lidar_bev(config)

    xbins = np.linspace(
        float(config.lidar_min_x),
        float(config.lidar_max_x),
        int(config.lidar_resolution_width) + 1,
    )
    ybins = np.linspace(
        float(config.lidar_min_y),
        float(config.lidar_max_y),
        int(config.lidar_resolution_height) + 1,
    )

    def splat(point_cloud: np.ndarray) -> np.ndarray:
        if point_cloud.size == 0:
            return np.zeros(
                (int(config.lidar_resolution_width), int(config.lidar_resolution_height)),
                dtype=np.float32,
            )
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0].astype(np.float32)
        hist[hist > hist_max] = hist_max
        return hist / hist_max

    above = points[points[:, 2] > split_height]
    above_features = splat(above)
    if use_ground_plane:
        below = points[points[:, 2] <= split_height]
        below_features = splat(below)
        features = np.stack([below_features, above_features], axis=0)
    else:
        features = np.expand_dims(above_features, axis=0)

    expected_channels = int(config.lidar_seq_len)
    if features.shape[0] < expected_channels:
        pad = np.zeros((expected_channels - features.shape[0], features.shape[1], features.shape[2]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    elif features.shape[0] > expected_channels:
        features = features[:expected_channels]

    return torch.from_numpy(features.astype(np.float32, copy=False))


def get_lidar_packet(
    lidar_data: Optional[Dict[str, Any]],
    sensor_name: Optional[str] = None,
    sensor_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Select one lidar packet from the unified runtime lidar container."""
    if not lidar_data:
        return None
    lidars = lidar_data.get("lidars", {})
    if not lidars:
        return None

    if sensor_name is not None:
        return lidars.get(sensor_name)

    if sensor_type is not None:
        for packet in lidars.values():
            if packet.get("sensor_type") == sensor_type:
                return packet
        return None

    default_lidar = lidar_data.get("default_lidar")
    if default_lidar in lidars:
        return lidars[default_lidar]

    return next(iter(lidars.values()), None)


def make_lidar_bev(
    lidar_data: Optional[Dict[str, Any]],
    config: Any,
    sensor_name: Optional[str] = None,
    sensor_type: Optional[str] = None,
) -> torch.Tensor:
    """Return rasterized lidar from the unified runtime lidar container."""
    packet = get_lidar_packet(lidar_data, sensor_name=sensor_name, sensor_type=sensor_type)
    if packet is None:
        return zero_lidar_bev(config)
    points = packet.get("points_ego")
    if points is None:
        return zero_lidar_bev(config)
    return lidar_points_to_bev(points, config)

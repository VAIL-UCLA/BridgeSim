"""
Camera-projection overlay for planned trajectories.

draw_trajectory_ribbon(img, plan_traj_ego, cam_cfg, ...) -> annotated img

ego frame convention (matches BridgeSim parse_output):
  col0 = lateral (left positive, metres)
  col1 = forward (metres)

vehicle frame: X = forward, Y = left, Z = up
camera body frame (OpenCV): X = right, Y = down, Z = optical axis
"""

import math
import cv2
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Camera geometry helpers
# ---------------------------------------------------------------------------

def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# Base rotation: vehicle frame → camera body frame when yaw=pitch=roll=0
# Camera body (Z=optical, X=right, Y=down) aligned with vehicle (X=fwd, Y=left, Z=up)
# Vehicle fwd → cam Z, vehicle left → -cam X, vehicle up → -cam Y
_R_VEH_TO_CAM_BASE = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=float)


def _build_extrinsics(cam_cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_veh_to_cam, cam_pos_veh) where cam_pos is in vehicle frame."""
    yaw   = math.radians(cam_cfg.get("yaw",   0.0))
    pitch = math.radians(cam_cfg.get("pitch", 0.0))
    roll  = math.radians(cam_cfg.get("roll",  0.0))

    # Camera body axes in vehicle frame = R_angle * R_cam_to_veh_base
    R_cam_to_veh = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll) @ _R_VEH_TO_CAM_BASE.T
    R_veh_to_cam = R_cam_to_veh.T

    cam_pos = np.array([cam_cfg["x"], cam_cfg["y"], cam_cfg["z"]], dtype=float)
    return R_veh_to_cam, cam_pos


def _build_intrinsics(cam_cfg: dict) -> Tuple[float, float, float, float]:
    W = float(cam_cfg["width"])
    H = float(cam_cfg["height"])
    fov_h_rad = math.radians(cam_cfg.get("fov_h", cam_cfg.get("fov", 63.71)))
    fx = (W / 2.0) / math.tan(fov_h_rad / 2.0)
    fov_v = cam_cfg.get("fov_v", None)
    fy = (H / 2.0) / math.tan(math.radians(fov_v) / 2.0) if fov_v else fx
    return fx, fy, W / 2.0, H / 2.0


def _project(pt_veh: np.ndarray,
             R: np.ndarray, cam_pos: np.ndarray,
             fx: float, fy: float, cx: float, cy: float,
             W: int, H: int,
             margin: int = 40) -> Optional[Tuple[int, int]]:
    """Project vehicle-frame 3-D point to pixel. Returns None if behind or off-screen."""
    p_cam = R @ (pt_veh - cam_pos)
    Z = p_cam[2]
    if Z < 0.1:
        return None
    u = fx * p_cam[0] / Z + cx
    v = fy * p_cam[1] / Z + cy
    if u < -margin or u > W + margin or v < -margin or v > H + margin:
        return None
    return int(round(u)), int(round(v))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _ego_to_veh(pt_ego: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Convert ego-frame (lateral, forward) point to vehicle-frame (fwd, left, z)."""
    return np.array([pt_ego[1], pt_ego[0], z], dtype=float)


def _draw_ribbon(img: np.ndarray,
                 pts_ego: np.ndarray,
                 R: np.ndarray, cam_pos: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float,
                 W: int, H: int,
                 color: Tuple[int, int, int],
                 path_width: float,
                 alpha: float) -> np.ndarray:
    if len(pts_ego) < 2:
        return img

    overlay = img.copy()

    # Compute left/right ribbon edges in ego frame
    tangents = np.diff(pts_ego, axis=0)
    tangents = np.vstack([tangents, tangents[-1:]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-6)
    # Perpendicular (left in 2-D ego frame)
    perp = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    hw = path_width / 2.0
    left_pts  = pts_ego + hw * perp
    right_pts = pts_ego - hw * perp

    for i in range(len(pts_ego) - 1):
        corners_ego = [left_pts[i], left_pts[i + 1], right_pts[i + 1], right_pts[i]]
        px = []
        for c in corners_ego:
            p = _project(_ego_to_veh(c, 0.0), R, cam_pos, fx, fy, cx, cy, W, H)
            if p is not None:
                px.append(p)
        if len(px) == 4:
            pts_arr = np.array(px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_arr], color)

    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)


def _draw_route_dots(img: np.ndarray,
                     route_ego: np.ndarray,
                     R: np.ndarray, cam_pos: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     W: int, H: int) -> np.ndarray:
    for pt in route_ego:
        p = _project(_ego_to_veh(pt, 0.0), R, cam_pos, fx, fy, cx, cy, W, H)
        if p is not None:
            cv2.circle(img, p, 5, (0, 0, 220), -1)
            cv2.circle(img, p, 5, (255, 255, 255), 1)
    return img


def _draw_target_pole(img: np.ndarray,
                      target_ego: np.ndarray,
                      R: np.ndarray, cam_pos: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      W: int, H: int) -> np.ndarray:
    base = _project(_ego_to_veh(target_ego, 0.0), R, cam_pos, fx, fy, cx, cy, W, H)
    top  = _project(_ego_to_veh(target_ego, 1.5), R, cam_pos, fx, fy, cx, cy, W, H)
    if base is not None and top is not None:
        cv2.line(img, base, top, (0, 220, 0), 3)
        cv2.circle(img, top, 8, (0, 220, 0), -1)
    elif top is not None:
        cv2.circle(img, top, 10, (0, 220, 0), -1)
    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_trajectory_ribbon(
    img: np.ndarray,
    plan_traj_ego: np.ndarray,
    cam_cfg: dict,
    color: Tuple[int, int, int] = (255, 255, 0),
    path_width: float = 0.9,
    alpha: float = 0.6,
    route_pts_ego: Optional[np.ndarray] = None,
    target_pt_ego: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Overlay planned trajectory ribbon, route dots, and target pole on camera image.

    Args:
        img: BGR image (H, W, 3).
        plan_traj_ego: (N, 2) planned trajectory, col0=lateral (left+), col1=forward.
        cam_cfg: camera config dict from NAVSIM_CAM_CONFIGS.
        color: BGR ribbon colour.
        path_width: ribbon half-width in metres.
        alpha: ribbon opacity.
        route_pts_ego: (M, 2) route waypoints in ego frame, or None.
        target_pt_ego: (2,) target waypoint in ego frame, or None.

    Returns:
        Annotated BGR image (copy).
    """
    img = img.copy()
    W = img.shape[1]
    H = img.shape[0]

    R, cam_pos = _build_extrinsics(cam_cfg)
    fx, fy, cx, cy = _build_intrinsics(cam_cfg)

    plan_traj_ego = np.asarray(plan_traj_ego, dtype=float)
    if plan_traj_ego.ndim == 2 and plan_traj_ego.shape[0] >= 2:
        img = _draw_ribbon(img, plan_traj_ego, R, cam_pos, fx, fy, cx, cy,
                           W, H, color, path_width, alpha)

    if route_pts_ego is not None and len(route_pts_ego) > 0:
        img = _draw_route_dots(img, np.asarray(route_pts_ego, dtype=float),
                               R, cam_pos, fx, fy, cx, cy, W, H)

    if target_pt_ego is not None:
        img = _draw_target_pole(img, np.asarray(target_pt_ego, dtype=float),
                                R, cam_pos, fx, fy, cx, cy, W, H)

    return img

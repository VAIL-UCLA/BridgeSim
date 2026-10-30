"""
DreamStream evaluator: replaces MetaDrive RGB perception with frames generated
by a DreamStream video-model server.

Every simulation frame is rendered as a traffic layout (projected HD map + 3D
boxes of vehicles and pedestrians). Each replan window is sent to the server as
one block: exactly 9 layouts for the first block and 12 afterwards. The window's
real frames are placed at evenly spaced slots (first and last pinned to the
block ends) and the slots in between are re-rendered from interpolated ego and
agent poses.
"""

import math
import numpy as np
import cv2
from typing import Optional, List, Tuple

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuplan.database.utils.boxes.box3d import BoxVisibility, box_in_image

from bridgesim.evaluation.core.batch_client import BatchClient
from bridgesim.evaluation.core.base_evaluator import BaseEvaluator
from bridgesim.evaluation.models.base_adapter import BaseModelAdapter

from PIL import Image
import os
import time


# ============================================================
# HD-Map color configuration (RGB for canvas)
# ============================================================

LANE_TYPES = [
    "LANE_SURFACE_STREET",
    "ROAD_LINE_BROKEN_SINGLE_WHITE",
    "LANE_SURFACE_UNSTRUCTURE",
    "ROAD_EDGE_SIDEWALK",
    "CROSSWALK",
    "ROAD_LINE_SOLID_SINGLE_WHITE",
]

_PL_COLORS_RAW = [None, [98, 183, 249], None, None, [206, 131, 63], [200, 36, 35]]
PL_COLORS = [
    tuple(v) if v is not None else None
    for v in _PL_COLORS_RAW
]
TYPE_TO_COLOR = dict(zip(LANE_TYPES, PL_COLORS))
PL_COLORS_FLOAT = [
    tuple(v_i / 255 for v_i in v) if v is not None else None
    for v in _PL_COLORS_RAW
]
TYPE_TO_COLOR_FLOAT = dict(zip(LANE_TYPES, PL_COLORS_FLOAT))

SKIP_LANE_TYPES = {"ROAD_EDGE_SIDEWALK", "LANE_SURFACE_STREET", "LANE_SURFACE_UNSTRUCTURE"}

# Layout rendering resolution
CONDITION_W, CONDITION_H = 1920, 1080
# Server input/output resolution
SERVER_W, SERVER_H = 1280, 704

# Rendering parameters
LINE_THICKNESS_PX = 8
CROSSWALK_THICKNESS_PX = 2
CROSSWALK_FACE_ALPHA = 0.15
CROSSWALK_EDGE_ALPHA = 0.6

BOX_COLOR_VEHICLE = (255, 0, 0)    # red in RGB
BOX_COLOR_PED = (0, 255, 0)        # green in RGB

# Server block sizes
FIRST_BLOCK_SIZE = 9
NORMAL_BLOCK_SIZE = 12


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate between two angles (radians) along the shortest arc."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t


# ============================================================
# Geometry helpers
# ============================================================

def _intersect_near_plane(A: np.ndarray, B: np.ndarray, eps: float = 1e-3):
    zA, zB = A[2], B[2]
    if zA > eps and zB > eps:
        return None
    if zA <= eps and zB <= eps:
        return None
    t = (eps - zA) / (zB - zA)
    return A + t * (B - A)


def _project_points(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = points_3d @ intrinsic.T
    depths = pts[:, 2:3].copy()
    depths[depths == 0] = 1e-8
    pts_2d = pts[:, :2] / depths
    return pts_2d


def _project_single_point(pt_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    p = intrinsic @ pt_3d
    if abs(p[2]) < 1e-8:
        p[2] = 1e-8
    return p[:2] / p[2]


def _hpr_to_rotation_matrix(yaw, pitch, roll):
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    return Ry @ Rx @ Rz


def _world_points_to_cam_batch(coords: np.ndarray, z_w: float,
                                ego_pos: np.ndarray, ego_theta: float,
                                rotation_matrix: np.ndarray, T: np.ndarray,
                                max_dist: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    N = coords.shape[0]
    centers = np.zeros((N, 3), dtype=np.float32)
    centers[:, 0] = coords[:, 0] - ego_pos[0]
    centers[:, 1] = coords[:, 1] - ego_pos[1]
    centers[:, 2] = z_w - ego_pos[2]

    dists_2d = np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2)
    valid_mask = dists_2d <= max_dist

    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), valid_mask

    c = centers[valid_mask]
    d = dists_2d[valid_mask]

    world_angles = np.arctan2(c[:, 1], c[:, 0])
    rel_angles = world_angles - ego_theta + np.pi / 2

    rot_x = d * np.cos(rel_angles)
    rot_y = d * np.sin(rel_angles)

    center_3d = np.stack([rot_x, -c[:, 2], rot_y], axis=1)
    cam_points = ((center_3d - T) @ rotation_matrix.T)

    return cam_points, valid_mask


# ============================================================
# OpenCV-based clipped rendering helpers
# ============================================================

def _clip_segment_near_plane(A, B, eps=1e-3):
    zA, zB = A[2], B[2]
    if zA > eps and zB > eps:
        return A, B
    if zA <= eps and zB <= eps:
        return None
    t = (eps - zA) / (zB - zA)
    C = A + t * (B - A)
    if zA > eps:
        return A, C
    else:
        return C, B


def _clip_line_to_rect(x0, y0, x1, y1, W, H):
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    def outcode(x, y):
        code = INSIDE
        if x < 0: code |= LEFT
        elif x > W: code |= RIGHT
        if y < 0: code |= BOTTOM
        elif y > H: code |= TOP
        return code
    out0, out1 = outcode(x0, y0), outcode(x1, y1)
    while True:
        if not (out0 | out1):
            return x0, y0, x1, y1
        elif out0 & out1:
            return None
        out = out0 or out1
        if out & TOP:
            x = x0 + (x1 - x0) * (H - y0) / (y1 - y0) if y1 != y0 else x0
            y = H
        elif out & BOTTOM:
            x = x0 + (x1 - x0) * (0 - y0) / (y1 - y0) if y1 != y0 else x0
            y = 0
        elif out & RIGHT:
            y = y0 + (y1 - y0) * (W - x0) / (x1 - x0) if x1 != x0 else y0
            x = W
        elif out & LEFT:
            y = y0 + (y1 - y0) * (0 - x0) / (x1 - x0) if x1 != x0 else y0
            x = 0
        if out == out0:
            x0, y0, out0 = x, y, outcode(x, y)
        else:
            x1, y1, out1 = x, y, outcode(x, y)


def _draw_clipped_line_cv2(canvas, A3d, B3d, intrinsic, color, thickness, W, H, eps=1e-3):
    result = _clip_segment_near_plane(A3d, B3d, eps)
    if result is None:
        return
    A_clip, B_clip = result
    A2d = _project_single_point(A_clip, intrinsic)
    B2d = _project_single_point(B_clip, intrinsic)
    clipped = _clip_line_to_rect(A2d[0], A2d[1], B2d[0], B2d[1], W, H)
    if clipped is None:
        return
    pt1 = (int(round(clipped[0])), int(round(clipped[1])))
    pt2 = (int(round(clipped[2])), int(round(clipped[3])))
    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)


def _render_polyline_cv2(canvas, points_3d, intrinsic, color, thickness, W, H, eps=1e-3):
    N = points_3d.shape[0]
    if N < 2:
        return
    for i in range(N - 1):
        _draw_clipped_line_cv2(canvas, points_3d[i], points_3d[i + 1],
                               intrinsic, color, thickness, W, H, eps)


def _clip_polygon_near_plane(poly_3d, eps=1e-3):
    N = poly_3d.shape[0]
    if N == 0:
        return poly_3d
    new_pts = []
    for i in range(N):
        A, B = poly_3d[i], poly_3d[(i + 1) % N]
        inA, inB = A[2] > eps, B[2] > eps
        if inA and inB:
            new_pts.append(B)
        elif inA and not inB:
            C = _intersect_near_plane(A, B, eps=eps)
            if C is not None:
                new_pts.append(C)
        elif not inA and inB:
            C = _intersect_near_plane(A, B, eps=eps)
            if C is not None:
                new_pts.append(C)
            new_pts.append(B)
    if not new_pts:
        return np.zeros((0, 3), dtype=poly_3d.dtype)
    return np.stack(new_pts, axis=0)


def _clip_polygon_to_rect(poly_2d, W, H):
    def clip_edge(poly, edge):
        if len(poly) == 0:
            return poly
        if edge == 'left':
            inside = lambda p: p[0] >= 0
            intersect_fn = lambda a, b: (0, a[1] + (b[1] - a[1]) * (0 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else a[1])
        elif edge == 'right':
            inside = lambda p: p[0] <= W
            intersect_fn = lambda a, b: (W, a[1] + (b[1] - a[1]) * (W - a[0]) / (b[0] - a[0]) if b[0] != a[0] else a[1])
        elif edge == 'bottom':
            inside = lambda p: p[1] >= 0
            intersect_fn = lambda a, b: (a[0] + (b[0] - a[0]) * (0 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else a[0], 0)
        elif edge == 'top':
            inside = lambda p: p[1] <= H
            intersect_fn = lambda a, b: (a[0] + (b[0] - a[0]) * (H - a[1]) / (b[1] - a[1]) if b[1] != a[1] else a[0], H)
        out = []
        n = len(poly)
        for i in range(n):
            A, B = poly[i], poly[(i + 1) % n]
            iA, iB = inside(A), inside(B)
            if iA and iB:
                out.append(B)
            elif iA and not iB:
                out.append(intersect_fn(A, B))
            elif not iA and iB:
                out.append(intersect_fn(A, B))
                out.append(B)
        return out
    poly = [(p[0], p[1]) for p in poly_2d]
    for edge in ['left', 'right', 'bottom', 'top']:
        poly = clip_edge(poly, edge)
        if len(poly) < 3:
            return np.zeros((0, 2))
    return np.array(poly, dtype=np.float64)


def _render_polygon_cv2(canvas, points_3d, intrinsic, fill_color, edge_color,
                        fill_alpha, edge_thickness, W, H, eps=1e-3):
    clipped_3d = _clip_polygon_near_plane(points_3d, eps)
    if clipped_3d.shape[0] < 3:
        return
    pts_2d = _project_points(clipped_3d, intrinsic)
    clipped_2d = _clip_polygon_to_rect(pts_2d, W, H)
    if len(clipped_2d) < 3:
        return
    int_pts = np.round(clipped_2d).astype(np.int32)
    if fill_alpha > 0:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [int_pts], fill_color)
        cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, canvas)
    cv2.polylines(canvas, [int_pts], isClosed=True, color=edge_color,
                  thickness=edge_thickness, lineType=cv2.LINE_AA)


def _render_box_cv2(box, canvas, intrinsic, color, thickness, W, H):
    corners_3d = box.corners()  # (3, 8)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        _draw_clipped_line_cv2(canvas, corners_3d[:, i], corners_3d[:, j],
                               intrinsic, color, thickness, W, H)


# ============================================================
# DreamStreamEvaluator
# ============================================================

class DreamStreamEvaluator(BaseEvaluator):
    """
    Closed-loop evaluator whose front-camera observations come from a
    DreamStream server.

    The policy replans every `replan_rate` frames; each replan window becomes
    one server block (9 layouts for the first block, 12 afterwards). During
    ego replay the policy sees the logged camera images, and the last replay
    image anchors the first block.
    """

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        scenario_path: str,
        output_dir: str,
        server_url: str = "http://127.0.0.1:8790",
        replan_rate: int = 6,
        # BaseEvaluator params with defaults
        traffic_mode: str = "log_replay",
        enable_vis: bool = False,
        save_perframe: bool = True,
        eval_mode: str = "closed_loop",
        controller_type: str = "pure_pursuit",
        sim_dt: float = 0.1,
        ego_replay_frames: int = 0,
        eval_frames: int = None,
        score_start_frame: int = None,
        vis_proposals: bool = True,
        first_frame_path: str = None,
        prompt: str = "a driving scene",
    ):
        super().__init__(
            model_adapter=model_adapter,
            scenario_path=scenario_path,
            output_dir=output_dir,
            traffic_mode=traffic_mode,
            enable_vis=enable_vis,
            save_perframe=save_perframe,
            eval_mode=eval_mode,
            controller_type=controller_type,
            replan_rate=replan_rate,
            sim_dt=sim_dt,
            ego_replay_frames=ego_replay_frames,
            eval_frames=eval_frames,
            score_start_frame=score_start_frame,
            vis_proposals=vis_proposals,
        )

        self.server_url = server_url
        self.replan_rate = replan_rate
        self.prompt = prompt

        assert 2 <= self.replan_rate <= 12, \
            "replan_rate must be between 2 and 12"

        # Optional anchor image for the first block, used instead of the last
        # ego-replay log image (e.g. an image-edited weather variant).
        self._first_frame_override = None
        if first_frame_path is not None:
            ff_img = np.array(Image.open(first_frame_path).convert("RGB"))
            if ff_img.shape[:2] != (CONDITION_H, CONDITION_W):
                ff_img = cv2.resize(
                    ff_img, (CONDITION_W, CONDITION_H),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            self._first_frame_override = ff_img

        # Anchor frame for the next block (the previous block's last output).
        if self._first_frame_override is not None:
            self.current_first_frame = self._first_frame_override.copy()
        else:
            self.current_first_frame = None

        # Per-window buffers: frame ids and the scene snapshots used to render
        # the server block.
        self._buffer_frame_ids: list[int] = []
        self._snapshot_buffer: list[dict] = []

        self.is_first_block = True
        self._cycle_frame_count = 0

        self.client = BatchClient(server_url)
        self.client.reset()

        # Get front camera config
        cam_configs = self.model_adapter.get_camera_configs()
        assert "CAM_F0" in cam_configs, (
            "DreamStreamEvaluator requires CAM_F0 in model adapter camera configs. "
            f"Available: {list(cam_configs.keys())}"
        )
        self.front_cam_config = cam_configs["CAM_F0"]
        self._setup_camera_params()

        print(f"DreamStream evaluator: server {self.server_url}, "
              f"replan every {self.replan_rate} frames, "
              f"blocks of {FIRST_BLOCK_SIZE}/{NORMAL_BLOCK_SIZE} layouts at {SERVER_W}x{SERVER_H}")

    def _setup_camera_params(self):
        cfg = self.front_cam_config
        self._cam_position = [cfg["y"], cfg["x"], cfg["z"]]
        self._cam_hpr = [-cfg["yaw"], cfg["pitch"], -cfg["roll"]]
        self._map_position = list(self._cam_position)
        self._map_hpr = list(self._cam_hpr)

        fov = cfg.get("fov", 70)
        fov_rad = np.radians(fov)
        fx = (CONDITION_W / 2) / np.tan(fov_rad / 2)
        fy = fx
        cx = CONDITION_W / 2
        cy = CONDITION_H / 2
        self._intrinsic = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

        yaw, pitch, roll = self._map_hpr[0], self._map_hpr[1], -self._map_hpr[2]
        self._map_rotation = _hpr_to_rotation_matrix(yaw, pitch, roll)
        self._map_T = np.array(
            [self._map_position[0], -self._map_position[2], self._map_position[1]]
        )
        self._map_cam_yaw = np.radians(yaw)

    # ------------------------------------------------------------------
    # Layout rendering (OpenCV)
    # ------------------------------------------------------------------

    def _snapshot_scene(self, env) -> Optional[dict]:
        """Capture the geometric state needed to render a layout: ego pose plus
        each vehicle/pedestrian's pose and box size."""
        ego_pos = None
        ego_theta = None
        objects = []
        for obj_id, obj in env.engine._spawned_objects.items():
            if obj.class_name == "DefaultVehicle":
                st = obj.get_state()
                ego_pos = np.asarray(st["position"], dtype=float)
                ego_theta = float(st["heading_theta"])
                continue
            is_vehicle = "Vehicle" in obj.class_name
            is_ped = "Pedestrian" in obj.class_name
            if not (is_vehicle or is_ped):
                continue
            st = obj.get_state()
            if is_vehicle:
                size = [st["size"][1], st["size"][0], st["size"][2]]
            else:
                size = [st["width"], st["length"], st["height"]]
            objects.append({
                "id": obj_id,
                "position": np.asarray(st["position"], dtype=float),
                "heading_theta": float(st["heading_theta"]),
                "roll": float(st["roll"]),
                "pitch": float(st["pitch"]),
                "is_vehicle": is_vehicle,
                "size": [float(s) for s in size],
            })
        if ego_pos is None:
            return None
        return {"ego_pos": ego_pos, "ego_theta": ego_theta, "objects": objects}

    def _render_condition(self, snap: Optional[dict]) -> np.ndarray:
        if snap is None:
            return np.zeros((CONDITION_H, CONDITION_W, 3), dtype=np.uint8)
        canvas = np.zeros((CONDITION_H, CONDITION_W, 3), dtype=np.uint8)
        self._render_hdmap_overlay_cv2(
            canvas, self.scenario_data, snap["ego_pos"], snap["ego_theta"],
            self._map_rotation, self._map_T, self._intrinsic,
        )
        self._render_objects_cv2(
            canvas, snap["objects"], snap["ego_pos"], snap["ego_theta"],
            self._map_cam_yaw, self._map_rotation, self._map_T, self._intrinsic,
        )
        return canvas

    @staticmethod
    def _interpolate_snapshot(a: dict, b: dict, t: float) -> dict:
        """Interpolate ego and agent poses between two snapshots. Agents present
        in only one snapshot are kept unchanged."""
        if t <= 0.0:
            return a
        if t >= 1.0:
            return b
        ego_pos = (1.0 - t) * a["ego_pos"] + t * b["ego_pos"]
        ego_theta = _lerp_angle(a["ego_theta"], b["ego_theta"], t)

        a_by_id = {o["id"]: o for o in a["objects"]}
        b_by_id = {o["id"]: o for o in b["objects"]}
        objects = []
        for oid, oa in a_by_id.items():
            ob = b_by_id.get(oid)
            if ob is None:
                objects.append(oa)
                continue
            objects.append({
                "id": oid,
                "position": (1.0 - t) * oa["position"] + t * ob["position"],
                "heading_theta": _lerp_angle(oa["heading_theta"], ob["heading_theta"], t),
                "roll": (1.0 - t) * oa["roll"] + t * ob["roll"],
                "pitch": (1.0 - t) * oa["pitch"] + t * ob["pitch"],
                "is_vehicle": oa["is_vehicle"],
                "size": oa["size"],
            })
        for oid, ob in b_by_id.items():
            if oid not in a_by_id:
                objects.append(ob)
        return {"ego_pos": ego_pos, "ego_theta": ego_theta, "objects": objects}

    def _render_interpolated_conditions(self, snapshots: List[dict],
                                        target_count: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Place the K real snapshots at evenly spaced slots (first/last at 0 and
        N-1) and render every slot from the interpolated surrounding snapshots.
        Returns (frames, key_positions), key_positions[i] being the slot of
        real frame i."""
        K = len(snapshots)
        N = target_count
        key_positions = np.linspace(0, N - 1, K)

        if K == 1:
            frame = self._render_condition(snapshots[0])
            return [frame.copy() for _ in range(N)], key_positions

        frames: List[np.ndarray] = []
        for out_idx in range(N):
            pos = float(out_idx)
            seg = int(np.searchsorted(key_positions, pos, side="right") - 1)
            seg = max(0, min(seg, K - 2))
            p0, p1 = key_positions[seg], key_positions[seg + 1]
            span = p1 - p0
            t = 0.0 if span < 1e-9 else (pos - p0) / span
            snap = self._interpolate_snapshot(snapshots[seg], snapshots[seg + 1], t)
            frames.append(self._render_condition(snap))
        return frames, key_positions

    def _render_hdmap_overlay_cv2(self, canvas, scenario_data, ego_pos, ego_theta,
                                   rotation_matrix, T, intrinsic):
        W, H = CONDITION_W, CONDITION_H
        for lane in scenario_data["map_features"].values():
            lane_type = lane.get("type", None)
            if lane_type is None or lane_type in SKIP_LANE_TYPES:
                continue
            color_rgb = TYPE_TO_COLOR.get(lane_type)
            color_float = TYPE_TO_COLOR_FLOAT.get(lane_type)
            if color_rgb is None and color_float is None:
                continue

            is_polyline = "polyline" in lane and lane["polyline"] is not None
            is_polygon = "polygon" in lane and lane["polygon"] is not None
            coords = lane.get("polyline") if is_polyline else lane.get("polygon")
            if coords is None:
                continue

            coords_np = np.asarray(coords, dtype=np.float32)
            if coords_np.ndim != 2 or coords_np.shape[0] < 2:
                continue

            cam_points, valid_mask = _world_points_to_cam_batch(
                coords_np, 0.0, ego_pos, ego_theta, rotation_matrix, T,
            )
            if cam_points.shape[0] < 2:
                continue

            if is_polyline:
                _render_polyline_cv2(
                    canvas, cam_points, intrinsic,
                    color_rgb, LINE_THICKNESS_PX, W, H,
                )
            elif is_polygon and cam_points.shape[0] >= 3:
                if color_float is not None:
                    fill_rgb = (
                        int(color_float[0] * 255),
                        int(color_float[1] * 255),
                        int(color_float[2] * 255),
                    )
                else:
                    fill_rgb = (128, 128, 128)
                _render_polygon_cv2(
                    canvas, cam_points, intrinsic,
                    fill_color=fill_rgb, edge_color=fill_rgb,
                    fill_alpha=CROSSWALK_FACE_ALPHA,
                    edge_thickness=CROSSWALK_THICKNESS_PX,
                    W=W, H=H,
                )

    def _render_objects_cv2(self, canvas, objects, ego_pos, ego_theta,
                            cam_yaw, rotation_matrix, T, intrinsic):
        """Render vehicle/pedestrian boxes from snapshot object states."""
        W, H = CONDITION_W, CONDITION_H
        for obj in objects:
            car_pos = np.asarray(obj["position"])
            center = car_pos - ego_pos
            world_angle = np.arctan2(center[1], center[0])
            rel_angle = world_angle - ego_theta + np.pi / 2
            dist = np.linalg.norm(center[:2])
            rot_pos_2d = dist * np.array([np.cos(rel_angle), np.sin(rel_angle)])
            center_3d = np.array([rot_pos_2d[0], -center[2], rot_pos_2d[1]])
            center_3d = (rotation_matrix @ (center_3d - T).T).T

            q_roll = Quaternion(axis=[1, 0, 0], angle=math.pi / 2 + obj["roll"])
            q_yaw = Quaternion(
                axis=[0, -1, 0],
                angle=math.pi / 2 + obj["heading_theta"] - ego_theta - cam_yaw,
            )
            q_pitch = Quaternion(axis=[0, 0, 1], angle=obj["pitch"])
            orientation = q_yaw * q_pitch * q_roll

            if obj["is_vehicle"]:
                color = BOX_COLOR_VEHICLE
                vis_size = (1600, 900)
            else:
                color = BOX_COLOR_PED
                vis_size = (W, H)

            box = Box(center=center_3d, size=obj["size"], orientation=orientation, label=0)
            if box_in_image(box, intrinsic, vis_size, vis_level=BoxVisibility.ANY):
                _render_box_cv2(box, canvas, intrinsic, color, LINE_THICKNESS_PX, W, H)

    # ------------------------------------------------------------------
    # Image / log helpers
    # ------------------------------------------------------------------

    def _load_log_image(self, frame_id: int) -> np.ndarray:
        """Load the front-camera image for a given simulation frame from the log.

        Dataset camera frames are recorded at 2 Hz while the simulation runs at
        10 Hz, so five consecutive simulation frames share a single dataset
        entry. We index `frame_info` by `frame_id // 5`.
        """
        dataset_idx = frame_id // 5
        frame_info = self.scenario_data['metadata']['frame_info'][dataset_idx]
        if 'cams' in frame_info:
            env_var = "NAVSIM_DATA_ROOT"  # e.g. <navsim>/sensor_blobs/test
            cam_f0_path = frame_info['cams']['CAM_F0']['data_path']
        elif 'CAM_FRONT' in frame_info:
            env_var = "NUSCENES_DATA_ROOT"  # e.g. <nuscenes>/samples/CAM_FRONT
            cam_f0_path = frame_info['CAM_FRONT']
        else:
            raise KeyError(
                f"No CAM_F0/CAM_FRONT entry in frame_info[{dataset_idx}]"
            )
        sensor_root = os.environ.get(env_var)
        if not sensor_root:
            raise EnvironmentError(
                f"{env_var} is not set. DreamStream reads the logged front-camera "
                f"images for ego replay and the first-frame anchor; point "
                f"{env_var} at the dataset's camera image root (see docs/dreamstream.md).")
        return np.array(Image.open(os.path.join(sensor_root, cam_f0_path)))

    def _finalize_image_output(self, env, frame_id, img):
        """Resize img to the policy's CAM_F0 size; with --enable-vis also save it
        and the MetaDrive RGB view of the same frame."""
        target_w = self.front_cam_config.get("width", 1600)
        target_h = self.front_cam_config.get("height", 900)
        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if self.enable_vis:
            frame_output_path = self.output_dir / f"{frame_id:05d}"
            frame_output_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_output_path / "cam_front.jpg"), img[:, :, ::-1])

            sensor = env.engine.get_sensor('rgb_camera')
            for name, cam_config in self.model_adapter.get_camera_configs().items():
                sensor.lens.setFov(cam_config['fov'])
                sensor_output = sensor.perceive(
                    to_float=False,
                    new_parent_node=env.agent.origin,
                    position=(cam_config['y'], cam_config['x'], cam_config['z']),
                    hpr=(-cam_config['yaw'], cam_config['pitch'], -cam_config['roll'])
                )
                sensor_data = sensor_output.get() if hasattr(sensor_output, 'get') else sensor_output
                cv2.imwrite(str(frame_output_path / f"{name.lower()}_md.jpg"), sensor_data)

        return {"CAM_F0": img}

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def perceive(self, env, frame_id):
        """
        Return the policy's CAM_F0 observation for this frame.

        During ego replay this is the logged camera image. Afterwards a layout
        is rendered for every frame; once `replan_rate` layouts are buffered,
        the window is expanded to a server block (9 slots for the first block,
        12 afterwards), sent to the server, and the last generated frame becomes
        both the policy input and the anchor of the next block.
        """
        if frame_id < self.ego_replay_frames:
            log_img = self._load_log_image(frame_id)
            if self._first_frame_override is None:
                self.current_first_frame = log_img.copy()

            if self.enable_vis:
                condition = self._render_condition(self._snapshot_scene(env))
                frame_output_path = self.output_dir / f"{frame_id:05d}"
                frame_output_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(frame_output_path / "condition.jpg"),
                    condition[:, :, ::-1],
                )
                # Save at the server resolution so all cam_f0.jpg frames of a
                # scenario have the same size for the video.
                if log_img.shape[:2] != (SERVER_H, SERVER_W):
                    log_img_vis = cv2.resize(
                        log_img, (SERVER_W, SERVER_H),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    log_img_vis = log_img
                cv2.imwrite(
                    str(frame_output_path / "cam_f0.jpg"),
                    log_img_vis[:, :, ::-1],
                )

            return self._finalize_image_output(env, frame_id, log_img)

        # Without ego replay, anchor the first block on log frame 0.
        if frame_id == 0 and self.current_first_frame is None:
            self.current_first_frame = self._load_log_image(0)

        snapshot = self._snapshot_scene(env)
        condition = self._render_condition(snapshot)
        self._buffer_frame_ids.append(frame_id)
        self._snapshot_buffer.append(snapshot)
        self._cycle_frame_count += 1

        if self.enable_vis:
            frame_output_path = self.output_dir / f"{frame_id:05d}"
            frame_output_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(frame_output_path / "condition.jpg"),
                condition[:, :, ::-1],
            )

        cycle_complete = (self._cycle_frame_count >= self.replan_rate)

        if cycle_complete:
            time_start = time.time()
            server_block_size = FIRST_BLOCK_SIZE if self.is_first_block else NORMAL_BLOCK_SIZE
            real_frame_ids = list(self._buffer_frame_ids)

            conditions_full, key_positions = self._render_interpolated_conditions(
                self._snapshot_buffer, server_block_size
            )
            conditions_for_server = [
                cv2.resize(c, (SERVER_W, SERVER_H), interpolation=cv2.INTER_AREA)
                for c in conditions_full
            ]
            # Generated frame shown for each real frame of the window
            real_to_rendered = [int(round(p)) for p in key_positions]

            first_frame_resized = cv2.resize(
                self.current_first_frame, (SERVER_W, SERVER_H),
                interpolation=cv2.INTER_AREA,
            )

            rendered_frames = self.client.infer(
                conditions_for_server,
                prompt=self.prompt,
                first_frame=first_frame_resized,
            )

            elapsed = time.time() - time_start
            print(f"Frames {real_frame_ids[0]}-{real_frame_ids[-1]}: "
                  f"{len(rendered_frames)} frames generated in {elapsed:.2f}s")

            if self.enable_vis:
                for i, real_fid in enumerate(real_frame_ids):
                    rf = rendered_frames[real_to_rendered[i]]
                    if rf.shape[:2] != (SERVER_H, SERVER_W):
                        rf = cv2.resize(
                            rf, (SERVER_W, SERVER_H),
                            interpolation=cv2.INTER_LANCZOS4
                            if rf.shape[1] < SERVER_W else cv2.INTER_AREA,
                        )
                    rf_output_path = self.output_dir / f"{real_fid:05d}"
                    rf_output_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(rf_output_path / "cam_f0.jpg"),
                        rf[:, :, ::-1],
                    )
                    self.render_cam_f0_vis(None, real_fid, None, None)

                if not self.is_first_block:
                    last_block_frame_id = real_frame_ids[0] - 1
                    if last_block_frame_id >= 0:
                        self.render_cam_f0_vis(None, last_block_frame_id, None, None)

            # The last generated frame (slot N-1 = last real frame) anchors the
            # next block.
            last_rendered = rendered_frames[-1]
            if last_rendered.shape[:2] != (CONDITION_H, CONDITION_W):
                self.current_first_frame = cv2.resize(
                    last_rendered, (CONDITION_W, CONDITION_H),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                self.current_first_frame = last_rendered.copy()

            self._buffer_frame_ids = []
            self._snapshot_buffer = []
            self._cycle_frame_count = 0
            self.is_first_block = False

        # The policy sees the latest generated frame on the frame that completes
        # a window and on the first frame of the next one (the replan frame);
        # on the other frames it gets the layout.
        if cycle_complete or self._cycle_frame_count == 1:
            img = self.current_first_frame
        else:
            img = condition

        return self._finalize_image_output(env, frame_id, img)

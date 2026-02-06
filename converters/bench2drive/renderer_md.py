#!/usr/bin/env python3
import cv2
import numpy as np
from tqdm import tqdm
from numpy import array
import math
from typing import Tuple, List

# ==========================================
# 1. NEW CONFIGURATION
# ==========================================

CAM_CONFIGS = {
    'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_BACK': {'x': -2.0, 'y': 0.0, 'z': 1.60, 'yaw': 180.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900},
    'CAM_BACK_LEFT': {'x': -0.32, 'y': -0.55, 'z': 1.60, 'yaw': -110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'CAM_BACK_RIGHT': {'x': -0.32, 'y': 0.55, 'z': 1.60, 'yaw': 110.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
    'TOP_DOWN': {'x': 0.0, 'y': 0.0, 'z': 50.0, 'yaw': 0.0, 'pitch': -90.0, 'roll': 0.0, 'fov': 110, 'width': 1600, 'height': 900}
}

COLOR_TABLE = {
    'lanelines': np.array([98, 183, 249], np.uint8),  # Light Blue
    'lanes': np.array([56, 103, 221], np.uint8),  # Dark Blue
    'road_boundaries': np.array([200, 36, 35], np.uint8),  # Dark Red
    'crosswalks': np.array([206, 131, 63], np.uint8),  # Earth Yellow
    'traffic_light_red': np.array([255, 0, 0], np.uint8),  # Red
    'traffic_light_yellow': np.array([255, 255, 0], np.uint8),  # Yellow
    'traffic_light_green': np.array([0, 255, 0], np.uint8),  # Green
    'traffic_light_unknown': np.array([255, 255, 255], np.uint8),  # White
    'pedestrian': np.array( [255, 0, 255], np.uint8),  # Cyan
    'vehicle': np.array([0, 128, 255], np.uint8),  # Blue
    'bicycle': np.array([255, 255, 0], np.uint8),  # Black
}

# ==========================================
# 2. MATHEMATICAL HELPERS
# ==========================================

def build_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 SE(3) Homogeneous Matrix"""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, t
    return T

def yaw_to_rot(yaw: float) -> np.ndarray:
    """2D Rotation around Z (for vehicle heading)"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float32)

def euler_to_rot(yaw_deg, pitch_deg, roll_deg):
    """
    Converts Euler angles (degrees) to a 3x3 Rotation Matrix.
    Order: Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    
    # Rz (Yaw)
    cz, sz = np.cos(y), np.sin(y)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    # Ry (Pitch)
    cy, sy = np.cos(p), np.sin(p)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    
    # Rx (Roll)
    cx, sx = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    
    return Rz @ Ry @ Rx

def get_cam_matrices_from_config(cfg):
    """
    Derives Intrinsics (K) and Extrinsics (T_lidar_cam) from the config dict.
    """
    # 1. Intrinsics (K)
    W, H = cfg['width'], cfg['height']
    fov = cfg['fov']
    
    # Calculate Focal Length from Horizontal FOV
    # f_x = (W / 2) / tan(fov / 2)
    f_x = (W / 2.0) / np.tan(np.radians(fov) / 2.0)
    f_y = f_x  # Assume square pixels
    c_x, c_y = W / 2.0, H / 2.0
    
    K = np.array([
        [f_x, 0.0, c_x],
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # 2. Extrinsics
    # Rotation of the mounting point (Yaw/Pitch/Roll) in Vehicle Frame
    R_mount = euler_to_rot(-cfg['yaw'], cfg['pitch'], cfg['roll'])
    t_mount = np.array([cfg['x'], cfg['y'], cfg['z']], dtype=np.float32)
    
    # 3. Basis Change: Optical (Z-fwd, X-right, Y-down) -> Vehicle (X-fwd, Y-left, Z-up)
    # The columns of this matrix represent where the Optical Basis vectors 
    # point in the Vehicle Frame.
    # Col 0: Optical X (Right) -> Vehicle -Y 
    # Col 1: Optical Y (Down)  -> Vehicle -Z
    # Col 2: Optical Z (Fwd)   -> Vehicle +X
    R_optical_to_vehicle = np.array([
        [ 0,  0,  1],  # Vehicle X component
        [-1,  0,  0],  # Vehicle Y component
        [ 0, -1,  0]   # Vehicle Z component
    ], dtype=np.float32)
    
    # Composition: First rotate the "camera box" into position (R_mount), 
    # then handle the internal axis swap (R_optical_to_vehicle).
    R_cam_in_lidar = R_mount @ R_optical_to_vehicle
    
    return K, R_cam_in_lidar, t_mount

def world_to_camera_T(lidar_pos, lidar_yaw,
                      cam2lidar_t, cam2lidar_R) -> np.ndarray:
    """
    Construct World -> Camera Transformation
    """
    # 1. World -> Lidar (Ego)
    T_w_lidar = build_se3(yaw_to_rot(lidar_yaw), lidar_pos) 
    
    # 2. Lidar -> Camera (Optical)
    # cam2lidar_R/t represents the Pose of the Camera in the Lidar Frame
    T_lidar_cam = build_se3(cam2lidar_R, cam2lidar_t)
    
    # 3. World -> Camera
    T_w_cam = T_w_lidar @ T_lidar_cam
    
    # Return Inverse (Camera <- World)
    return np.linalg.inv(T_w_cam)


# ==========================================
# 3. PROJECTION & DRAWING UTILS
# ==========================================

def project_points_cam(points_cam: np.ndarray,
                       K: np.ndarray, img_hw) -> Tuple[np.ndarray, np.ndarray]:
    """
    Camera Points -> Pixel Coords & Valid Mask
    points_cam: (N,3)
    """
    x, y, z = points_cam.T
    eps_mask = z > 1e-3
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    H, W = img_hw
    uv = np.stack([u, v], axis=1)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = eps_mask & in_img
    return uv.astype(np.int32), valid

def reconstruct_lane_boundaries(polyline: np.ndarray, widths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(polyline) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    directions = np.diff(polyline[:, :2], axis=0)
    directions = np.vstack([directions, directions[-1:]])
    normals = np.hstack([-directions[:, 1:2], directions[:, 0:1]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    left_points_2d = polyline[:, :2] + normals * widths[:, 0:1]
    right_points_2d = polyline[:, :2] - normals * widths[:, 1:2]
    left_boundary_3d = np.hstack([left_points_2d, np.zeros((left_points_2d.shape[0], 1), dtype=np.float32)])
    right_boundary_3d = np.hstack([right_points_2d, np.zeros((right_points_2d.shape[0], 1), dtype=np.float32)])
    return left_boundary_3d, right_boundary_3d

def draw_polyline_depth(canvas, polyline3d, T_w2c, K, color,
                        radius=8, seg_interval=0.5,
                        near=1e-3, depth_max=80.):
    H, W = canvas.shape[:2]
    pts_cam = (T_w2c[:3, :3] @ polyline3d.T + T_w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    cam_mask = z >= near
    proj_uv = (K @ pts_cam.T)[:2].T
    proj_uv /= (z[:, None] + 1e-6)
    u, v = proj_uv[:, 0], proj_uv[:, 1]

    for i in range(len(pts_cam) - 1):
        p1c, p2c = pts_cam[i].copy(), pts_cam[i + 1].copy()
        z1, z2 = z[i], z[i + 1]

        if z1 < near and z2 < near: continue
        if z1 < near or z2 < near:
            t = (near - z1) / (z2 - z1) if z1 < near else (near - z2) / (z1 - z2)
            inter = p1c + t * (p2c - p1c) if z1 < near else p2c + t * (p1c - p2c)
            if z1 < near: p1c, z1 = inter, near
            else: p2c, z2 = inter, near
            p = (K @ p1c) if z1 == near and (p1c is inter) else (K @ p2c)
            if z1 == near and (p1c is inter): proj_uv[i] = p[:2] / p[2]
            else: proj_uv[i + 1] = p[:2] / p[2]
            u, v = proj_uv[:, 0], proj_uv[:, 1]

        coord_limit = 100000000
        
        u1_safe = np.clip(u[i], -coord_limit, coord_limit)
        v1_safe = np.clip(v[i], -coord_limit, coord_limit)
        u2_safe = np.clip(u[i+1], -coord_limit, coord_limit)
        v2_safe = np.clip(v[i+1], -coord_limit, coord_limit)

        p1 = (int(round(u1_safe)), int(round(v1_safe)))
        p2 = (int(round(u2_safe)), int(round(v2_safe)))
        inside = (0 <= p1[0] < W and 0 <= p1[1] < H and 0 <= p2[0] < W and 0 <= p2[1] < H)
        if inside:
            p1_img, p2_img = p1, p2
        else:
            ok, p1_img, p2_img = cv2.clipLine((0, 0, W - 1, H - 1), p1, p2)
            if not ok: continue

        depth_mean = max(min((z1 + z2) * 0.5, depth_max), 0.)
        alpha = (depth_max - depth_mean) / depth_max
        col = (alpha * color).astype(np.uint8).tolist()
        cv2.line(canvas, p1_img, p2_img, col, radius, cv2.LINE_AA)

def _sutherland_hodgman(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    def clip_edge(pts: List[np.ndarray], inside_fn, intersect_fn):
        if not pts: return []
        output = []
        prev = pts[-1]
        prev_inside = inside_fn(prev)
        for curr in pts:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside: output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside: output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    pts = [np.asarray(p, float) for p in poly.tolist()]
    pts = clip_edge(pts, lambda p: p[0] >= 0, lambda p, q: p + (q - p) * ((0 - p[0]) / (q[0] - p[0])))
    if not pts: return np.empty((0, 2))
    pts = clip_edge(pts, lambda p: p[0] <= w - 1, lambda p, q: p + (q - p) * ((w - 1 - p[0]) / (q[0] - p[0])))
    if not pts: return np.empty((0, 2))
    pts = clip_edge(pts, lambda p: p[1] >= 0, lambda p, q: p + (q - p) * ((0 - p[1]) / (q[1] - p[1])))
    if not pts: return np.empty((0, 2))
    pts = clip_edge(pts, lambda p: p[1] <= h - 1, lambda p, q: p + (q - p) * ((h - 1 - p[1]) / (q[1] - p[1])))
    return np.asarray(pts, dtype=np.float32)

def draw_polygon_depth(canvas: np.ndarray, hull3d: np.ndarray, T_w2c: np.ndarray, K: np.ndarray, color: np.ndarray, depth_max) -> None:
    pts_cam = (T_w2c[:3, :3] @ hull3d.T + T_w2c[:3, 3:4]).T
    in_front = pts_cam[:, 2] > 1e-6
    if not np.any(in_front): return
    pts_cam_front = pts_cam[in_front]
    uv_h = (K @ pts_cam_front.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    h, w = canvas.shape[:2]
    poly_clipped = _sutherland_hodgman(uv, w, h)
    if poly_clipped.shape[0] < 3: return
    hull_uv = poly_clipped.astype(np.int32)
    depth_mean = float(np.clip(pts_cam_front[:, 2].mean(), 0.0, depth_max))
    alpha = (depth_max - depth_mean) / depth_max
    col = (alpha * np.asarray(color, dtype=float)).astype(np.uint8).tolist()
    cv2.fillConvexPoly(canvas, hull_uv, col)

def vehicle_corners_local(L, W, H):
    return np.array([
        [L / 2, W / 2, H / 2], [L / 2, -W / 2, H / 2], [-L / 2, -W / 2, H / 2], [-L / 2, W / 2, H / 2],
        [L / 2, W / 2, -H / 2], [L / 2, -W / 2, -H / 2], [-L / 2, -W / 2, -H / 2], [-L / 2, W / 2, -H / 2],
    ], dtype=np.float32)

def draw_cuboids_with_occlusion(canvas, bboxes, T_w2c, K, depth_max=120.0):
    H, W = canvas.shape[:2]
    base_face_colors = [
        (247, 37, 133), (76, 201, 240), (114, 9, 183),
        (67, 97, 238), (58, 12, 163), (58, 12, 163),
    ]
    face_indices = [
        [0, 1, 5, 4], [2, 3, 7, 6], [3, 0, 4, 7],
        [1, 2, 6, 5], [3, 2, 1, 0], [4, 5, 6, 7],
    ]
    faces_to_draw = []
    num_vehicles = bboxes.shape[0]
    for vi in range(num_vehicles):
        info = bboxes[vi]
        pos, L, Wd, H_box, yaw = info[:3], info[3], info[4], info[5], info[6]
        corners_loc = vehicle_corners_local(L, Wd, H_box)
        corners_world = (yaw_to_rot(yaw) @ corners_loc.T).T + pos
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, (H, W))
        if valid.sum() < 4: continue
        
        for fi, idxs in enumerate(face_indices):
            pts_cam_face = pts_cam[idxs]
            if np.all(pts_cam_face[:, 2] <= 0): continue
            z_vals = pts_cam_face[:, 2].clip(0, depth_max)
            z_mean = float(np.mean(z_vals))
            if not np.any(valid[idxs]): continue
            poly_2d = np.array([uv[j] for j in idxs], dtype=np.int32)
            faces_to_draw.append({'poly': poly_2d, 'depth': z_mean, 'base_color': base_face_colors[fi]})
            
    faces_to_draw.sort(key=lambda x: x['depth'], reverse=True)
    for face in faces_to_draw:
        poly, depth_mean, (base_B, base_G, base_R) = face['poly'], face['depth'], face['base_color']
        alpha = np.clip((depth_max - depth_mean) / depth_max, 0.0, 1.0)
        B, G, R = int(base_B * alpha), int(base_G * alpha), int(base_R * alpha)
        cv2.fillConvexPoly(canvas, poly, (B, G, R), cv2.LINE_AA)

def draw_cuboid_at(canvas, center_pos, dims, T_w2c, K, color_rgb=(0, 255, 0), thickness=-1):
    H_img, W_img = canvas.shape[:2]
    L, W, H_box = dims
    local_corners = np.array([
        [ L/2,  W/2, 0.0], [ L/2, -W/2, 0.0], [-L/2, -W/2, 0.0], [-L/2,  W/2, 0.0],
        [ L/2,  W/2, H_box], [ L/2, -W/2, H_box], [-L/2, -W/2, H_box], [-L/2,  W/2, H_box]
    ], dtype=np.float32)
    world_corners = local_corners + np.array(center_pos, dtype=np.float32).reshape(3,)
    pts_cam = (T_w2c[:3, :3] @ world_corners.T + T_w2c[:3, 3:4]).T
    uv, valid = project_points_cam(pts_cam, K, (H_img, W_img))
    face_idxs = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    
    for idxs in face_idxs:
        if not np.any([valid[i] for i in idxs]): continue
        if np.all(pts_cam[idxs, 2] <= 0): continue
        poly2d = np.array([uv[i] for i in idxs], dtype=np.int32)
        if thickness < 0: cv2.fillConvexPoly(canvas, poly2d, color_rgb, cv2.LINE_AA)
        else: cv2.polylines(canvas, [poly2d], isClosed=True, color=color_rgb, thickness=thickness, lineType=cv2.LINE_AA)

# ==========================================
# 4. UPDATED RENDERER
# ==========================================

class ScenarioRenderer:
    def __init__(self, camera_list=None, depth_max=120.0):
        """
        camera_list: list of strings (keys in CAM_CONFIGS). If None, use all keys.
        """
        self.depth_max = depth_max
        
        # Select cameras
        if camera_list is None:
            self.target_cams = list(CAM_CONFIGS.keys())
        else:
            self.target_cams = [c for c in camera_list if c in CAM_CONFIGS]

        # Precompute Matrices for requested cameras
        self.camera_models = {}
        for cam_name in self.target_cams:
            cfg = CAM_CONFIGS[cam_name]
            K, R_cam_in_lidar, t_mount = get_cam_matrices_from_config(cfg)
            
            self.camera_models[cam_name] = {
                "K": K,
                "R_lidar_cam": R_cam_in_lidar,
                "t_lidar_cam": t_mount,
                "width": cfg['width'],
                "height": cfg['height']
            }

    def observe(self, scenario):
        lidar_pos = scenario['ego_pos_3d']
        lidar_yaw = scenario['ego_heading']
        ret_dict = {}
        
        for cam_id, model in self.camera_models.items():
            W, H = model['width'], model['height']
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            
            # --- Generate Sky/Ground (Optional, makes it look nicer) ---
            # You can uncomment this if you want the gradient background:
            # canvas = make_sky_ground_canvas(H, W) 
            
            # Get Extrinsics and Intrinsics
            cam_t = model["t_lidar_cam"]  
            cam_R = model["R_lidar_cam"]  
            K = model["K"]
            
            # Create World -> Camera Matrix
            # Note: We pass raw cam_t and cam_R directly. No hacky offsets needed anymore.
            T_w2c = world_to_camera_T(lidar_pos, lidar_yaw, cam_t, cam_R)

            # 1. Traffic Lights
            for feat in scenario.get('traffic_lights', []):
                is_red = feat[1]
                xy = feat[2]
                z_base = 5.0  # Approx height
                pos_world = [xy[0], xy[1], z_base]
                dims = (0.5, 0.5, 1.0)
                col = COLOR_TABLE['traffic_light_red'].tolist() if is_red else COLOR_TABLE['traffic_light_green'].tolist()
                draw_cuboid_at(canvas, pos_world, dims, T_w2c, K, color_rgb=col, thickness=-1)
            
            # 2. Map Features (Lanes, Crosswalks)
            for feat in scenario['map_features'].values():
                ftype = feat['type']
                if 'LANE' in ftype:      
                    if 'polygon' in feat:
                        poly2d = feat['polygon'].astype(np.float32)
                        pts3d = np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])
                        pts_dist = np.linalg.norm(poly2d - lidar_pos[np.newaxis, :2], axis=1)
                        if np.min(pts_dist) > self.depth_max: continue
                        draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'], radius=2, depth_max=self.depth_max)

                    elif 'polyline' in feat and 'width' in feat:
                        polyline_3d = feat['polyline'].astype(np.float32)
                        widths = feat['width'].astype(np.float32)
                        pts_dist = np.linalg.norm(polyline_3d[:, :2] - lidar_pos[np.newaxis, :2], axis=1)
                        if np.min(pts_dist) > self.depth_max: continue
                        
                        left, right = reconstruct_lane_boundaries(polyline_3d, widths)
                        draw_polyline_depth(canvas, left, T_w2c, K, COLOR_TABLE['lanelines'], radius=2, depth_max=self.depth_max)
                        draw_polyline_depth(canvas, right, T_w2c, K, COLOR_TABLE['lanelines'], radius=2, depth_max=self.depth_max)

                elif 'CROSSWALK' in ftype or 'SPEED_BUMP' in ftype:
                    poly2d = feat['polygon'].astype(np.float32)
                    pts3d = np.hstack([poly2d[:, :2], np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polygon_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['crosswalks'], self.depth_max)
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['lanelines'], depth_max=self.depth_max)

                elif 'BOUNDARY' in ftype or 'SOLID' in ftype:
                    if 'polyline' not in feat: continue
                    poly2d = feat['polyline'].astype(np.float32)
                    pts3d = np.hstack([poly2d[:, :2], np.zeros((poly2d.shape[0], 1), np.float32)])
                    draw_polyline_depth(canvas, pts3d, T_w2c, K, COLOR_TABLE['road_boundaries'], radius=10, depth_max=self.depth_max)
            
            # 3. Dynamic Objects (Vehicles)
            if "anns" in scenario:
                anns = scenario["anns"]
                bboxes = anns["gt_boxes_world"]
                draw_cuboids_with_occlusion(canvas, bboxes, T_w2c, K, depth_max=self.depth_max)
            
            ret_dict[cam_id] = canvas            

        return ret_dict

# Re-included make_sky_ground_canvas just in case needed, unchanged
def make_sky_ground_canvas(H, W, horizon=0.60, sky_top=(180,120,60), sky_horizon=(230,205,185), 
                           ground_far=(105,105,105), ground_near=(35,35,35), sun=None, 
                           vignette_strength=0.22, noise_std=2, seed=None):
    if seed is not None: np.random.seed(seed)
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    sky_mask = (y < horizon).astype(np.float32)
    t_sky = np.clip(y / max(horizon, 1e-6), 0, 1)
    t_gnd = np.clip((y - horizon) / max(1 - horizon, 1e-6), 0, 1)
    sky_top  = np.array(sky_top,  np.float32)[None, None, :]
    sky_hori = np.array(sky_horizon, np.float32)[None, None, :]
    gnd_far  = np.array(ground_far,  np.float32)[None, None, :]
    gnd_near = np.array(ground_near, np.float32)[None, None, :]
    sky = (1 - t_sky) * sky_top + t_sky * sky_hori
    gnd = (1 - t_gnd) * gnd_far + t_gnd * gnd_near
    row_bg = sky_mask * sky + (1 - sky_mask) * gnd
    bg = np.broadcast_to(row_bg, (H, W, 3)).astype(np.float32).copy()
    return np.clip(bg, 0, 255).astype(np.uint8)

# Example Usage Block (Commented out)
# if __name__ == "__main__":
#     renderer = ScenarioRenderer(camera_list=['CAM_FRONT', 'CAM_BACK', 'TOP_DOWN'])
#     # scenario = { ... load your data here ... }
#     # imgs = renderer.observe(scenario)
#!/usr/bin/env python3
"""
Visualize model's predicted trajectory as a ribbon overlay on front camera view.
Similar to Tesla's navigation path visualization.

Usage:
    python scripts/vis_trajectory_on_cam.py <scenario_pkl_path> [--frame 0] [--output vis_output.jpg]
"""

import cv2
import numpy as np
import argparse
import os
import sys


def project_ego_waypoints_to_camera(waypoints_ego, cam_config, image_shape, z_height=0.0):
    """
    Project ego-frame waypoints onto front camera image.
    
    Args:
        waypoints_ego: (N, 2) array in ego frame, [:, 0]=lateral, [:, 1]=forward
        cam_config: dict with x, y, z, yaw, pitch, roll, fov, width, height
        image_shape: (H, W) of the actual image
        z_height: height above ground (default 0.0 for ground points)
        
    Returns:
        (N, 2) array of pixel coordinates (u, v)
    """
    H, W = image_shape[:2]
    fov_deg = cam_config['fov']
    
    # Camera intrinsics from FOV
    fov_rad = np.radians(fov_deg)
    fx = W / (2 * np.tan(fov_rad / 2))
    fy = fx  # Square pixels
    cx, cy = W / 2, H / 2
    
    # Camera extrinsics relative to ego
    # MetaDrive camera position: (y_offset, x_offset, z_offset) in perceive()
    cam_x = cam_config['x']   # forward offset from ego center
    cam_y = cam_config['y']   # lateral offset
    cam_z = cam_config['z']   # height
    cam_yaw = np.radians(cam_config['yaw'])
    
    N = waypoints_ego.shape[0]
    points_3d = np.zeros((N, 3))
    # Ego frame: lateral=[:,0], forward=[:,1], z=height
    points_3d[:, 0] = waypoints_ego[:, 0]  # lateral (left positive)
    points_3d[:, 1] = waypoints_ego[:, 1]  # forward
    points_3d[:, 2] = z_height              # height above ground
    
    # Transform to camera frame
    # Camera is at (cam_x forward, cam_y lateral, cam_z height) from ego center
    # Camera looks forward along ego's forward direction (for CAM_F0, yaw=0)
    
    # Step 1: Translate to camera origin
    points_cam = points_3d.copy()
    points_cam[:, 0] -= cam_y    # lateral offset
    points_cam[:, 1] -= cam_x    # forward offset
    points_cam[:, 2] -= cam_z    # height (camera is above ground, so z becomes negative)
    
    # Step 2: Rotate by camera yaw (for non-front cameras)
    if abs(cam_yaw) > 1e-6:
        cos_y = np.cos(-cam_yaw)
        sin_y = np.sin(-cam_yaw)
        x_rot = cos_y * points_cam[:, 0] - sin_y * points_cam[:, 1]
        y_rot = sin_y * points_cam[:, 0] + cos_y * points_cam[:, 1]
        points_cam[:, 0] = x_rot
        points_cam[:, 1] = y_rot
    
    # Step 3: Convert ego frame to camera frame
    # Ego: X=lateral(left+), Y=forward, Z=up
    # Camera (OpenCV): X=right, Y=down, Z=forward
    cam_coords = np.zeros_like(points_cam)
    cam_coords[:, 0] = -points_cam[:, 0]  # ego lateral(left+) -> cam X(right+)
    cam_coords[:, 1] = -points_cam[:, 2]  # ego Z(up) -> cam Y(down)
    cam_coords[:, 2] = points_cam[:, 1]   # ego forward -> cam Z(forward)
    
    # Filter: only keep points in front of camera
    valid = cam_coords[:, 2] > 0.1
    
    # Project to image
    u = fx * cam_coords[:, 0] / cam_coords[:, 2] + cx
    v = fy * cam_coords[:, 1] / cam_coords[:, 2] + cy
    
    # Clip v to image bounds but keep the point (for ribbon continuity)
    u = np.clip(u, -W, 2 * W)
    v = np.clip(v, -H, 2 * H)
    
    pixels = np.column_stack([u, v])
    pixels[~valid] = -1  # Mark invalid points
    
    return pixels, valid


def _draw_route_and_target(img, cam_config, route_pts_ego=None, target_pt_ego=None):
    """Draw route waypoints (red) and target (green) on camera image, independent of ribbon."""
    H, W = img.shape[:2]
    
    # Route waypoints (RED dots)
    if route_pts_ego is not None and len(route_pts_ego) > 0:
        pts_route, valid_route = project_ego_waypoints_to_camera(route_pts_ego, cam_config, img.shape)
        for i in range(len(pts_route)):
            if valid_route[i]:
                x, y = int(pts_route[i][0]), int(pts_route[i][1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
                    cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
    
    # Target waypoint (GREEN)
    if target_pt_ego is not None:
        pts_base, valid_base = project_ego_waypoints_to_camera(
            target_pt_ego.reshape(1, 2), cam_config, img.shape, z_height=0.0)
        pts_top, valid_top = project_ego_waypoints_to_camera(
            target_pt_ego.reshape(1, 2), cam_config, img.shape, z_height=1.5)
        if valid_top[0]:
            x_top, y_top = int(pts_top[0][0]), int(pts_top[0][1])
            x_top = max(0, min(W - 1, x_top))
            y_top = max(0, min(H - 1, y_top))
            if valid_base[0]:
                x_base, y_base = int(pts_base[0][0]), int(pts_base[0][1])
                x_base = max(0, min(W - 1, x_base))
                if y_base < H + 100:
                    cv2.line(img, (x_top, y_top), (x_base, y_base), (0, 255, 0), 8)
                    cv2.line(img, (x_top, y_top), (x_base, y_base), (255, 255, 255), 3)
            cv2.circle(img, (x_top, y_top), 10, (0, 255, 0), -1)
            cv2.circle(img, (x_top, y_top), 13, (255, 255, 255), 3)


def draw_trajectory_ribbon(img, waypoints_ego, cam_config, 
                           color=(0, 200, 0), path_width=0.9, alpha=0.6,
                           route_pts_ego=None, target_pt_ego=None, plan_ego_traj=None):
    """
    Draw a ribbon-style trajectory overlay on the camera image.
    Uses 3D projection for accurate perspective, with bottom anchored at image center.
    """
    H, W = img.shape[:2]
    
    # Draw route dots and target first (always, regardless of ribbon validity)
    # This ensures navigation info is visible even when plan trajectory is too short
    _draw_route_and_target(img, cam_config, route_pts_ego, target_pt_ego)
    
    if len(waypoints_ego) < 2:
        return img
    
    # Generate left/right offset waypoints for ribbon
    offsets_left, offsets_right = [], []
    
    for i in range(len(waypoints_ego) - 1):
        p1, p2 = waypoints_ego[i], waypoints_ego[i + 1]
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        dir_unit = direction / norm
        normal = np.array([-dir_unit[1], dir_unit[0]])
        offsets_left.append(p1 + path_width * normal)
        offsets_right.append(p1 - path_width * normal)
    
    if len(waypoints_ego) > 1:
        direction = waypoints_ego[-1] - waypoints_ego[-2]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            dir_unit = direction / norm
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left.append(waypoints_ego[-1] + path_width * normal)
            offsets_right.append(waypoints_ego[-1] - path_width * normal)
    
    if len(offsets_left) < 2:
        return img
    
    offsets_left = np.array(offsets_left)
    offsets_right = np.array(offsets_right)
    
    # Project all points
    pts_left, valid_l = project_ego_waypoints_to_camera(offsets_left, cam_config, img.shape)
    pts_right, valid_r = project_ego_waypoints_to_camera(offsets_right, cam_config, img.shape)
    
    valid = valid_l & valid_r
    if not np.any(valid):
        return img
    
    pts_left = pts_left[valid]
    pts_right = pts_right[valid]
    
    # Always anchor at bottom center of image (ego position)
    anchor_left = np.array([[W / 2 - 40, H - 1]])
    anchor_right = np.array([[W / 2 + 40, H - 1]])
    pts_left = np.vstack([anchor_left, pts_left])
    pts_right = np.vstack([anchor_right, pts_right])
    
    # Clip to image
    pts_left[:, 0] = np.clip(pts_left[:, 0], 0, W - 1)
    pts_left[:, 1] = np.clip(pts_left[:, 1], 0, H - 1)
    pts_right[:, 0] = np.clip(pts_right[:, 0], 0, W - 1)
    pts_right[:, 1] = np.clip(pts_right[:, 1], 0, H - 1)
    
    polygon = np.vstack([pts_left, pts_right[::-1]]).astype(np.int32)
    
    # Draw with gradient transparency
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon], color=color)
    
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    
    gradient_mask = np.zeros((H, W), dtype=np.float32)
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) > 0:
        normalized_y = y_coords / H
        alpha_values = 0.15 + (alpha - 0.15) * (normalized_y ** 1.2)
        gradient_mask[y_coords, x_coords] = alpha_values
    
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1 - gradient_mask) + 
                        overlay[:, :, c] * gradient_mask).astype(np.uint8)
    
    # Edge lines
    edge_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
    if len(pts_left) > 1:
        cv2.polylines(img, [pts_left.astype(np.int32)], False, edge_color, 2, cv2.LINE_AA)
    if len(pts_right) > 1:
        cv2.polylines(img, [pts_right.astype(np.int32)], False, edge_color, 2, cv2.LINE_AA)
    
    # Route dots and target already drawn by _draw_route_and_target() at start
    
    # 3. Plan(E) ego-frame trajectory (MAGENTA ribbon, matching topdown)
    if plan_ego_traj is not None and len(plan_ego_traj) > 1:
        # Draw a thinner semi-transparent magenta ribbon for Plan(E)
        offsets_left_e, offsets_right_e = [], []
        pw = path_width * 0.7  # Thinner than Plan(W)
        for i in range(len(plan_ego_traj) - 1):
            p1, p2 = plan_ego_traj[i], plan_ego_traj[i + 1]
            direction = p2 - p1
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            dir_unit = direction / norm
            normal = np.array([-dir_unit[1], dir_unit[0]])
            offsets_left_e.append(p1 + pw * normal)
            offsets_right_e.append(p1 - pw * normal)
        
        if len(plan_ego_traj) > 1:
            direction = plan_ego_traj[-1] - plan_ego_traj[-2]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                dir_unit = direction / norm
                normal = np.array([-dir_unit[1], dir_unit[0]])
                offsets_left_e.append(plan_ego_traj[-1] + pw * normal)
                offsets_right_e.append(plan_ego_traj[-1] - pw * normal)
        
        if len(offsets_left_e) >= 2:
            offsets_left_e = np.array(offsets_left_e)
            offsets_right_e = np.array(offsets_right_e)
            
            pts_left_e, valid_l_e = project_ego_waypoints_to_camera(offsets_left_e, cam_config, img.shape)
            pts_right_e, valid_r_e = project_ego_waypoints_to_camera(offsets_right_e, cam_config, img.shape)
            
            valid_e = valid_l_e & valid_r_e
            if np.any(valid_e):
                pts_left_e = pts_left_e[valid_e]
                pts_right_e = pts_right_e[valid_e]
                
                anchor_left_e = np.array([[W / 2 - 30, H - 1]])
                anchor_right_e = np.array([[W / 2 + 30, H - 1]])
                pts_left_e = np.vstack([anchor_left_e, pts_left_e])
                pts_right_e = np.vstack([anchor_right_e, pts_right_e])
                
                pts_left_e[:, 0] = np.clip(pts_left_e[:, 0], 0, W - 1)
                pts_left_e[:, 1] = np.clip(pts_left_e[:, 1], 0, H - 1)
                pts_right_e[:, 0] = np.clip(pts_right_e[:, 0], 0, W - 1)
                pts_right_e[:, 1] = np.clip(pts_right_e[:, 1], 0, H - 1)
                
                polygon_e = np.vstack([pts_left_e, pts_right_e[::-1]]).astype(np.int32)
                
                # MAGENTA with lower alpha
                overlay_e = img.copy()
                cv2.fillPoly(overlay_e, [polygon_e], color=(255, 0, 255))
                img = cv2.addWeighted(overlay_e, 0.25, img, 0.75, 0)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory on front camera")
    parser.add_argument("scenario_pkl", type=str, help="Path to scenario .pkl file")
    parser.add_argument("--frame", type=int, default=0, help="Frame to visualize (default: 0)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--replan-rate", type=int, default=10)
    parser.add_argument("--ego-replay-frames", type=int, default=20)
    args = parser.parse_args()
    
    # Add paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, 'nuplan-devkit'))
    os.chdir(os.path.join(repo_root, 'bridgesim', 'evaluation'))
    
    from bridgesim.evaluation.core.base_evaluator import BaseEvaluator
    from bridgesim.evaluation.models.transfuser_adapter import TransfuserAdapter
    
    ckpt = os.path.join(repo_root, 'ckpts/BridgeSim/navsimv2/transfuser/transfuser_seed_0.ckpt')
    
    adapter = TransfuserAdapter(checkpoint_path=ckpt)
    evaluator = BaseEvaluator(
        model_adapter=adapter,
        scenario_path=args.scenario_pkl,
        output_dir='/tmp/vis_traj_test',
        traffic_mode='log_replay', enable_vis=True, save_perframe=True,
        eval_mode='closed_loop', controller_type='pure_pursuit',
        replan_rate=args.replan_rate, ego_replay_frames=args.ego_replay_frames,
        scorer_type='navsim',
    )
    
    # Run evaluation for just the target frame + a few more
    evaluator.run()
    
    # Now overlay trajectory on the saved cam_f0 image
    target_frame = args.frame
    cam_path = f'/tmp/vis_traj_test/transfuser_rr{args.replan_rate}_erf{args.ego_replay_frames}_efNone/{evaluator.scenario_name}/{target_frame:05d}/cam_f0.jpg'
    
    if not os.path.exists(cam_path):
        print(f"Camera image not found: {cam_path}")
        # Try to find any frame
        import glob
        frames = sorted(glob.glob(f'/tmp/vis_traj_test/transfuser_rr{args.replan_rate}_erf{args.ego_replay_frames}_efNone/{evaluator.scenario_name}/*/cam_f0.jpg'))
        if frames:
            cam_path = frames[0]
            print(f"Using: {cam_path}")
        else:
            print("No camera images found!")
            return
    
    print(f"\nOverlaying trajectory on: {cam_path}")
    img = cv2.imread(cam_path)
    
    # Get camera config for front camera
    cam_config = adapter.get_camera_configs()['CAM_F0']
    
    # Create a sample forward trajectory for testing
    # In ego frame: lateral=[:,0], forward=[:,1]
    N = 40
    t = np.linspace(0, 4, N)  # 4 seconds ahead
    test_traj = np.zeros((N, 2))
    test_traj[:, 1] = t * 4.0  # Forward at ~4 m/s (straight ahead)
    # test_traj[:, 0] = 0.5 * np.sin(t)  # Slight lateral movement for testing
    
    print(f"Test trajectory: {N} points, max forward={test_traj[-1,1]:.1f}m")
    
    img_with_traj = draw_trajectory_ribbon(
        img.copy(), test_traj, cam_config,
        color=(0, 200, 0), path_width=0.8, alpha=0.6
    )
    
    output_path = args.output or f'/tmp/vis_traj_test/frame{target_frame:05d}_with_traj.jpg'
    cv2.imwrite(output_path, img_with_traj)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

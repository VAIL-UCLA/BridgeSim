#!/usr/bin/env python
"""
Script to capture sensor images from MetaDrive ego vehicle in a scenario with traffic.
This demonstrates RGB, depth, semantic, instance cameras and LiDAR sensors.
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.component.sensors.instance_camera import InstanceCamera
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar

def save_rgb_image(image, filename):
    """Save RGB image to file"""
    # Convert from MetaDrive format (H, W, C) to standard BGR for OpenCV
    if len(image.shape) == 4:
        image = image[..., -1]  # Take the last frame from stack
    
    # If normalized to [0,1], convert back to [0,255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved RGB image: {filename}")

def save_depth_image(image, filename):
    """Save depth image with colormap"""
    if len(image.shape) == 4:
        image = image[..., -1]  # Take the last frame from stack
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image[..., 0]  # Remove single channel dimension
    
    # Convert depth to uint8 for visualization
    depth_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    cv2.imwrite(filename, depth_colored)
    print(f"Saved depth image: {filename}")

def save_semantic_image(image, filename):
    """Save semantic segmentation image"""
    if len(image.shape) == 4:
        image = image[..., -1]  # Take the last frame from stack
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved semantic image: {filename}")

def save_instance_image(image, filename):
    """Save instance segmentation image"""
    if len(image.shape) == 4:
        image = image[..., -1]  # Take the last frame from stack
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved instance image: {filename}")

def visualize_lidar(points, filename):
    """Visualize and save PointCloud LiDAR data as image"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    if hasattr(points, 'shape'):
        print(f"LiDAR data shape: {points.shape}")
        
        # PointCloudLidar returns 3D coordinates in shape (H, W, 3) or similar
        if len(points.shape) == 3 and points.shape[-1] == 3:
            # Reshape to get all points as N x 3
            points_flat = points.reshape(-1, 3)
            x = points_flat[:, 0]
            y = points_flat[:, 1]
            z = points_flat[:, 2]
            
            # Filter out invalid points (zeros or very distant)
            valid_mask = (np.abs(x) > 0.1) | (np.abs(y) > 0.1) | (np.abs(z) > 0.1)
            valid_mask &= (np.abs(x) < 100) & (np.abs(y) < 100) & (np.abs(z) < 100)
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            
            if len(x_valid) > 0:
                # Create top-down scatter plot colored by height
                scatter = ax.scatter(x_valid, y_valid, c=z_valid, s=1, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='Height (z) [m]')
                ax.set_xlim([x_valid.min()-1, x_valid.max()+1])
                ax.set_ylim([y_valid.min()-1, y_valid.max()+1])
                print(f"Plotted {len(x_valid)} valid LiDAR points")
            else:
                ax.text(0.5, 0.5, 'No valid LiDAR points found', transform=ax.transAxes, ha='center')
                
        elif len(points.shape) == 2 and points.shape[-1] == 3:
            # Already flattened N x 3 format
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            
            # Filter out invalid points
            valid_mask = (np.abs(x) > 0.1) | (np.abs(y) > 0.1) | (np.abs(z) > 0.1)
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            z_valid = z[valid_mask]
            
            if len(x_valid) > 0:
                scatter = ax.scatter(x_valid, y_valid, c=z_valid, s=1, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax, label='Height (z) [m]')
                print(f"Plotted {len(x_valid)} valid LiDAR points")
            else:
                ax.text(0.5, 0.5, 'No valid LiDAR points found', transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, 0.5, f'Unexpected LiDAR shape: {points.shape}', transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, f'LiDAR data type: {type(points)}', transform=ax.transAxes, ha='center')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('PointCloud LiDAR (Top-down view)')
    ax.set_aspect('equal')
    ax.grid(True)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved LiDAR visualization: {filename}")

def main():
    """Main function to capture sensor images"""
    # Create output directory
    output_dir = "sensor_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration for environment with multiple sensors
    config = dict(
        num_scenarios=100,  # Multiple scenarios available
        traffic_density=1.0,  # Maximum traffic density for crowded conditions
        use_render=False,  # No need for visual rendering
        start_seed=42,  # Fixed seed for reproducible results
        map=4,  # Use a more complex map with multiple lanes (highway-like)
        image_observation=True,  # Enable image observations
        
        # Make traffic more aggressive and closer
        traffic_mode="respawn",  # Respawn traffic to keep density high
        accident_prob=0.0,  # No accidents to avoid stopping traffic
        
        # Additional traffic configurations for closer proximity
        random_traffic=False,  # Disable randomness for more predictable dense traffic
        horizon=1000,  # Longer episodes
        out_of_route_done=False,  # Don't end episode if going off route
        
        # Vehicle configuration
        vehicle_config=dict(
            image_source="rgb_camera",  # Set main image source
            lidar=dict(num_lasers=240, distance=50, num_others=4)  # Configure default lidar to avoid conflicts
        ),
        
        # Configure multiple sensors
        sensors={
            "rgb_camera": (RGBCamera, 400, 300),
            "depth_camera": (DepthCamera, 400, 300), 
            "semantic_camera": (SemanticCamera, 400, 300),
            "instance_camera": (InstanceCamera, 400, 300),
            "point_cloud_lidar": (PointCloudLidar, 200, 64, True)  # width, height, return_coordinates
        }
    )
    
    print("Creating MetaDrive environment with multiple sensors...")
    env = MetaDriveEnv(config)
    
    try:
        print("Starting simulation...")
        obs, _ = env.reset()
        
        # Run for more steps to get into dense traffic
        for step in range(20):
            # Use more aggressive driving to get closer to other vehicles
            if step < 10:
                action = [0.8, 0.0]  # More aggressive acceleration
            elif step < 15:
                action = [0.3, 0.0]  # Moderate speed to blend with traffic
            else:
                action = [0.1, 0.0]  # Slow down to get closer to vehicles ahead
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Capture sensors when we should be in dense traffic
            if step == 15:  # Capture later when we're more embedded in traffic
                print(f"\nCapturing sensor data at step {step}...")
                
                # Get sensor data directly from the sensors
                # RGB Camera
                if "rgb_camera" in env.engine.sensors:
                    rgb_sensor = env.engine.get_sensor("rgb_camera")
                    rgb_image = rgb_sensor.get_rgb_array_cpu()
                    save_rgb_image(rgb_image, os.path.join(output_dir, "rgb_camera.png"))
                
                # Depth Camera
                if "depth_camera" in env.engine.sensors:
                    depth_sensor = env.engine.get_sensor("depth_camera")
                    depth_image = depth_sensor.get_rgb_array_cpu()  # For depth, this returns depth values
                    save_depth_image(depth_image, os.path.join(output_dir, "depth_camera.png"))
                
                # Semantic Camera
                if "semantic_camera" in env.engine.sensors:
                    semantic_sensor = env.engine.get_sensor("semantic_camera")
                    semantic_image = semantic_sensor.get_rgb_array_cpu()
                    save_semantic_image(semantic_image, os.path.join(output_dir, "semantic_camera.png"))
                
                # Instance Camera
                if "instance_camera" in env.engine.sensors:
                    instance_sensor = env.engine.get_sensor("instance_camera")
                    instance_image = instance_sensor.get_rgb_array_cpu()
                    save_instance_image(instance_image, os.path.join(output_dir, "instance_camera.png"))
                
                # PointCloudLiDAR - get 3D point cloud data
                if "point_cloud_lidar" in env.engine.sensors:
                    lidar_sensor = env.engine.get_sensor("point_cloud_lidar")
                    points = lidar_sensor.get_rgb_array_cpu()  # This returns 3D coordinates
                    visualize_lidar(points, os.path.join(output_dir, "lidar_visualization.png"))
                
                print("\nAlso capturing from observation (main camera)...")
                # Also capture the main observation image if available
                if "image" in obs:
                    obs_image = obs["image"]
                    if len(obs_image.shape) == 4:
                        obs_image = obs_image[..., -1]  # Take last stack frame
                    save_rgb_image(obs_image, os.path.join(output_dir, "main_observation.png"))
                
                break
            
            # Reset if episode ends
            if terminated or truncated:
                obs, _ = env.reset()
        
        print(f"\nSensor images saved to '{output_dir}' directory")
        print("Available sensor data:")
        print("- RGB Camera: Standard color image")
        print("- Depth Camera: Distance information with color coding")
        print("- Semantic Camera: Semantic segmentation of objects")
        print("- Instance Camera: Instance segmentation for object detection")
        print("- LiDAR: Top-down point cloud visualization")
        print("- Main Observation: The main camera feed used by the agent")
        
    finally:
        env.close()

if __name__ == "__main__":
    main()
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.env_input_policy import EnvInputPolicy

# Import the controllers provided
from controller_md import PIDController, PurePursuitController

def get_future_trajectory(current_frame, gt_positions, steps=6, stride=5):
    """
    Extracts the next 'steps' points, sampled every 'stride' frames.
    
    Args:
        current_frame: Current index in GT.
        steps: Number of waypoints to retrieve (default 6 for UniAD).
        stride: Sampling rate (default 5 for 10Hz -> 0.5s spacing).
    """
    future_points = []
    total_frames = len(gt_positions)
    
    for i in range(1, steps + 1):
        idx = current_frame + (i * stride)
        if idx < total_frames:
            future_points.append(gt_positions[idx])
        else:
            # Pad with the last available point if we run off the end
            future_points.append(gt_positions[-1])
            
    return np.array(future_points)

def world_to_ego(world_points, ego_pos, ego_heading):
    """
    Transforms points from World Frame to Ego Frame (Y-Forward).
    """
    # 1. Translate
    diff = world_points - ego_pos
    dx = diff[:, 0]
    dy = diff[:, 1]
    
    # 2. Rotate
    # Using the offline_evaluator_md.py logic where:
    # Ego Forward is +Y
    # Ego Lateral is +X
    
    c = np.cos(ego_heading)
    s = np.sin(ego_heading)
    
    # Rotation matching the reference implementation
    # cos_h = cos(heading), sin_h = -sin(heading) in reference
    cos_h = c
    sin_h = -s
    
    ego_x = sin_h * dx + cos_h * dy
    ego_y = cos_h * dx - sin_h * dy
    
    return np.stack((ego_x, ego_y), axis=1)

def main(args):
    # 1. Setup
    scenario_path = Path(args.scenario_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading scenario from: {scenario_path}")
    scenario_name = scenario_path.name
    scenario_subfolder = scenario_path / f"{scenario_name}_0"
    pkl_files = list(scenario_subfolder.glob("*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl file found in {scenario_subfolder}")
    
    with open(pkl_files[0], 'rb') as f:
        try:
            import pickle5
            scenario_data = pickle5.load(f)
        except ImportError:
            scenario_data = pickle.load(f)

    sdc_id = scenario_data['metadata']['sdc_id']
    gt_track = scenario_data['tracks'][sdc_id]
    gt_positions = gt_track['state']['position'][:, :2]
    gt_headings = gt_track['state']['heading']
    scenario_len = scenario_data['length']

    # 2. Config
    config = {
        "use_render": False,
        "agent_policy": EnvInputPolicy,
        "reactive_traffic": False, 
        "num_scenarios": 1,
        "horizon": scenario_len + 100,
        "data_directory": str(scenario_path.absolute()),
        "vehicle_config": {
            "no_wheel_friction": False 
        },
        "window_size": (100, 100),
        "show_interface": False,
        "show_logo": False,
        "show_fps": False,
    }

    env = ScenarioEnv(config)
    
    try:
        env.reset(seed=0)

        # Select controller based on argument
        if args.controller == 'pid':
            controller = PIDController()
            print("Using PIDController")
        else:
            controller = PurePursuitController()
            print("Using PurePursuitController")
        
        # Offsets
        gt_pos_0 = gt_positions[0]
        gt_heading_0 = gt_headings[0]
        sim_pos_0 = env.agent.position
        sim_heading_0 = env.agent.heading_theta
        
        position_offset = gt_pos_0 - sim_pos_0
        heading_offset = gt_heading_0 - sim_heading_0
        
        history_sim_pos = []
        history_gt_pos = []
        errors = []

        # 3. Simulation
        # We run for the length of the scenario
        for t in tqdm(range(scenario_len - 1), desc="Simulating"):
            
            # Current State (Sim -> World)
            current_sim_pos = env.agent.position
            current_sim_heading = env.agent.heading_theta
            
            # Use Norm(Velocity) for reliable m/s speed
            current_sim_speed_ms = np.linalg.norm(env.agent.velocity)
            
            current_world_pos = current_sim_pos + position_offset
            current_world_heading = current_sim_heading + heading_offset
            
            history_sim_pos.append(current_world_pos)
            history_gt_pos.append(gt_positions[t])
            
            # Get Trajectory (Downsampled stride=5 for 0.5s spacing)
            future_traj_world = get_future_trajectory(t, gt_positions, steps=40, stride=1)
            
            # Transform to Ego Frame
            trajectory_ego = world_to_ego(
                future_traj_world, 
                current_world_pos, 
                current_world_heading
            )
            
            # Target for aiming (Last point)
            target_ego = trajectory_ego[-1]
            
            # Control
            steer, throttle, brake, _ = controller.control_pid(
                trajectory_ego, 
                current_sim_speed_ms, 
                target_ego
            )
            
            # MetaDrive Action
            action = [steer, float(throttle - brake)]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Error Calc
            dist_error = np.linalg.norm(current_world_pos - gt_positions[t])
            errors.append(dist_error)
            
            if terminated or truncated:
                print(f"Terminated early at frame {t}")
                break

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()

    # 4. Results
    history_sim_pos = np.array(history_sim_pos)
    history_gt_pos = np.array(history_gt_pos)
    errors = np.array(errors)
    
    ade = np.mean(errors) if len(errors) > 0 else 0.0
    
    print("-" * 40)
    print(f"RESULTS for {scenario_name}")
    print(f"Average Displacement Error (ADE): {ade:.4f} m")
    print("-" * 40)
    
    # Plot
    plt.figure(figsize=(12, 12))
    
    # Plot GT
    plt.plot(history_gt_pos[:, 0], history_gt_pos[:, 1], 'b--', label='Ground Truth', linewidth=2, alpha=0.7)
    
    # Plot Sim
    if len(history_sim_pos) > 0:
        plt.plot(history_sim_pos[:, 0], history_sim_pos[:, 1], 'r-', label='Controller (Sim)', linewidth=1)
    
    # Markers
    plt.scatter(history_gt_pos[0,0], history_gt_pos[0,1], c='g', marker='^', s=100, label='Start')
    plt.scatter(history_gt_pos[-1,0], history_gt_pos[-1,1], c='k', marker='x', s=100, label='End')
    
    plt.title(f"Controller Tracking Performance\nScenario: {scenario_name} | ADE: {ade:.3f}m")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plot_path = output_dir / f"tracking_result_{scenario_name}.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario-path', type=str, required=True,
                        help="Path to the MetaDrive scenario data folder")
    parser.add_argument('--output-dir', type=str, default='./controller_test_output',
                        help="Directory to save the results")
    parser.add_argument('--controller', type=str, default='pid', choices=['pid', 'pure_pursuit'],
                        help="Controller type: 'pid' or 'pure_pursuit' (default: pid)")

    args = parser.parse_args()
    main(args)
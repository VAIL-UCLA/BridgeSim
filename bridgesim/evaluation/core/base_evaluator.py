"""
Base evaluator with core evaluation pipeline.
This is where cross-cutting features (like flow matching) should be added.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import deque
from typing import Dict, Any
from datetime import datetime

# Add repository root to path for imports (evaluation/ is at top level)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.parse_object_state import parse_object_state

from bridgesim.evaluation.utils import PIDController, PurePursuitController, OfflineStatisticsManager
from bridgesim.evaluation.utils.epdms_scorer_md import EPDMSScorer
from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.core.environment_manager import EnvironmentManager
from bridgesim.evaluation.utils.constants import VOID, LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT


class BaseEvaluator:
    """
    Core evaluation pipeline for autonomous driving models.

    This class handles:
    - Scenario loading
    - Environment management
    - Sensor perception
    - Route generation
    - Control and statistics
    - Main evaluation loop

    To add cross-cutting features (like flow matching), modify this class.
    """

    def __init__(self,
                 model_adapter: BaseModelAdapter,
                 scenario_path: str,
                 output_dir: str,
                 traffic_mode: str = "log_replay",
                 enable_vis: bool = False,
                 save_perframe: bool = True,
                 eval_mode: str = "closed_loop",
                 controller_type: str = "pure_pursuit",
                 replan_rate: int = 1,
                 sim_dt: float = 0.1,
                 ego_replay_frames: int = 0,
                 eval_frames: int = None,
                 scorer_type: str = "legacy",
                 score_start_frame: int = None,
                 ):
        """
        Initialize evaluator.

        Args:
            model_adapter: Model adapter instance
            scenario_path: Path to converted scenario
            output_dir: Output directory for results
            traffic_mode: Traffic mode (no_traffic, log_replay, IDM)
            enable_vis: Enable visualization outputs
            save_perframe: Save per-frame outputs (planning_traj.npy, etc.)
            eval_mode: 'closed_loop' or 'open_loop'
            controller_type: Controller type ('pid' or 'pure_pursuit')
            replan_rate: How often to run model inference (1=every frame, N=every N frames)
            sim_dt: Simulation timestep in seconds (default: 0.1s for 10Hz)
            ego_replay_frames: Number of initial frames to replay ego log actions (inference still runs)
            eval_frames: Number of frames to evaluate after ego replay (None = full scenario)
            scorer_type: Scorer type for closed_loop ('legacy' or 'navsim')
            score_start_frame: Frame to start calculating scores (None = use ego_replay_frames)
        """
        self.model_adapter = model_adapter
        self.sim_dt = sim_dt
        self.scenario_path = Path(scenario_path)
        self.scenario_name = self.scenario_path.name
        self.output_dir = Path(output_dir) / self.scenario_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.traffic_mode = traffic_mode
        self.enable_vis = enable_vis
        self.save_perframe = save_perframe
        self.eval_mode = eval_mode
        self.controller_type = controller_type
        self.replan_rate = replan_rate
        self.ego_replay_frames = max(0, ego_replay_frames)
        self.eval_frames = eval_frames
        self.scorer_type = scorer_type
        # Score start frame: defaults to ego_replay_frames if not specified
        self.score_start_frame = score_start_frame if score_start_frame is not None else self.ego_replay_frames

        assert replan_rate 

        # Environment and scenario data
        self.env_manager = None
        self.scenario_data = None
        self.sdc_id = None
        self.route = None

        # Control and statistics
        self.controller = None
        self.stats_manager = None
        self.epdms_scorer = None
        self.epdms_results_openloop = []
        self.epdms_results_closedloop = []
        self.route_completion_baseline = None  # Baseline for route completion at score_start_frame
        self.expected_rc_delta = None  # Expected route completion delta from log (for eval_frames normalization)

        # Coordinate offsets (simulation space -> world space)
        self.position_offset = np.zeros(3)
        self.heading_offset = 0.0

        # Termination flag
        self.has_terminated = False

        # Replan caching (for replan_rate > 1)
        self.cached_plan_traj_world = None  # Cached trajectory in world coordinates
        self.cached_model_output = None  # Cached model output for visualization
        self.last_replan_frame = -1  # Track when last replan happened
        self.cached_prediction_position = None  # Ego position at prediction time (for candidate visualization)
        self.cached_prediction_heading = None  # Ego heading at prediction time (for candidate visualization)
        self.ego_log_actions = []
        self.warned_missing_ego_actions = False

        # Planning history for replan summary visualization (stores ALL past plans)
        # Each entry: (frame_id, world_trajectory, ego_position, ego_heading, consumed_waypoints, topk_world)
        # topk_world: top-k candidate trajectories in world coordinates (for coverage visualization)
        self.planning_history = []

        print(f"Initialized evaluator for {self.scenario_name}")
        print(f"  Mode: {self.eval_mode}")
        print(f"  Traffic mode: {self.traffic_mode}")
        print(f"  Visualization: {self.enable_vis}")
        print(f"  Save perframe: {self.save_perframe}")
        print(f"  Replan rate: {self.replan_rate}")
        if self.ego_replay_frames > 0:
            print(f"  Ego log replay frames: {self.ego_replay_frames}")
        if self.eval_frames is not None:
            print(f"  Eval frames (after replay): {self.eval_frames}")
        print(f"  Score start frame: {self.score_start_frame}")
        if self.eval_mode == "closed_loop":
            print(f"  Scorer type: {self.scorer_type}")

    def prepare_ego_replay_actions(self):
        """
        Pre-load ego log states for replay during warmup frames.
        Uses parse_object_state to extract position, velocity, heading, etc.
        """
        if self.ego_replay_frames <= 0:
            return

        ego_track = self.scenario_data['tracks'][self.sdc_id]
        num_frames = len(ego_track['state']['position'])

        for i in range(min(self.ego_replay_frames, num_frames)):
            state = parse_object_state(ego_track, i)
            if state and state.get('valid', False):
                self.ego_log_actions.append(state)
            else:
                self.ego_log_actions.append(None)

        print(f"Prepared {len(self.ego_log_actions)} ego log states for replay")

    def _maybe_override_with_log_action(self, control, frame_id, env):
        """
        Override ego state with logged trajectory during replay frames.

        During replay frames (frame_id < ego_replay_frames):
        - Sets ego position, heading, velocity, and angular_velocity from log
        - Returns neutral control [0, 0] to minimize physics interference
        - This matches MetaDrive's ReplayTrafficParticipantPolicy behavior

        When replay ends (frame_id == ego_replay_frames):
        - Logs the transition to model control
        - Replan forcing is handled by process_frame()

        Args:
            control: The control computed by the model/controller
            frame_id: Current frame index
            env: The MetaDrive environment

        Returns:
            control (neutral during replay, original otherwise)
        """
        # Check if we're still in replay phase
        if frame_id < self.ego_replay_frames:
            if frame_id < len(self.ego_log_actions):
                log_state = self.ego_log_actions[frame_id]
                if log_state is not None and log_state.get('valid', False):
                    ego_vehicle = env.agent

                    # Set position, heading, velocity, and angular velocity
                    # (same as MetaDrive's ReplayTrafficParticipantPolicy)
                    ego_vehicle.set_position(log_state['position'])
                    ego_vehicle.set_heading_theta(log_state['heading_theta'])
                    ego_vehicle.set_velocity(log_state['velocity'], in_local_frame=True)
                    if 'angular_velocity' in log_state:
                        ego_vehicle.set_angular_velocity(log_state['angular_velocity'])

                    # Return neutral control to minimize physics interference
                    return np.array([0.0, 0.0])
            elif not self.warned_missing_ego_actions:
                print(f"[WARNING] No logged ego action for frame {frame_id}")
                self.warned_missing_ego_actions = True

        # Log transition from replay to model control
        if frame_id == self.ego_replay_frames:
            print(f"[INFO] Ego log replay ended at frame {frame_id}. Switching to model control.")

        return control

    def _interpolate_trajectory(self, trajectory: np.ndarray, model_dt: float, sim_dt: float) -> np.ndarray:
        """
        Interpolate trajectory from model_dt intervals to sim_dt intervals.

        Model predicts waypoints at model_dt intervals (e.g., 0.5s).
        Simulation runs at sim_dt intervals (e.g., 0.1s).
        This method resamples the trajectory to match simulation timestep.

        The trajectory is in ego frame where ego is at (0, 0) at time 0.
        We prepend (0, 0) as the starting point for smooth interpolation.

        Args:
            trajectory: (N, 2) waypoints at model_dt intervals in ego frame
                       waypoint[0] is at time model_dt, waypoint[1] at 2*model_dt, etc.
            model_dt: Time between model waypoints (e.g., 0.5s)
            sim_dt: Simulation timestep (e.g., 0.1s)

        Returns:
            Interpolated (M, 2) trajectory at sim_dt intervals, starting at sim_dt
        """
        if model_dt <= sim_dt:
            return trajectory  # No interpolation needed

        n_original = len(trajectory)
        if n_original == 0:
            return trajectory

        # Prepend current ego position (0, 0) at time 0
        ego_origin = np.array([[0.0, 0.0]])
        trajectory_with_origin = np.vstack([ego_origin, trajectory])

        # Time values: 0, model_dt, 2*model_dt, ...
        t_original = np.arange(0, n_original + 1) * model_dt

        # Time values for interpolated waypoints: sim_dt, 2*sim_dt, ...
        # (starting at sim_dt since we want future waypoints, not current position)
        t_max = t_original[-1]
        t_interp = np.arange(sim_dt, t_max + sim_dt / 2, sim_dt)

        # Interpolate x and y separately
        interp_x = np.interp(t_interp, t_original, trajectory_with_origin[:, 0])
        interp_y = np.interp(t_interp, t_original, trajectory_with_origin[:, 1])

        return np.stack([interp_x, interp_y], axis=1)

    def load_scenario(self):
        """Load scenario data from pickle file."""
        
        scenario_subfolder = self.scenario_path / f"{self.scenario_name}_0"
        scenario_pkl_files = list(scenario_subfolder.glob("*.pkl"))

        if not scenario_pkl_files:
            raise FileNotFoundError(f"No .pkl file found in {scenario_subfolder}")

        scenario_pkl_path = scenario_pkl_files[0]
        print(f"Loading scenario from: {scenario_pkl_path}")

        with open(scenario_pkl_path, 'rb') as f:
            try:
                import pickle5
                self.scenario_data = pickle5.load(f)
            except:
                import pickle
                self.scenario_data = pickle.load(f)

        self.sdc_id = self.scenario_data['metadata']['sdc_id']
        print(f"SDC (ego) ID: {self.sdc_id}")

    def generate_route(self):
        """Generate waypoints and commands from ground-truth trajectory."""
        waypoint_spacing = 1.0
        turn_threshold_deg = 3.0 # prev: 5

        waypoints = []
        self.route = deque()
        last_pos = None

        # Extract ground truth trajectory
        ego_track = self.scenario_data['tracks'][self.sdc_id]
        positions = ego_track['state']['position']
        headings = ego_track['state']['heading']

        # Subsample trajectory to reasonable waypoint spacing
        for i in range(len(positions)):
            pos = positions[i]
            if last_pos is None or np.linalg.norm(pos[:2] - last_pos[:2]) >= waypoint_spacing:
                waypoints.append({
                    'position': pos[:2].copy(),
                    'heading': headings[i],
                    'frame_idx': i
                })
                last_pos = pos[:2]

        print(f"Subsampled to {len(waypoints)} waypoints (spacing={waypoint_spacing}m)")

        # Generate commands based on heading changes
        for i in range(len(waypoints)):
            wp = waypoints[i]
            if i < len(waypoints) - 1:
                current_heading = wp['heading']
                next_heading = waypoints[i + 1]['heading']
                heading_diff = np.rad2deg(next_heading - current_heading)

                # Normalize to [-180, 180]
                while heading_diff > 180:
                    heading_diff -= 360
                while heading_diff < -180:
                    heading_diff += 360

                # Classify command
                if abs(heading_diff) < turn_threshold_deg:
                    command = STRAIGHT
                elif heading_diff > turn_threshold_deg:
                    command = LEFT
                elif heading_diff < -turn_threshold_deg:
                    command = RIGHT
                else:
                    command = LANEFOLLOW
            else:
                command = LANEFOLLOW

            self.route.append((wp['position'], command, wp['frame_idx']))

        print(f"Generated {len(self.route)} route commands")

    def get_next_waypoint(self, current_position):
        """Get next navigation waypoint and command."""
        min_distance = 4.0
        max_distance = 50.0

        if len(self.route) == 1:
            return self.route[0]

        # Find waypoints that have been reached
        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - current_position)

            if distance <= min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        # Pop reached waypoints
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        # Return next waypoint
        next_waypoint = self.route[1] if len(self.route) > 1 else self.route[0]
        waypoint_pos, road_option, frame_idx = next_waypoint
        command = road_option - 1  # Convert to 0-indexed

        return waypoint_pos, command, frame_idx

    def perceive(self, env, frame_id):
        """Capture images from all cameras."""
        imgs = {}
        sensor = env.engine.get_sensor('rgb_camera')
        camera_configs = self.model_adapter.get_camera_configs()

        # Save images if visualization is enabled
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        if self.enable_vis:
            frame_output_path.mkdir(parents=True, exist_ok=True)

        for name, cam_config in camera_configs.items():
            sensor.lens.setFov(cam_config['fov'])

            try:
                sensor_output = sensor.perceive(
                    to_float=False,
                    new_parent_node=env.agent.origin,
                    position=(cam_config['y'], cam_config['x'], cam_config['z']),
                    hpr=(-cam_config['yaw'], cam_config['pitch'], -cam_config['roll'])
                )

                # Handle both CUDA tensors (with .get()) and numpy arrays
                if hasattr(sensor_output, 'get'):
                    sensor_data = sensor_output.get()
                else:
                    sensor_data = sensor_output

                imgs[name] = sensor_data

                if self.enable_vis:
                    cv2.imwrite(str(frame_output_path / f"{name.lower()}.jpg"), sensor_data)

            except Exception as e:
                print(f"Error capturing {name}: {e}")
                raise

        return imgs

    def compute_ego_state(self, env, frame_id, prev_velocity, prev_heading):
        """Compute ego vehicle state in world coordinates."""
        ego_vehicle = env.agent

        # Get state in simulation space
        ego_position_sim = np.array(ego_vehicle.position)
        if len(ego_position_sim) == 2:
            ego_position_sim = np.array([ego_position_sim[0], ego_position_sim[1], 0.0])
        ego_heading_sim = ego_vehicle.heading_theta

        # Initialize coordinate offsets at frame 0
        if frame_id == 0:
            world_pos_0 = self.scenario_data['tracks'][self.sdc_id]['state']['position'][0]
            world_heading_0 = self.scenario_data['tracks'][self.sdc_id]['state']['heading'][0]
            self.position_offset = world_pos_0 - ego_position_sim
            self.heading_offset = world_heading_0 - ego_heading_sim

            print(f"Initialized coordinate offsets:")
            print(f"  Position offset: {self.position_offset}")
            print(f"  Heading offset: {self.heading_offset:.4f} rad")

        # Transform to world space
        ego_position = ego_position_sim + self.position_offset
        ego_heading = ego_heading_sim + self.heading_offset

        # Compute acceleration and angular velocity
        velocity = np.array(ego_vehicle.velocity)
        if len(velocity) == 2:
            velocity = np.array([velocity[0], velocity[1], 0.0])

        if frame_id > 0:
            dt = 0.1  # 10Hz
            acceleration = (velocity - prev_velocity) / dt
            angular_velocity_z = (ego_heading - prev_heading) / dt
            angular_velocity = np.array([0.0, 0.0, angular_velocity_z])
        else:
            acceleration = np.array([0.0, 0.0, 9.8])
            angular_velocity = np.zeros(3)

        # Compute speed (magnitude of velocity)
        speed = np.linalg.norm(velocity)

        return {
            'position': ego_position,
            'heading': ego_heading,
            'velocity': velocity,
            'speed': speed,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity
        }

    def render_topdown_bev(self, env, frame_id, ego_position, ego_heading, next_waypoint, command=None, planned_traj_world=None, planned_traj_ego=None):
        """
        Render and save top-down BEV visualization with route waypoints.

        Coordinate system (MetaDrive):
            - World: right-handed, +X right, +Y up
            - Image: top-left origin, Y inverted
            - Ego is centered in the output image

        Args:
            env: MetaDrive environment
            frame_id: Current frame index
            ego_position: Ego vehicle position in world coordinates
            ego_heading: Ego vehicle heading in radians
            next_waypoint: Next navigation waypoint position
            command: Current driving command (0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANEFOLLOW)
            planned_traj_world: (N, 2) array of planned trajectory waypoints in world coordinates
            planned_traj_ego: (N, 2) array of planned trajectory in ego frame (X=lateral, Y=forward)
        """
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        frame_output_path.mkdir(parents=True, exist_ok=True)

        screen_size = 800

        try:
            # Render top-down view (ego-centered)
            topdown_frame = env.render(
                mode="top_down",
                semantic_map=True,
                film_size=(10000, 10000),
                screen_size=(screen_size, screen_size),
                draw_target_vehicle_trajectory=True,
                window=False
            )

            # Get actual scaling from MetaDrive's renderer (pixels per meter)
            scaling = env.engine.top_down_renderer.scaling
            center_px = screen_size // 2

            def world_to_pixel(world_pos):
                """Convert world position to pixel coordinates relative to ego."""
                # Delta from ego in world frame (meters)
                dx_m = world_pos[0] - ego_position[0]
                dy_m = world_pos[1] - ego_position[1]

                # Convert to pixel offset using MetaDrive's scaling
                dx_px = dx_m * scaling
                dy_px = dy_m * scaling

                # Ego is at center, Y is inverted in image space
                pixel_x = int(center_px + dx_px)
                pixel_y = int(center_px - dy_px)

                return pixel_x, pixel_y

            def is_in_bounds(px, py):
                return 0 <= px < screen_size and 0 <= py < screen_size

            # Draw route waypoints within eval window (red dots)
            # Calculate eval window bounds
            eval_start_frame = self.score_start_frame
            if self.eval_frames is not None:
                eval_end_frame = min(self.score_start_frame + self.eval_frames, self.scenario_data['length'])
            else:
                eval_end_frame = self.scenario_data['length']

            waypoint_list = list(self.route)
            visible_idx = 0
            for i, waypoint_data in enumerate(waypoint_list):
                waypoint_pos = waypoint_data[0]
                waypoint_frame_idx = waypoint_data[2]  # frame_idx is the 3rd element

                # Only draw waypoints within the eval window
                if waypoint_frame_idx < eval_start_frame or waypoint_frame_idx > eval_end_frame:
                    continue

                px, py = world_to_pixel(waypoint_pos)

                if is_in_bounds(px, py):
                    radius = 5 if visible_idx % 10 == 0 else 2
                    cv2.circle(topdown_frame, (px, py), radius, (0, 0, 255), -1)
                visible_idx += 1

            # Draw planned trajectory (cyan/yellow gradient line with dots)
            if planned_traj_world is not None and len(planned_traj_world) > 0:
                traj_points = []
                for i, wp in enumerate(planned_traj_world):
                    px, py = world_to_pixel(wp)
                    if is_in_bounds(px, py):
                        traj_points.append((px, py))
                        # Draw dots with gradient from cyan (near) to yellow (far)
                        t = i / max(len(planned_traj_world) - 1, 1)
                        color = (
                            int(255 * (1 - t)),      # B: 255 -> 0
                            int(255),                 # G: 255
                            int(255 * t)              # R: 0 -> 255
                        )
                        cv2.circle(topdown_frame, (px, py), 4, color, -1)

                # Draw lines connecting trajectory points
                if len(traj_points) > 1:
                    for i in range(len(traj_points) - 1):
                        t = i / max(len(traj_points) - 2, 1)
                        color = (
                            int(255 * (1 - t)),
                            int(255),
                            int(255 * t)
                        )
                        cv2.line(topdown_frame, traj_points[i], traj_points[i + 1], color, 2)

            # Draw ego-frame trajectory (magenta) - transformed to world for verification
            # If transformation is correct, this should overlap with the world trajectory
            if planned_traj_ego is not None and len(planned_traj_ego) > 0:
                # Transform ego-frame to world coordinates
                # Ego frame: X = lateral (right positive), Y = forward
                cos_h = np.cos(ego_heading)
                sin_h = np.sin(ego_heading)

                ego_traj_points = []
                for i, wp in enumerate(planned_traj_ego):
                    # wp[0] = lateral (X in ego), wp[1] = forward (Y in ego)
                    # Transform: world_x = ego_x + cos(h)*forward - sin(h)*lateral
                    #            world_y = ego_y + sin(h)*forward + cos(h)*lateral
                    world_x = ego_position[0] + cos_h * wp[1] - sin_h * wp[0]
                    world_y = ego_position[1] + sin_h * wp[1] + cos_h * wp[0]

                    px, py = world_to_pixel(np.array([world_x, world_y]))
                    if is_in_bounds(px, py):
                        ego_traj_points.append((px, py))
                        # Draw magenta dots with slight offset to distinguish from world traj
                        cv2.circle(topdown_frame, (px + 2, py + 2), 3, (255, 0, 255), -1)

                # Draw lines connecting ego trajectory points
                if len(ego_traj_points) > 1:
                    for i in range(len(ego_traj_points) - 1):
                        cv2.line(topdown_frame, ego_traj_points[i], ego_traj_points[i + 1], (255, 0, 255), 1)

            # Draw next waypoint (green dot, larger)
            px, py = world_to_pixel(next_waypoint)
            if is_in_bounds(px, py):
                cv2.circle(topdown_frame, (px, py), 8, (0, 255, 0), -1)

            # Draw ego marker (blue dot at center)
            cv2.circle(topdown_frame, (center_px, center_px), 6, (255, 0, 0), -1)

            # Draw legend with driving command
            legend_x, legend_y = 10, 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Command name mapping (0-indexed as passed to model)
            command_names = {
                0: "LEFT",
                1: "RIGHT",
                2: "STRAIGHT",
                3: "LANEFOLLOW",
                -1: "VOID"
            }
            command_colors = {
                0: (0, 165, 255),    # Orange for LEFT
                1: (255, 0, 255),    # Magenta for RIGHT
                2: (0, 255, 255),    # Yellow for STRAIGHT
                3: (0, 255, 0),      # Green for LANEFOLLOW
                -1: (128, 128, 128)  # Gray for VOID
            }

            # Draw semi-transparent background for legend
            overlay = topdown_frame.copy()
            cv2.rectangle(overlay, (5, 5), (200, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, topdown_frame, 0.5, 0, topdown_frame)

            # Draw frame info
            cv2.putText(topdown_frame, f"Frame: {frame_id}", (legend_x, legend_y),
                        font, font_scale, (255, 255, 255), thickness)

            # Draw command
            if command is not None:
                cmd_name = command_names.get(command, f"UNKNOWN({command})")
                cmd_color = command_colors.get(command, (255, 255, 255))
                cv2.putText(topdown_frame, f"Command: {cmd_name}", (legend_x, legend_y + 25),
                            font, font_scale, cmd_color, thickness)

            # Draw legend markers
            cv2.circle(topdown_frame, (legend_x + 5, legend_y + 50), 5, (255, 0, 0), -1)
            cv2.putText(topdown_frame, "Ego", (legend_x + 15, legend_y + 55),
                        font, 0.4, (255, 255, 255), 1)

            cv2.circle(topdown_frame, (legend_x + 60, legend_y + 50), 5, (0, 255, 0), -1)
            cv2.putText(topdown_frame, "Target", (legend_x + 70, legend_y + 55),
                        font, 0.4, (255, 255, 255), 1)

            cv2.circle(topdown_frame, (legend_x + 130, legend_y + 50), 5, (0, 0, 255), -1)
            cv2.putText(topdown_frame, "Route", (legend_x + 140, legend_y + 55),
                        font, 0.4, (255, 255, 255), 1)

            # Draw planned trajectory legend (cyan-yellow gradient = world, magenta = ego-frame)
            cv2.circle(topdown_frame, (legend_x + 5, legend_y + 70), 5, (255, 255, 0), -1)
            cv2.putText(topdown_frame, "Plan(W)", (legend_x + 15, legend_y + 75),
                        font, 0.4, (255, 255, 255), 1)

            cv2.circle(topdown_frame, (legend_x + 90, legend_y + 70), 5, (255, 0, 255), -1)
            cv2.putText(topdown_frame, "Plan(E)", (legend_x + 100, legend_y + 75),
                        font, 0.4, (255, 255, 255), 1)

            # Save frame
            topdown_path = frame_output_path / "topdown.png"
            cv2.imwrite(str(topdown_path), topdown_frame)

        except Exception as e:
            print(f"Warning: Failed to render top-down view at frame {frame_id}: {e}")

    def render_topdown_bev_candidates_topk(self, env, frame_id, ego_position, ego_heading,
                                           trajectory_topk_ego, topk_scores=None, planned_traj_world=None,
                                           prediction_position=None, prediction_heading=None):
        """
        Render top-down BEV with top-K (32) trajectory candidates for DiffusionDriveV2.
        No legend for individual candidates - just shows the distribution of candidates.

        Args:
            env: MetaDrive environment
            frame_id: Current frame index
            ego_position: Current ego vehicle position in world coordinates (for view centering)
            ego_heading: Current ego vehicle heading in radians (unused, kept for API consistency)
            trajectory_topk_ego: (K, N, 2) array of K candidate trajectories in ego frame at prediction time
            topk_scores: (K,) array of scores for each candidate (optional, for color coding)
            planned_traj_world: (N, 2) array of final planned trajectory in world coordinates
            prediction_position: Ego position at prediction time (for transforming candidates)
            prediction_heading: Ego heading at prediction time (for transforming candidates)
        """
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        frame_output_path.mkdir(parents=True, exist_ok=True)

        screen_size = 800

        # Use prediction-time pose for candidate transformation, fallback to current pose
        pred_pos = prediction_position if prediction_position is not None else ego_position
        pred_heading = prediction_heading if prediction_heading is not None else ego_heading

        try:
            # Render top-down view (ego-centered)
            topdown_frame = env.render(
                mode="top_down",
                semantic_map=True,
                film_size=(10000, 10000),
                screen_size=(screen_size, screen_size),
                draw_target_vehicle_trajectory=True,
                window=False
            )

            scaling = env.engine.top_down_renderer.scaling
            center_px = screen_size // 2

            def world_to_pixel(world_pos):
                # Center view on CURRENT ego position
                dx_m = world_pos[0] - ego_position[0]
                dy_m = world_pos[1] - ego_position[1]
                dx_px = dx_m * scaling
                dy_px = dy_m * scaling
                pixel_x = int(center_px + dx_px)
                pixel_y = int(center_px - dy_px)
                return pixel_x, pixel_y

            def is_in_bounds(px, py):
                return 0 <= px < screen_size and 0 <= py < screen_size

            # Use PREDICTION-TIME pose for transforming candidates from ego frame to world
            cos_h = np.cos(pred_heading)
            sin_h = np.sin(pred_heading)

            # Draw all top-K candidates with color based on score (blue=low, red=high)
            if trajectory_topk_ego is not None and len(trajectory_topk_ego) > 0:
                # Normalize scores for color mapping
                if topk_scores is not None:
                    scores_min = topk_scores.min()
                    scores_max = topk_scores.max()
                    if scores_max > scores_min:
                        scores_norm = (topk_scores - scores_min) / (scores_max - scores_min)
                    else:
                        scores_norm = np.ones_like(topk_scores) * 0.5
                else:
                    scores_norm = np.linspace(0, 1, len(trajectory_topk_ego))

                for cand_idx, candidate_traj in enumerate(trajectory_topk_ego):
                    # Color: blue (low score) -> red (high score)
                    t = scores_norm[cand_idx]
                    color = (
                        int(255 * (1 - t)),  # B
                        0,                    # G
                        int(255 * t)          # R
                    )

                    cand_points = []
                    for wp in candidate_traj:
                        # Transform from ego frame (at prediction time) to world coordinates
                        world_x = pred_pos[0] + cos_h * wp[1] - sin_h * wp[0]
                        world_y = pred_pos[1] + sin_h * wp[1] + cos_h * wp[0]
                        px, py = world_to_pixel(np.array([world_x, world_y]))
                        if is_in_bounds(px, py):
                            cand_points.append((px, py))

                    # Draw thin lines for candidates
                    if len(cand_points) > 1:
                        for i in range(len(cand_points) - 1):
                            cv2.line(topdown_frame, cand_points[i], cand_points[i + 1], color, 1)

            # Draw final planned trajectory (thick cyan line) on top
            if planned_traj_world is not None and len(planned_traj_world) > 0:
                traj_points = []
                for wp in planned_traj_world:
                    px, py = world_to_pixel(wp)
                    if is_in_bounds(px, py):
                        traj_points.append((px, py))
                        cv2.circle(topdown_frame, (px, py), 3, (255, 255, 0), -1)

                if len(traj_points) > 1:
                    for i in range(len(traj_points) - 1):
                        cv2.line(topdown_frame, traj_points[i], traj_points[i + 1], (255, 255, 0), 2)

            # Draw ego marker
            cv2.circle(topdown_frame, (center_px, center_px), 6, (255, 0, 0), -1)

            # Simple legend
            overlay = topdown_frame.copy()
            cv2.rectangle(overlay, (5, 5), (180, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, topdown_frame, 0.5, 0, topdown_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(topdown_frame, f"Frame: {frame_id}", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(topdown_frame, f"Top-32 Candidates", (10, 40), font, 0.5, (255, 255, 255), 1)
            cv2.putText(topdown_frame, "Blue=Low, Red=High score", (10, 55), font, 0.35, (200, 200, 200), 1)

            # Save frame
            topdown_path = frame_output_path / "topdown_topk.png"
            cv2.imwrite(str(topdown_path), topdown_frame)

        except Exception as e:
            print(f"Warning: Failed to render top-K candidates at frame {frame_id}: {e}")

    def render_topdown_bev_candidates_finegrained(self, env, frame_id, ego_position, ego_heading,
                                                   trajectory_candidates_ego, planned_traj_world=None,
                                                   prediction_position=None, prediction_heading=None):
        """
        Render top-down BEV with fine-grained (4) trajectory candidates for DiffusionDriveV2.
        Shows legend for each candidate type.

        Args:
            env: MetaDrive environment
            frame_id: Current frame index
            ego_position: Current ego vehicle position in world coordinates (for view centering)
            ego_heading: Current ego vehicle heading in radians (unused, kept for API consistency)
            trajectory_candidates_ego: (4, N, 2) array of 4 fine-grained candidates in ego frame at prediction time
            planned_traj_world: (N, 2) array of final planned trajectory in world coordinates
            prediction_position: Ego position at prediction time (for transforming candidates)
            prediction_heading: Ego heading at prediction time (for transforming candidates)
        """
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        frame_output_path.mkdir(parents=True, exist_ok=True)

        screen_size = 800

        # Use prediction-time pose for candidate transformation, fallback to current pose
        pred_pos = prediction_position if prediction_position is not None else ego_position
        pred_heading = prediction_heading if prediction_heading is not None else ego_heading

        try:
            # Render top-down view (ego-centered)
            topdown_frame = env.render(
                mode="top_down",
                semantic_map=True,
                film_size=(10000, 10000),
                screen_size=(screen_size, screen_size),
                draw_target_vehicle_trajectory=True,
                window=False
            )

            scaling = env.engine.top_down_renderer.scaling
            center_px = screen_size // 2

            def world_to_pixel(world_pos):
                # Center view on CURRENT ego position
                dx_m = world_pos[0] - ego_position[0]
                dy_m = world_pos[1] - ego_position[1]
                dx_px = dx_m * scaling
                dy_px = dy_m * scaling
                pixel_x = int(center_px + dx_px)
                pixel_y = int(center_px - dy_px)
                return pixel_x, pixel_y

            def is_in_bounds(px, py):
                return 0 <= px < screen_size and 0 <= py < screen_size

            # Use PREDICTION-TIME pose for transforming candidates from ego frame to world
            cos_h = np.cos(pred_heading)
            sin_h = np.sin(pred_heading)

            # Colors for 4 fine-grained candidates
            candidate_colors = [
                (0, 165, 255),    # Orange (BGR) - Coarse best
                (0, 255, 0),      # Green - Fine best 1
                (255, 0, 0),      # Blue - Fine best 2
                (255, 0, 255),    # Purple - Fine best 3 (selected)
            ]
            candidate_labels = ["Coarse", "Fine1", "Fine2", "Selected"]

            # Draw all 4 candidates
            if trajectory_candidates_ego is not None and len(trajectory_candidates_ego) > 0:
                for cand_idx, candidate_traj in enumerate(trajectory_candidates_ego):
                    if cand_idx >= len(candidate_colors):
                        break
                    color = candidate_colors[cand_idx]
                    thickness = 2 if cand_idx == len(trajectory_candidates_ego) - 1 else 1  # Thick for selected

                    cand_points = []
                    for wp in candidate_traj:
                        # Transform from ego frame (at prediction time) to world coordinates
                        world_x = pred_pos[0] + cos_h * wp[1] - sin_h * wp[0]
                        world_y = pred_pos[1] + sin_h * wp[1] + cos_h * wp[0]
                        px, py = world_to_pixel(np.array([world_x, world_y]))
                        if is_in_bounds(px, py):
                            cand_points.append((px, py))
                            cv2.circle(topdown_frame, (px, py), 3, color, -1)

                    if len(cand_points) > 1:
                        for i in range(len(cand_points) - 1):
                            cv2.line(topdown_frame, cand_points[i], cand_points[i + 1], color, thickness)

            # Draw ego marker
            cv2.circle(topdown_frame, (center_px, center_px), 6, (255, 0, 0), -1)

            # Legend with candidate labels
            overlay = topdown_frame.copy()
            legend_height = 80 + len(candidate_colors) * 18
            cv2.rectangle(overlay, (5, 5), (150, legend_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, topdown_frame, 0.5, 0, topdown_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(topdown_frame, f"Frame: {frame_id}", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(topdown_frame, "Fine Candidates:", (10, 40), font, 0.45, (255, 255, 255), 1)

            for i, (color, label) in enumerate(zip(candidate_colors[:len(trajectory_candidates_ego) if trajectory_candidates_ego is not None else 0], candidate_labels)):
                y_offset = 60 + i * 18
                cv2.circle(topdown_frame, (15, y_offset), 5, color, -1)
                cv2.putText(topdown_frame, label, (25, y_offset + 5), font, 0.4, (255, 255, 255), 1)

            # Save frame
            topdown_path = frame_output_path / "topdown_finegrained.png"
            cv2.imwrite(str(topdown_path), topdown_frame)

        except Exception as e:
            print(f"Warning: Failed to render fine-grained candidates at frame {frame_id}: {e}")

    def render_topdown_bev_candidates_coarse(self, env, frame_id, ego_position, ego_heading,
                                              trajectory_coarse_ego, coarse_scores=None, planned_traj_world=None,
                                              prediction_position=None, prediction_heading=None):
        """
        Render top-down BEV with all coarse (200) trajectory candidates for DiffusionDriveV2.
        No legend for individual candidates - just shows the full distribution of candidates.

        Args:
            env: MetaDrive environment
            frame_id: Current frame index
            ego_position: Current ego vehicle position in world coordinates (for view centering)
            ego_heading: Current ego vehicle heading in radians (unused, kept for API consistency)
            trajectory_coarse_ego: (200, N, 2) array of 200 coarse candidate trajectories in ego frame at prediction time
            coarse_scores: (200,) array of scores for each candidate (optional, for color coding)
            planned_traj_world: (N, 2) array of final planned trajectory in world coordinates
            prediction_position: Ego position at prediction time (for transforming candidates)
            prediction_heading: Ego heading at prediction time (for transforming candidates)
        """
        frame_output_path = self.output_dir / f"{frame_id:05d}"
        frame_output_path.mkdir(parents=True, exist_ok=True)

        screen_size = 800

        # Use prediction-time pose for candidate transformation, fallback to current pose
        pred_pos = prediction_position if prediction_position is not None else ego_position
        pred_heading = prediction_heading if prediction_heading is not None else ego_heading

        try:
            # Render top-down view (ego-centered)
            topdown_frame = env.render(
                mode="top_down",
                semantic_map=True,
                film_size=(10000, 10000),
                screen_size=(screen_size, screen_size),
                draw_target_vehicle_trajectory=True,
                window=False
            )

            scaling = env.engine.top_down_renderer.scaling
            center_px = screen_size // 2

            def world_to_pixel(world_pos):
                # Center view on CURRENT ego position
                dx_m = world_pos[0] - ego_position[0]
                dy_m = world_pos[1] - ego_position[1]
                dx_px = dx_m * scaling
                dy_px = dy_m * scaling
                pixel_x = int(center_px + dx_px)
                pixel_y = int(center_px - dy_px)
                return pixel_x, pixel_y

            def is_in_bounds(px, py):
                return 0 <= px < screen_size and 0 <= py < screen_size

            # Use PREDICTION-TIME pose for transforming candidates from ego frame to world
            cos_h = np.cos(pred_heading)
            sin_h = np.sin(pred_heading)

            # Draw all coarse candidates with color based on score (blue=low, red=high)
            if trajectory_coarse_ego is not None and len(trajectory_coarse_ego) > 0:
                # Normalize scores for color mapping
                if coarse_scores is not None:
                    scores_min = coarse_scores.min()
                    scores_max = coarse_scores.max()
                    if scores_max > scores_min:
                        scores_norm = (coarse_scores - scores_min) / (scores_max - scores_min)
                    else:
                        scores_norm = np.ones_like(coarse_scores) * 0.5
                else:
                    scores_norm = np.linspace(0, 1, len(trajectory_coarse_ego))

                for cand_idx, candidate_traj in enumerate(trajectory_coarse_ego):
                    # Color: blue (low score) -> red (high score)
                    t = scores_norm[cand_idx]
                    color = (
                        int(255 * (1 - t)),  # B
                        0,                    # G
                        int(255 * t)          # R
                    )

                    cand_points = []
                    for wp in candidate_traj:
                        # Transform from ego frame (at prediction time) to world coordinates
                        world_x = pred_pos[0] + cos_h * wp[1] - sin_h * wp[0]
                        world_y = pred_pos[1] + sin_h * wp[1] + cos_h * wp[0]
                        px, py = world_to_pixel(np.array([world_x, world_y]))
                        if is_in_bounds(px, py):
                            cand_points.append((px, py))

                    # Draw very thin lines for coarse candidates (thinner than topk)
                    if len(cand_points) > 1:
                        for i in range(len(cand_points) - 1):
                            cv2.line(topdown_frame, cand_points[i], cand_points[i + 1], color, 1)

            # Draw final planned trajectory (thick cyan line) on top
            if planned_traj_world is not None and len(planned_traj_world) > 0:
                traj_points = []
                for wp in planned_traj_world:
                    px, py = world_to_pixel(wp)
                    if is_in_bounds(px, py):
                        traj_points.append((px, py))
                        cv2.circle(topdown_frame, (px, py), 3, (255, 255, 0), -1)

                if len(traj_points) > 1:
                    for i in range(len(traj_points) - 1):
                        cv2.line(topdown_frame, traj_points[i], traj_points[i + 1], (255, 255, 0), 2)

            # Draw ego marker
            cv2.circle(topdown_frame, (center_px, center_px), 6, (255, 0, 0), -1)

            # Simple legend
            overlay = topdown_frame.copy()
            cv2.rectangle(overlay, (5, 5), (200, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, topdown_frame, 0.5, 0, topdown_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(topdown_frame, f"Frame: {frame_id}", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(topdown_frame, f"All 200 Coarse Candidates", (10, 40), font, 0.5, (255, 255, 255), 1)
            cv2.putText(topdown_frame, "Blue=Low, Red=High score", (10, 55), font, 0.35, (200, 200, 200), 1)

            # Save frame
            topdown_path = frame_output_path / "topdown_coarse.png"
            cv2.imwrite(str(topdown_path), topdown_frame)

        except Exception as e:
            print(f"Warning: Failed to render coarse candidates at frame {frame_id}: {e}")

    def render_replan_summary(self, env, final_ego_position):
        """
        Render a single summary image showing all replan trajectories and their top-k candidate coverage.

        For each replan:
        - Top-k candidates: Light, transparent coverage showing possible paths
        - Executed trajectory: Solid #DD8452 color

        This is called ONCE at the end of evaluation to create a summary visualization.

        Args:
            env: MetaDrive environment
            final_ego_position: Final ego position in world coordinates (for view centering)
        """
        if len(self.planning_history) == 0:
            print("No planning history to render.")
            return

        screen_size = 800

        # Colors
        # #4C72B0 in BGR format: B=0xB0=176, G=0x72=114, R=0x4C=76
        candidate_color = (176, 114, 76)  # Blue for proposal coverage
        executed_color = (0, 255, 0)  # Green for executed trajectories
        route_color = (0, 0, 255)  # Red for route

        try:
            # Render top-down view centered on final ego position
            topdown_frame = env.render(
                mode="top_down",
                semantic_map=True,
                film_size=(10000, 10000),
                screen_size=(screen_size, screen_size),
                draw_target_vehicle_trajectory=False,  # We'll draw our own
                window=False
            )

            scaling = env.engine.top_down_renderer.scaling
            center_px = screen_size // 2

            def world_to_pixel(world_pos):
                # Center view on final ego position
                dx_m = world_pos[0] - final_ego_position[0]
                dy_m = world_pos[1] - final_ego_position[1]
                dx_px = dx_m * scaling
                dy_px = dy_m * scaling
                pixel_x = int(center_px + dx_px)
                pixel_y = int(center_px - dy_px)
                return pixel_x, pixel_y

            def is_in_bounds(px, py):
                return 0 <= px < screen_size and 0 <= py < screen_size

            # Debug counter
            total_candidates_drawn = 0

            # 1. Collect route waypoints from self.full_route (same as render_topdown_bev)
            # Calculate eval window bounds
            eval_start_frame = self.score_start_frame
            if self.eval_frames is not None:
                eval_end_frame = min(self.score_start_frame + self.eval_frames, self.scenario_data['length'])
            else:
                eval_end_frame = self.scenario_data['length']

            route_points = []
            if hasattr(self, 'full_route') and self.full_route:
                for waypoint_data in self.full_route:
                    waypoint_pos = waypoint_data[0]  # Position is first element
                    waypoint_frame_idx = waypoint_data[2]  # frame_idx is 3rd element

                    # Only include waypoints within the eval window
                    if waypoint_frame_idx < eval_start_frame or waypoint_frame_idx > eval_end_frame:
                        continue

                    px, py = world_to_pixel(waypoint_pos)
                    if is_in_bounds(px, py):
                        route_points.append((px, py))

            # 2. Draw all top-k candidates as transparent coverage (blue)
            candidate_layer = topdown_frame.copy()

            for plan_frame_id, world_traj, pred_pos, pred_heading, consumed_count, topk_world in self.planning_history:
                if topk_world is None or len(topk_world) == 0:
                    continue

                # Draw each candidate trajectory
                for cand_traj in topk_world:
                    if cand_traj is None or len(cand_traj) < 2:
                        continue

                    cand_points = []
                    for wp in cand_traj:
                        px, py = world_to_pixel(wp)
                        if is_in_bounds(px, py):
                            cand_points.append((px, py))

                    # Draw lines for candidates (thin, anti-aliased)
                    if len(cand_points) > 1:
                        for i in range(len(cand_points) - 1):
                            cv2.line(candidate_layer, cand_points[i], cand_points[i + 1], candidate_color, 1, cv2.LINE_AA)
                        total_candidates_drawn += 1

            # Blend candidate layer with main frame (make candidates semi-transparent)
            alpha = 0.5  # 50% opacity for candidates
            topdown_frame = cv2.addWeighted(candidate_layer, alpha, topdown_frame, 1 - alpha, 0)

            # 3. Draw executed trajectory in green
            all_executed_points = []
            for plan_frame_id, world_traj, pred_pos, pred_heading, consumed_count, topk_world in self.planning_history:
                if world_traj is None or len(world_traj) == 0:
                    continue
                # Get the consumed (executed) portion
                consumed_traj = world_traj[:consumed_count] if consumed_count > 0 else []
                for wp in consumed_traj:
                    px, py = world_to_pixel(wp)
                    if is_in_bounds(px, py):
                        all_executed_points.append((px, py))

            # 4. Draw route as dotted red line
            if len(route_points) > 1:
                # Draw dotted line by drawing short segments with gaps
                for i in range(len(route_points) - 1):
                    pt1 = route_points[i]
                    pt2 = route_points[i + 1]
                    # Calculate distance and direction
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        # Normalize direction
                        dx, dy = dx / dist, dy / dist
                        # Draw dashes (every 6 pixels: 4 on, 2 off)
                        dash_len, gap_len = 4, 2
                        pos = 0
                        while pos < dist:
                            end_pos = min(pos + dash_len, dist)
                            start_pt = (int(pt1[0] + dx * pos), int(pt1[1] + dy * pos))
                            end_pt = (int(pt1[0] + dx * end_pos), int(pt1[1] + dy * end_pos))
                            cv2.line(topdown_frame, start_pt, end_pt, route_color, 2, cv2.LINE_AA)
                            pos += dash_len + gap_len

            # 5. Draw executed trajectory in green on top of route
            if len(all_executed_points) > 1:
                for i in range(len(all_executed_points) - 1):
                    cv2.line(topdown_frame, all_executed_points[i], all_executed_points[i + 1], executed_color, 2, cv2.LINE_AA)

            print(f"[Replan Summary] Drew {total_candidates_drawn} candidate trajectories")

            # Draw legend
            legend_overlay = topdown_frame.copy()
            legend_height = 95
            cv2.rectangle(legend_overlay, (5, 5), (200, legend_height), (0, 0, 0), -1)
            cv2.addWeighted(legend_overlay, 0.6, topdown_frame, 0.4, 0, topdown_frame)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(topdown_frame, f"Replan Summary (RR={self.replan_rate})", (10, 20), font, 0.4, (255, 255, 255), 1)

            # Legend items
            # Proposal Coverage (blue)
            cv2.line(topdown_frame, (10, 35), (30, 35), candidate_color, 2)
            cv2.putText(topdown_frame, "Proposal Coverage", (35, 38), font, 0.35, (255, 255, 255), 1)

            # Executed Trajectories (green)
            cv2.line(topdown_frame, (10, 52), (30, 52), executed_color, 2)
            cv2.putText(topdown_frame, "Executed Trajectories", (35, 55), font, 0.35, (255, 255, 255), 1)

            # Route (dotted red)
            cv2.line(topdown_frame, (10, 69), (15, 69), route_color, 1)
            cv2.line(topdown_frame, (20, 69), (25, 69), route_color, 1)
            cv2.line(topdown_frame, (30, 69), (35, 69), route_color, 1)
            cv2.putText(topdown_frame, "Route", (40, 72), font, 0.35, (255, 255, 255), 1)

            cv2.putText(topdown_frame, f"Total Replans: {len(self.planning_history)}", (10, 88), font, 0.35, (200, 200, 200), 1)

            # Save summary image
            summary_path = self.output_dir / "replan_summary.png"
            cv2.imwrite(str(summary_path), topdown_frame)
            print(f"Saved replan summary to: {summary_path}")

        except Exception as e:
            print(f"Warning: Failed to render replan summary: {e}")

    def visualize_segmentation(self, frame_id, seg_output_path):
        """
        Visualize segmentation outputs (drivable area, lanes) as BEV image.

        Args:
            frame_id: Current frame index
            seg_output_path: Path to seg_output.pth file
        """
        try:
            import torch
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            frame_output_path = self.output_dir / f"{frame_id:05d}"
            seg_output = torch.load(seg_output_path, map_location='cpu')

            if 'pts_bbox' not in seg_output:
                return
            pts_bbox = seg_output['pts_bbox']

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            if 'drivable' in pts_bbox:
                drivable = pts_bbox['drivable'].cpu().numpy()
                axes[0].imshow(drivable, cmap='gray', origin='lower')
                axes[0].set_title('Drivable Area')
                axes[0].axis('off')

            if 'lane' in pts_bbox:
                lane = pts_bbox['lane'].cpu().numpy()
                lane_combined = lane.max(axis=0)
                axes[1].imshow(lane_combined, cmap='hot', origin='lower')
                axes[1].set_title('Lane Lines')
                axes[1].axis('off')

            plt.tight_layout()
            seg_vis_path = frame_output_path / "segmentation_vis.png"
            plt.savefig(str(seg_vis_path), dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to visualize segmentation at frame {frame_id}: {e}")

    def visualize_occupancy(self, frame_id, occ_output_path):
        """Visualize occupancy prediction."""
        try:
            import torch
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            frame_output_path = self.output_dir / f"{frame_id:05d}"
            occ_output = torch.load(occ_output_path, map_location='cpu')

            if 'seg_out' not in occ_output:
                return

            seg_out = occ_output['seg_out'][0, 0, 0].cpu().numpy()

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            im = ax.imshow(seg_out, cmap='viridis', origin='lower')
            ax.set_title('Occupancy Grid (BEV)')
            ax.axis('off')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            occ_vis_path = frame_output_path / "occupancy_vis.png"
            plt.savefig(str(occ_vis_path), dpi=100)
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to visualize occupancy at frame {frame_id}: {e}")

    def visualize_bev_embed(self, frame_id, bev_embed_path):
        """Visualize BEV embeddings."""
        try:
            import torch
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            frame_output_path = self.output_dir / f"{frame_id:05d}"
            bev_embed = torch.load(bev_embed_path, map_location='cpu')
            bev_embed = bev_embed.squeeze(1).reshape(200, 200, 256)

            bev_mean = bev_embed.mean(dim=-1).cpu().numpy()
            bev_rgb = bev_embed[:, :, :3].cpu().numpy()

            for i in range(3):
                channel = bev_rgb[:, :, i]
                channel_min, channel_max = channel.min(), channel.max()
                if channel_max > channel_min:
                    bev_rgb[:, :, i] = (channel - channel_min) / (channel_max - channel_min)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            im1 = axes[0].imshow(bev_mean, cmap='hot', origin='lower')
            axes[0].set_title('BEV Embeddings (Mean)')
            axes[0].axis('off')

            divider1 = make_axes_locatable(axes[0])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax1)

            axes[1].imshow(bev_rgb, origin='lower')
            axes[1].set_title('BEV Embeddings (RGB: First 3 Channels)')
            axes[1].axis('off')

            bev_vis_path = frame_output_path / "bev_embed_vis.png"
            plt.savefig(str(bev_vis_path), dpi=100)
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to visualize BEV embeddings at frame {frame_id}: {e}")

    def generate_gif(self, frame_ids, filename, source_filename):
        """Helper to generate GIFs from image sequences."""
        print(f"\nGenerating GIF {filename} from {source_filename}...")
        try:
            images = []
            for frame_id in frame_ids:
                path = self.output_dir / f"{frame_id:05d}" / source_filename
                if path.exists():
                    images.append(path)

            if images:
                frames = []
                for img_path in images:
                    frame = cv2.imread(str(img_path))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                try:
                    import imageio
                    gif_path = self.output_dir / filename
                    imageio.mimsave(str(gif_path), frames, fps=10, loop=0)
                    print(f"✓ Saved GIF to: {gif_path}")
                except ImportError:
                    print("Note: Install imageio to generate GIF: pip install imageio")
            else:
                print(f"No images found for {filename}")
        except Exception as e:
            print(f"Warning: Failed to generate {filename}: {e}")

    def process_frame(self, env, frame_id, prev_velocity, prev_heading):
        """Process a single frame."""
        # 1. Perceive
        imgs = self.perceive(env, frame_id)

        # 2. Compute ego state
        ego_state = self.compute_ego_state(env, frame_id, prev_velocity, prev_heading)

        # 3. Get navigation waypoint
        waypoint_pos, command, frame_idx = self.get_next_waypoint(ego_state['position'][:2])

        # Add navigation info to ego state
        ego_state['waypoint'] = waypoint_pos
        ego_state['command'] = command

        # 4. Prepare model input
        model_input = self.model_adapter.prepare_input(
            images=imgs,
            ego_state=ego_state,
            scenario_data=self.scenario_data,
            frame_id=frame_id
        )

        # 5. Determine if we need to replan
        # During replay: inference at frame 0 and every replan_rate (for warmup)
        # After replay: force replan at ego_replay_frames, then every replan_rate from there
        # e.g., replay=30, replan_rate=20 -> inference at 0, 20, 30, 50, 70...
        if frame_id < self.ego_replay_frames:
            # During replay: warmup inference follows normal replan_rate from frame 0
            replan_offset = frame_id % self.replan_rate
            needs_replan = (replan_offset == 0)
        elif frame_id == self.ego_replay_frames:
            # Transition frame: force replan to restart cycle
            replan_offset = 0
            needs_replan = True
        else:
            # After replay: replan cycle counts from ego_replay_frames
            frames_since_replay_end = frame_id - self.ego_replay_frames
            replan_offset = frames_since_replay_end % self.replan_rate
            needs_replan = (replan_offset == 0)

        # 6. Run inference only when it's time to replan
        if needs_replan:
            # Set simulation time for temporal consistency scoring (if supported)
            current_sim_time = frame_id * self.sim_dt
            if hasattr(self.model_adapter, 'set_simulation_time'):
                self.model_adapter.set_simulation_time(current_sim_time)

            model_output = self.model_adapter.run_inference(model_input)
            parsed_output = self.model_adapter.parse_output(model_output, ego_state)
            plan_traj_ego = np.asarray(parsed_output['trajectory'])  # In ego frame

            # Interpolate trajectory from model_dt to sim_dt intervals
            model_dt = self.model_adapter.get_waypoint_dt()
            plan_traj_ego = self._interpolate_trajectory(plan_traj_ego, model_dt, self.sim_dt)

            # Transform trajectory from ego frame to world coordinates and cache
            cos_h = np.cos(ego_state['heading'])
            sin_h = np.sin(ego_state['heading'])

            ego_left = plan_traj_ego[:, 0]
            ego_forward = plan_traj_ego[:, 1]
            world_x = ego_state['position'][0] + cos_h * ego_forward - sin_h * ego_left
            world_y = ego_state['position'][1] + sin_h * ego_forward + cos_h * ego_left

            self.cached_plan_traj_world = np.stack((world_x, world_y), axis=1)
            self.cached_model_output = model_output
            self.last_replan_frame = frame_id
            # Store ego pose at prediction time for candidate trajectory visualization
            self.cached_prediction_position = ego_state['position'].copy()
            self.cached_prediction_heading = ego_state['heading']

            # Update planning history for replan summary visualization
            # Update consumed_count for the previous plan (it was consumed up to replan_rate waypoints)
            if len(self.planning_history) > 0:
                prev_entry = self.planning_history[-1]
                # The previous plan was consumed for replan_rate waypoints (or until end of trajectory)
                consumed = min(self.replan_rate, len(prev_entry[1]) if prev_entry[1] is not None else 0)
                self.planning_history[-1] = (prev_entry[0], prev_entry[1], prev_entry[2], prev_entry[3], consumed, prev_entry[5])

            # Extract and transform top-k candidates to world coordinates (if available)
            topk_world = None
            if 'trajectory_topk' in parsed_output:
                topk_ego = np.asarray(parsed_output['trajectory_topk'])  # (K, N, 2) in ego frame
                if topk_ego is not None and len(topk_ego) > 0:
                    # Transform each candidate from ego frame to world coordinates
                    topk_world = []
                    for cand in topk_ego:
                        cand_left = cand[:, 0]
                        cand_forward = cand[:, 1]
                        cand_world_x = ego_state['position'][0] + cos_h * cand_forward - sin_h * cand_left
                        cand_world_y = ego_state['position'][1] + sin_h * cand_forward + cos_h * cand_left
                        topk_world.append(np.stack((cand_world_x, cand_world_y), axis=1))
                    topk_world = np.array(topk_world)

            # Add new plan to history with consumed_count=0 (will be updated on next replan)
            self.planning_history.append((
                frame_id,
                self.cached_plan_traj_world.copy(),
                ego_state['position'].copy(),
                ego_state['heading'],
                0,  # consumed_count, will be updated on next replan
                topk_world  # top-k candidates in world coordinates
            ))

        # 7. Get trajectory subset based on replan offset
        if self.cached_plan_traj_world is None:
            raise RuntimeError(f"Frame {frame_id}: No cached trajectory available. This should not happen.")

        # Check if we have enough waypoints for the current offset
        if replan_offset >= len(self.cached_plan_traj_world):
            raise RuntimeError(
                f"Frame {frame_id}: Consumed all {len(self.cached_plan_traj_world)} waypoints "
                f"(offset={replan_offset}, last_replan={self.last_replan_frame}). "
                f"Model prediction too short for replan_rate={self.replan_rate}. "
                f"Reduce replan_rate or use model that predicts more waypoints."
            )

        # Get waypoints from offset onwards (sequential consumption)
        world_traj_subset = self.cached_plan_traj_world[replan_offset:]

        # 8. Transform waypoints from world to current ego frame
        # This is critical because the controller expects waypoints in the current ego frame
        delta = world_traj_subset - ego_state['position'][:2]
        cos_h = np.cos(-ego_state['heading'])
        sin_h = np.sin(-ego_state['heading'])
        plan_traj = np.empty_like(delta)
        plan_traj[:, 0] = sin_h * delta[:, 0] + cos_h * delta[:, 1]
        plan_traj[:, 1] = cos_h * delta[:, 0] - sin_h * delta[:, 1]

        # Original loop for reference:
        # plan_traj = []
        # for i in range(len(world_traj_subset)):
        #     delta = world_traj_subset[i] - ego_state['position'][:2]
        #     cos_h = np.cos(-ego_state['heading'])
        #     sin_h = np.sin(-ego_state['heading'])
        #     ego_x = sin_h * delta[0] + cos_h * delta[1]
        #     ego_y = cos_h * delta[0] - sin_h * delta[1]
        #     plan_traj.append([ego_x, ego_y])
        # plan_traj = np.array(plan_traj)

        # Also compute full world trajectory for compatibility
        world_traj = world_traj_subset

        # 9. Control
        next_waypoint = ego_state['waypoint']
        delta_world = next_waypoint - ego_state['position'][:2]

        # Rotate into ego frame (ego heading points forward in ego frame)
        cos_h = np.cos(-ego_state['heading'])
        sin_h = np.sin(-ego_state['heading'])
        target_ego_x = sin_h * delta_world[0] + cos_h * delta_world[1]
        target_ego_y = cos_h * delta_world[0] - sin_h * delta_world[1]
        target_ego = np.array([target_ego_x, target_ego_y])

        plan_traj_subset = plan_traj #[:min(6, len(plan_traj))]

        steer, throttle, brake, control_metadata = self.controller.control_pid(
            plan_traj_subset,
            ego_state['speed'],
            target_ego,
            waypoint_dt=self.sim_dt,  # Trajectory is now interpolated to sim_dt intervals
        )

        throttle = float(throttle)
        brake = float(int(brake))

        # MetaDrive expects action as [steer, throttle - brake]
        control = np.array([steer, throttle - brake])

        # 10. Render top-down BEV visualization every frame for smooth GIFs
        if self.enable_vis:
            self.render_topdown_bev(
                env, frame_id, ego_state['position'], ego_state['heading'],
                ego_state['waypoint'], ego_state['command'],
                planned_traj_world=world_traj, planned_traj_ego=plan_traj
            )

            # Render DiffusionDriveV2-specific candidate visualizations
            # Use cached prediction pose (not current ego pose) so candidates stay anchored to prediction point
            if type(self.model_adapter).__name__ == 'DiffusionDriveV2Adapter' and self.cached_model_output is not None:
                parsed = self.model_adapter.parse_output(self.cached_model_output, ego_state)
                # Use prediction-time pose for candidate transformations
                pred_pos = self.cached_prediction_position
                pred_heading = self.cached_prediction_heading

                # Render top-32 candidates visualization
                if 'trajectory_topk' in parsed:
                    topk_scores = parsed.get('topk_scores', None)
                    self.render_topdown_bev_candidates_topk(
                        env, frame_id, ego_state['position'], ego_state['heading'],
                        trajectory_topk_ego=parsed['trajectory_topk'],
                        topk_scores=topk_scores,
                        planned_traj_world=world_traj,
                        prediction_position=pred_pos,
                        prediction_heading=pred_heading
                    )

                # Render fine-grained 4 candidates visualization
                if 'trajectory_candidates' in parsed:
                    self.render_topdown_bev_candidates_finegrained(
                        env, frame_id, ego_state['position'], ego_state['heading'],
                        trajectory_candidates_ego=parsed['trajectory_candidates'],
                        planned_traj_world=world_traj,
                        prediction_position=pred_pos,
                        prediction_heading=pred_heading
                    )

                # Render all 200 coarse candidates visualization
                if 'trajectory_coarse' in parsed:
                    coarse_scores = parsed.get('coarse_scores', None)
                    self.render_topdown_bev_candidates_coarse(
                        env, frame_id, ego_state['position'], ego_state['heading'],
                        trajectory_coarse_ego=parsed['trajectory_coarse'],
                        coarse_scores=coarse_scores,
                        planned_traj_world=world_traj,
                        prediction_position=pred_pos,
                        prediction_heading=pred_heading
                    )

        # 12. Visualize model outputs (seg, occ, bev_embed) only when model runs
        if self.enable_vis and replan_offset == 0:
            frame_output_path = self.output_dir / f"{frame_id:05d}"
            seg_output_path = frame_output_path / "seg_output.pth"
            occ_output_path = frame_output_path / "occ_output.pth"
            bev_embed_path = frame_output_path / "bev_embed.pth"

            if seg_output_path.exists():
                self.visualize_segmentation(frame_id, seg_output_path)
            if occ_output_path.exists():
                self.visualize_occupancy(frame_id, occ_output_path)
            if bev_embed_path.exists():
                self.visualize_bev_embed(frame_id, bev_embed_path)

        return control, ego_state, plan_traj, world_traj

    def run(self):
        """Run the complete evaluation."""
        # Load model
        print("Loading model...")
        self.model_adapter.load_model()

        # Reset temporal consistency history if adapter supports it
        if hasattr(self.model_adapter, 'reset_temporal_history'):
            self.model_adapter.reset_temporal_history()

        # Reset planning history for replan visualization
        self.planning_history = []

        # Validate that model trajectory supports the configured replan_rate
        time_horizon = self.model_adapter.get_trajectory_time_horizon()
        interpolated_waypoints = int(time_horizon / self.sim_dt)
        assert interpolated_waypoints >= self.replan_rate, (
            f"Model trajectory time horizon ({time_horizon}s) is insufficient for replan_rate={self.replan_rate}. "
            f"After interpolation to sim_dt={self.sim_dt}s, trajectory has {interpolated_waypoints} waypoints, "
            f"but need at least {self.replan_rate} waypoints. "
            f"Either reduce replan_rate to <= {interpolated_waypoints} or use a model with longer trajectory horizon."
        )
        print(f"  Trajectory time horizon: {time_horizon}s ({interpolated_waypoints} waypoints at {self.sim_dt}s intervals)")

        # Load scenario
        print("Loading scenario...")
        self.load_scenario()
        self.prepare_ego_replay_actions()

        # Generate route
        print("Generating route...")
        self.generate_route()
        # Store a copy of the full route for visualization (route gets consumed during evaluation)
        self.full_route = list(self.route)

        # Configure Environment Policy based on Mode
        agent_policy = EnvInputPolicy
        if self.eval_mode == "open_loop":
            print("[INFO] Running in Open Loop Mode (Agent follows Ground Truth)")
            agent_policy = ReplayEgoCarPolicy
        else:
            print("[INFO] Running in Closed Loop Mode (Agent controlled by Model)")

        # Create environment
        print("Creating environment...")
        self.env_manager = EnvironmentManager(
            self.scenario_path,
            traffic_mode=self.traffic_mode,
            render=self.enable_vis,
            image_on_cuda=True,
            agent_policy=agent_policy
        )

        # Initialize controller based on type
        if self.controller_type == "pid":
            self.controller = PIDController()
            print(f"  Controller: PIDController")
        else:
            self.controller = PurePursuitController()
            print(f"  Controller: PurePursuitController")
        
        if self.eval_mode == "open_loop":
            # In open loop, env creation happens inside run but we need to create it early for EPDMS init if needed
            # But EnvironmentManager creates env on create_env()
            pass
        
        env = self.env_manager.create_env()
        obs, info = env.reset(seed=0)
        print("Environment reset successful")
        
        # Initialize Scorers
        if self.eval_mode == "open_loop":
            self.epdms_scorer = EPDMSScorer(self.scenario_data, env)
            self.epdms_results_openloop = []
        else:
            # Closed loop mode
            self.stats_manager = OfflineStatisticsManager(self.output_dir, self.scenario_name, self.scenario_data)

            # Initialize EPDMS scorer for closed-loop (uses score_frame_live)
            self.epdms_scorer = EPDMSScorer(self.scenario_data, env)
            self.epdms_scorer.reset_live_state()  # Reset motion history for new episode
            self.epdms_results_closedloop = []

        prev_velocity = np.array([0.0, 0.0, 0.0])
        prev_heading = 0.0

        # Main evaluation loop
        scenario_length = self.scenario_data['length']
        if self.eval_frames is not None:
            # Limit frames: replay + eval, capped by scenario length
            num_frames = min(self.ego_replay_frames + self.eval_frames, scenario_length)
            print(f"Frame limit: {self.ego_replay_frames} replay + {self.eval_frames} eval = {num_frames} total (scenario has {scenario_length})")
        else:
            num_frames = scenario_length
        frame_ids = list(range(num_frames))

        # Compute expected route completion delta from log data (for partial scenario evaluation)
        # Only compute if eval_frames is set and results in fewer frames than full scenario
        if self.eval_frames is not None and num_frames < scenario_length:
            ego_track = self.scenario_data['tracks'][self.sdc_id]
            positions = ego_track['state']['position']

            start_frame = self.score_start_frame
            end_frame = num_frames  # Already capped by scenario_length

            # Sum distances between consecutive frames in the log for eval window
            expected_distance = 0.0
            for i in range(start_frame, end_frame - 1):
                expected_distance += np.linalg.norm(positions[i+1][:2] - positions[i][:2])

            # Compute total trajectory distance from log
            total_distance = 0.0
            for i in range(len(positions) - 1):
                total_distance += np.linalg.norm(positions[i+1][:2] - positions[i][:2])

            # Expected RC delta as fraction of total trajectory
            if total_distance > 0:
                self.expected_rc_delta = expected_distance / total_distance
                print(f"[INFO] Partial scenario evaluation: expected RC delta = {self.expected_rc_delta:.4f} "
                      f"(frames {start_frame}-{end_frame}, {expected_distance:.2f}m / {total_distance:.2f}m total)")
            else:
                self.expected_rc_delta = None

        try:
            for frame_id in tqdm(frame_ids, desc="Evaluating"):
                # Process frame
                control, ego_state, plan_traj, world_traj = self.process_frame(
                    env, frame_id, prev_velocity, prev_heading
                )
                
                # Optionally override with logged ego action for warmup frames
                control = self._maybe_override_with_log_action(control, frame_id, env)
                # Apply control
                obs, reward, done, truncated, info = env.step(control)

                # --- SCORING: Only score frames >= score_start_frame ---
                if frame_id >= self.score_start_frame:
                    # Capture route completion baseline at score start frame
                    if self.route_completion_baseline is None:
                        self.route_completion_baseline = info.get('route_completion', 0.0)
                        print(f"[INFO] Score calculation started at frame {frame_id} (route_completion baseline: {self.route_completion_baseline:.4f})")

                    # --- OPEN LOOP SPECIFIC SCORING ---
                    if self.eval_mode == "open_loop":
                        # Special transform for EPDMS scoring in open-loop mode
                        # 1. Swap x and y columns (index 0 and 1)
                        plan_traj_ego = plan_traj[:, [1, 0]]

                        # 2. Transform to global using rotation matrix
                        c, s = np.cos(ego_state['heading']), np.sin(ego_state['heading'])
                        R = np.array([[c, -s], [s, c]])
                        plan_traj_global_epdms = (R @ plan_traj_ego.T).T + ego_state['position'][:2]

                        # 3. Score frame
                        metrics = self.epdms_scorer.score_frame(plan_traj_global_epdms, int(frame_id))
                        metrics['token'] = frame_id
                        self.epdms_results_openloop.append(metrics)

                    # --- CLOSED LOOP SPECIFIC SCORING ---
                    if self.eval_mode == "closed_loop":
                        self.stats_manager.update_per_step(env.agent, info, frame_id)

                        # EPDMS per-frame scoring using LIVE simulation state
                        if self.epdms_scorer is not None:
                            # Score frame using current env state (no predicted trajectory needed)
                            metrics = self.epdms_scorer.score_frame_live(int(frame_id))
                            metrics['token'] = frame_id
                            self.epdms_results_closedloop.append(metrics)

                # Update for next iteration
                prev_velocity = ego_state['velocity']
                prev_heading = ego_state['heading']
                # Store final ego position for replan summary visualization
                self._final_ego_position = ego_state['position'].copy()

                # Check termination
                if (done or truncated) and not self.has_terminated:
                    self.has_terminated = True
                    print(f"Episode terminated at frame {frame_id}")
                    break

        finally:
            # Finalize the last plan's consumed count in planning history
            if len(self.planning_history) > 0:
                prev_entry = self.planning_history[-1]
                last_plan_frame = prev_entry[0]
                # Calculate how many waypoints were consumed since the last replan
                frames_since_last_replan = frame_id - last_plan_frame + 1
                consumed = min(frames_since_last_replan, len(prev_entry[1]) if prev_entry[1] is not None else 0)
                self.planning_history[-1] = (prev_entry[0], prev_entry[1], prev_entry[2], prev_entry[3], consumed, prev_entry[5])

            # Save final statistics
            print("Saving results...")

            # --- FINALIZATION: OPEN LOOP ---
            if self.eval_mode == "open_loop":
                if self.epdms_results_openloop:
                    # Create DataFrame
                    df = pd.DataFrame(self.epdms_results_openloop)
                    final_cols = [
                        "token", "valid", "score", 
                        "no_at_fault_collisions", "drivable_area_compliance", 
                        "driving_direction_compliance", "traffic_light_compliance",
                        "ego_progress", "time_to_collision_within_bound", 
                        "lane_keeping", "history_comfort", "extended_comfort"
                    ]
                    df_final = df[final_cols]
                    df_valid = df_final[df_final['valid'] == True]

                    if not df_valid.empty:
                        # Select only numeric columns for averaging
                        numeric_cols = df_valid.select_dtypes(include=[np.number]).columns
                        mean_values = df_valid[numeric_cols].mean()
                        summary_row = pd.DataFrame([mean_values])
                        summary_row['token'] = 'AVERAGE'
                        summary_row['valid'] = True 
                        summary_row = summary_row.reindex(columns=df_final.columns)
                        df_with_summary = pd.concat([df_final, summary_row], ignore_index=True)
                        
                        print("\n=== FINAL OPEN LOOP RESULTS (AVERAGE OF VALID FRAMES) ===")
                        # Use a safe formatter that checks type before applying float formatting
                        print(summary_row.to_string(index=False, formatters={
                            col: lambda x: "{:,.4f}".format(x) if isinstance(x, (int, float, np.number)) else str(x)
                            for col in numeric_cols
                        }))
                    else:
                        print("\n[WARNING] No valid frames found to calculate average.")
                        df_with_summary = df_final

                    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
                    csv_path = self.output_dir / f"{timestamp}_openloop_epdms_results.csv"
                    df_with_summary.to_csv(csv_path, index=False)
                    print(f"Saved EPDMS metrics to {csv_path}")

            # --- FINALIZATION: CLOSED LOOP ---
            elif self.eval_mode == "closed_loop":
                # Compute route completion once (used by both stats_manager and EPDMS)
                last_info = info if 'info' in locals() else {}
                raw_route_completion = last_info.get('route_completion', 0.0)
                if self.route_completion_baseline is not None:
                    actual_rc_delta = max(0.0, raw_route_completion - self.route_completion_baseline)
                    # Normalize by expected RC delta if partial scenario evaluation
                    if self.expected_rc_delta is not None and self.expected_rc_delta > 0:
                        route_completion = min(1.0, actual_rc_delta / self.expected_rc_delta)
                        print(f"[INFO] Route completion: {route_completion:.4f} (actual delta {actual_rc_delta:.4f} / expected {self.expected_rc_delta:.4f})")
                    else:
                        route_completion = actual_rc_delta
                        print(f"[INFO] Route completion: {route_completion:.4f} (delta from baseline {self.route_completion_baseline:.4f})")
                else:
                    route_completion = raw_route_completion

                # Finalize stats manager (legacy scoring)
                from metadrive.constants import TerminationState
                if not self.has_terminated:
                    # Normal completion (reached end of eval frames)
                    last_info[TerminationState.SUCCESS] = True
                self.stats_manager.finalize_route(
                    env.agent, last_info, self.has_terminated, False,
                    route_completion_override=route_completion
                )

                if self.stats_manager is not None:
                    self.stats_manager.save_to_json()
                    print(f"Saved evaluation metrics to: {self.stats_manager.get_save_path()}")

                # Finalize EPDMS scorer for closed-loop
                if self.epdms_scorer is not None and self.epdms_results_closedloop:
                    df = pd.DataFrame(self.epdms_results_closedloop)
                    df_valid = df[df['valid'] == True]

                    if not df_valid.empty:
                        # Compute mean of per-frame EPDMS scores (without EP)
                        mean_epdms = df_valid['score'].mean()

                        # Final closed-loop score: mean(EPDMS_no_ep) × Route_Completion
                        final_score = mean_epdms * route_completion

                        # Also compute mean of individual metrics for reporting
                        mean_nc = df_valid['no_at_fault_collisions'].mean()
                        mean_dac = df_valid['drivable_area_compliance'].mean()
                        mean_ddc = df_valid['driving_direction_compliance'].mean()
                        mean_tlc = df_valid['traffic_light_compliance'].mean()
                        mean_ttc = df_valid['time_to_collision_within_bound'].mean()
                        mean_lk = df_valid['lane_keeping'].mean()
                        mean_hc = df_valid['history_comfort'].mean()
                        mean_ec = df_valid['extended_comfort'].mean()

                        print("\n=== CLOSED-LOOP EPDMS RESULTS ===")
                        print(f"  Per-frame metrics (averaged over {len(df_valid)} valid frames):")
                        print(f"    No At-Fault Collisions:      {mean_nc:.4f}")
                        print(f"    Drivable Area Compliance:    {mean_dac:.4f}")
                        print(f"    Driving Direction Compliance:{mean_ddc:.4f}")
                        print(f"    Traffic Light Compliance:    {mean_tlc:.4f}")
                        print(f"    Time-to-Collision:           {mean_ttc:.4f}")
                        print(f"    Lane Keeping:                {mean_lk:.4f}")
                        print(f"    History Comfort:             {mean_hc:.4f}")
                        print(f"    Extended Comfort:            {mean_ec:.4f}")
                        print(f"  Mean EPDMS (no EP):            {mean_epdms:.4f}")
                        print(f"  Route Completion:              {route_completion:.4f}")
                        print(f"  ----------------------------------------")
                        print(f"  FINAL SCORE: {mean_epdms:.4f} × {route_completion:.4f} = {final_score:.4f}")

                        # Save per-frame results to CSV
                        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
                        epdms_csv_path = self.output_dir / f"{timestamp}_closedloop_epdms_perframe.csv"
                        df.to_csv(epdms_csv_path, index=False)
                        print(f"  Saved per-frame EPDMS to {epdms_csv_path}")

                        # Save summary to CSV
                        summary_dict = {
                            'mean_no_at_fault_collisions': mean_nc,
                            'mean_drivable_area_compliance': mean_dac,
                            'mean_driving_direction_compliance': mean_ddc,
                            'mean_traffic_light_compliance': mean_tlc,
                            'mean_time_to_collision': mean_ttc,
                            'mean_lane_keeping': mean_lk,
                            'mean_history_comfort': mean_hc,
                            'mean_extended_comfort': mean_ec,
                            'mean_epdms_no_ep': mean_epdms,
                            'route_completion': route_completion,
                            'final_score': final_score,
                            'num_valid_frames': len(df_valid),
                            'num_total_frames': len(df),
                        }
                        summary_csv_path = self.output_dir / f"{timestamp}_closedloop_epdms_summary.csv"
                        pd.DataFrame([summary_dict]).to_csv(summary_csv_path, index=False)
                        print(f"  Saved EPDMS summary to {summary_csv_path}")
                    else:
                        print("\n[WARNING] No valid EPDMS frames found for closed-loop scoring.")

            # Generate GIFs from visualizations
            if self.enable_vis:
                self.generate_gif(frame_ids, "topdown_visualization.gif", "topdown.png")
                self.generate_gif(frame_ids, "segmentation_visualization.gif", "segmentation_vis.png")
                self.generate_gif(frame_ids, "occupancy_visualization.gif", "occupancy_vis.png")
                self.generate_gif(frame_ids, "bev_embed_visualization.gif", "bev_embed_vis.png")
                # self.generate_gif(frame_ids, "third_persom.gif", "cam_third_person.jpg")

                # Generate replan summary (single image showing all replans with top-k coverage)
                if hasattr(self, '_final_ego_position') and self._final_ego_position is not None:
                    self.render_replan_summary(env, self._final_ego_position)

                # Generate DiffusionDriveV2-specific candidate visualization GIFs
                if type(self.model_adapter).__name__ == 'DiffusionDriveV2Adapter':
                    self.generate_gif(frame_ids, "topdown_topk_visualization.gif", "topdown_topk.png")
                    self.generate_gif(frame_ids, "topdown_finegrained_visualization.gif", "topdown_finegrained.png")
                    self.generate_gif(frame_ids, "topdown_coarse_visualization.gif", "topdown_coarse.png")

            if self.env_manager is not None:
                self.env_manager.close()

        print(f"Evaluation complete! Results saved to: {self.output_dir}")
        print("="*60)

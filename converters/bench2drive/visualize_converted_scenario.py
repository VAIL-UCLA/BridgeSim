#!/usr/bin/env python3
"""
Visualization script to verify Bench2Drive to Metadrive conversion using MetaDrive.
Supports both 3D and 2D top-down rendering modes.
"""

import argparse
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.scenario.utils import get_number_of_scenarios
    METADRIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MetaDrive not available: {e}")
    METADRIVE_AVAILABLE = False

class ScenarioVisualizer:
    """Visualizer for converted Bench2Drive scenarios"""
    
    def __init__(self, scenario_path: str):
        self.scenario_dir = Path(scenario_path).resolve()
        if not self.scenario_dir.exists():
            raise ValueError(f"Scenario directory does not exist: {scenario_path}")
        
        if not self.scenario_dir.is_dir():
            raise ValueError(f"Path is not a directory: {scenario_path}")
        
        # Check for required files
        self.dataset_mapping_path = self.scenario_dir / "dataset_mapping.pkl"
        self.dataset_summary_path = self.scenario_dir / "dataset_summary.pkl"
        
        if not self.dataset_mapping_path.exists():
            raise ValueError(f"dataset_mapping.pkl not found in {scenario_path}")
        if not self.dataset_summary_path.exists():
            raise ValueError(f"dataset_summary.pkl not found in {scenario_path}")
    
    def load_scenario_info(self):
        """Load scenario information"""
        with open(self.dataset_mapping_path, 'rb') as f:
            self.dataset_mapping = pickle.load(f)
        
        with open(self.dataset_summary_path, 'rb') as f:
            self.dataset_summary = pickle.load(f)
        
        logger.info(f"Found {len(self.dataset_mapping)} scenarios")
        for filename, summary in list(self.dataset_summary.items())[:3]:
            logger.info(f"Scenario: {filename}")
            logger.info(f"  - Objects: {summary['number_summary']['num_objects']}")
            logger.info(f"  - Moving objects: {summary['number_summary']['num_moving_objects']}")
            logger.info(f"  - Track length: {summary['track_length']}")
            logger.info(f"  - SDC ID: {summary['sdc_id']}")
    
    
    def visualize_with_metadrive(self, max_scenarios: int = 1, render_frames: int = None, record_video: bool = False, video_output: str = None, render_2d: bool = False, zoom_2d: float = 50.0):
        """Visualize scenarios using MetaDrive with 3D or 2D rendering"""
        if not METADRIVE_AVAILABLE:
            raise ImportError("MetaDrive is not available. Please install it to use visualization.")
        
        database_path = str(self.scenario_dir.absolute())
        num_scenarios = get_number_of_scenarios(database_path)
        num_scenarios = min(num_scenarios, max_scenarios)
        
        render_mode = "2D top-down" if render_2d else "3D"
        logger.info(f"Visualizing {num_scenarios} scenarios with MetaDrive ({render_mode})")
        if render_2d:
            # Calculate visible area based on screen_size / scaling
            visible_area = 800 / zoom_2d  # Default screen_size is 800x800
            logger.info(f"2D scaling: {zoom_2d} pixels/meter (visible area: ~{visible_area:.1f}m × {visible_area:.1f}m)")
        if record_video:
            logger.info(f"Recording video to: {video_output or 'scenario_recording.mp4'}")
        
        # Configure environment for 2D or 3D rendering
        config = {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_logo": False,
            "show_fps": True,
            "log_level": logging.INFO,
            "num_scenarios": num_scenarios,
            "horizon": 1000,
            "vehicle_config": {
                "show_navi_mark": True,
                "show_line_to_dest": False,
                "show_dest_mark": False,
                "no_wheel_friction": True,
            },
            "data_directory": database_path,
        }
        
        # Configure for 2D or 3D rendering
        if render_2d:
            config.update({
                "render_pipeline": False,
                "show_interface": False,
                "interface_panel": [],
            })
        else:
            config.update({
                "render_pipeline": True,
                "show_interface": False,
                "interface_panel": [],
                "vehicle_config": {
                    **config["vehicle_config"],
                    "lidar": {"num_lasers": 120, "distance": 50, "num_others": 4},
                    "lane_line_detector": {"num_lasers": 12, "distance": 50},
                    "side_detector": {"num_lasers": 160, "distance": 50}
                }
            })
        
        env = ScenarioEnv(config)
        
        try:
            # Initialize video recording if requested
            frames_for_video = []
            
            for scenario_idx in range(num_scenarios):
                logger.info(f"Visualizing scenario {scenario_idx}")
                env.reset(seed=scenario_idx)
                
                # Get actual scenario length if render_frames not specified
                scenario_length = env.engine.data_manager.current_scenario_length
                actual_render_frames = render_frames if render_frames is not None else scenario_length
                logger.info(f"Scenario {scenario_idx} length: {scenario_length} frames, rendering: {actual_render_frames} frames")
                
                for frame_idx in range(actual_render_frames):
                    env.step([0, 0])  # No input, just replay
                    
                    # Configure render parameters based on mode
                    if render_2d:
                        # 2D top-down rendering with proper scaling control
                        frame = env.render(
                            mode="top_down",
                            window=not record_video,  # Show window if not recording
                            screen_record=record_video,
                            target_agent_heading_up=True,
                            semantic_map=True,  # Use semantic colors to differentiate object types
                            scaling=zoom_2d,  # pixels per meter - higher = closer zoom
                            film_size=(5000, 5000),  # Smaller film for closer zoom
                            screen_size=(800, 800),  # Window/output size
                            num_stack=1,  # Remove vehicle trails by showing only current position
                            history_smooth=0,  # No trajectory smoothing to minimize trails
                            # show_agent_name=True,  # DISABLED: This function is broken in MetaDrive
                            draw_target_vehicle_trajectory=True,  # Draw ego trajectory to help identify it
                            text={
                                "scenario": f"{env.engine.global_seed + env.config.get('start_scenario_index', 0)}",
                                "frame": f"{frame_idx}/{actual_render_frames}",
                                "ego_speed": f"{env.vehicle.speed:.1f} m/s",
                                "ego_id": f"Ego: {env.config.get('sdc_id', 'N/A')}",
                                "mode": f"2D Top-Down (scale: {zoom_2d} px/m)"
                            }
                        )
                    else:
                        # 3D rendering (default)
                        frame = env.render(
                            mode="rgb_array",
                            film_size=(1200, 800),
                            window=not record_video,
                            screen_record=record_video
                        )
                    
                    # Collect frame for video recording
                    if record_video and frame is not None:
                        frames_for_video.append(frame)
                    
                    # Check if scenario is complete
                    if env.episode_step >= scenario_length:
                        logger.info(f"Scenario {scenario_idx} completed at frame {frame_idx}")
                        break
                
                logger.info(f"Finished scenario {scenario_idx}")
            
            # Save video if frames were collected
            if record_video and frames_for_video:
                self._save_video(frames_for_video, video_output or 'scenario_recording.mp4')
        
        finally:
            env.close()
    
    
    def _save_video(self, frames, output_path: str):
        """Save collected frames as video"""
        try:
            import cv2
            
            if not frames:
                logger.warning("No frames to save for video")
                return
            
            # Get dimensions from first frame
            first_frame = frames[0]
            if len(first_frame.shape) == 3:
                height, width = first_frame.shape[:2]
            else:
                height, width = first_frame.shape
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0  # 20 FPS for smooth playback
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info(f"Saving {len(frames)} frames to video: {output_path}")
            
            for frame in frames:
                # Convert frame format if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                video_writer.write(frame_bgr)
            
            video_writer.release()
            logger.info(f"Video saved successfully: {output_path}")
            
        except ImportError:
            logger.error("OpenCV (cv2) not available. Cannot save video. Install with: pip install opencv-python")
        except Exception as e:
            logger.error(f"Failed to save video: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize converted Bench2Drive scenarios using MetaDrive")
    parser.add_argument("scenario_path", help="Full path to directory containing converted scenario files")
    parser.add_argument("--max-scenarios", type=int, default=1,
                       help="Maximum scenarios to visualize")
    parser.add_argument("--render-frames", type=int, default=None,
                       help="Number of frames to render per scenario (default: all frames in scenario)")
    parser.add_argument("--record-video", action="store_true", 
                       help="Record MetaDrive visualization as video")
    parser.add_argument("--video-output", help="Output path for recorded video (default: scenario_recording.mp4)")
    parser.add_argument("--2d", "--render-2d", action="store_true", dest="render_2d",
                       help="Use 2D top-down rendering instead of 3D")
    parser.add_argument("--zoom", type=float, default=50.0,
                       help="2D zoom scaling (pixels per meter, default: 50.0, higher = closer zoom)")
    
    args = parser.parse_args()
    
    visualizer = ScenarioVisualizer(args.scenario_path)
    visualizer.load_scenario_info()
    
    # Always use MetaDrive visualization
    visualizer.visualize_with_metadrive(
        max_scenarios=args.max_scenarios, 
        render_frames=args.render_frames, 
        record_video=args.record_video, 
        video_output=args.video_output,
        render_2d=args.render_2d,
        zoom_2d=args.zoom
    )

if __name__ == "__main__":
    main()
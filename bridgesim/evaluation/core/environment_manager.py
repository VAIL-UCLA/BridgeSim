"""
Environment Manager for MetaDrive scenario environments.
Handles traffic mode configuration and environment setup.
"""

from pathlib import Path
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.envs.scenario_env import ScenarioEnv


class TrafficMode:
    """Traffic mode constants."""
    NO_TRAFFIC = "no_traffic"
    LOG_REPLAY = "log_replay"
    IDM = "IDM"


class EnvironmentManager:
    """
    Manages MetaDrive environment setup with configurable traffic modes.

    Args:
        scenario_path: Path to the converted scenario directory
        traffic_mode: One of 'no_traffic', 'log_replay', 'IDM'
        render: Enable rendering for visualization
        image_on_cuda: Enable CUDA for image processing
        agent_policy: Policy class for the ego agent (default: EnvInputPolicy)
    """

    def __init__(self, scenario_path, traffic_mode="log_replay", render=False, image_on_cuda=True, agent_policy=EnvInputPolicy):
        self.scenario_path = Path(scenario_path)
        self.traffic_mode = traffic_mode
        self.render = render
        self.image_on_cuda = image_on_cuda
        self.agent_policy = agent_policy
        self.env = None

    def create_env(self):
        """Create and return a MetaDrive ScenarioEnv with the configured traffic mode."""
        config = {
            "use_render": False,  # Always use offscreen rendering (env.render with window=False)
            "image_on_cuda": self.image_on_cuda,
            "agent_policy": self.agent_policy,
            "num_scenarios": 1,
            "horizon": 1000,
            "data_directory": str(self.scenario_path.absolute()),

            # Traffic mode configuration
            "reactive_traffic": self._get_reactive_traffic_setting(),
            "no_traffic": (self.traffic_mode == TrafficMode.NO_TRAFFIC),

            # Camera sensor configuration
            "sensors": {
                "rgb_camera": (RGBCamera, 900, 900),
            },
            "image_observation": True,

            # Vehicle configuration
            "vehicle_config": {
                "image_source": "rgb_camera",
                "no_wheel_friction": False  # Enable friction for realistic control
            },

            # UI settings
            "show_interface": False,
            "show_logo": False,
            "show_fps": False,
            "window_size": (800, 600),
        }

        self.env = ScenarioEnv(config)
        return self.env

    def _get_reactive_traffic_setting(self):
        """Determine reactive_traffic setting based on traffic mode."""
        if self.traffic_mode == TrafficMode.NO_TRAFFIC:
            return False
        elif self.traffic_mode == TrafficMode.LOG_REPLAY:
            return False  # Log replay uses non-reactive traffic
        elif self.traffic_mode == TrafficMode.IDM:
            return True   # IDM uses reactive traffic
        else:
            raise ValueError(f"Unknown traffic mode: {self.traffic_mode}")

    def reset(self, seed=0):
        """Reset the environment."""
        if self.env is None:
            self.create_env()
        return self.env.reset(seed=seed)

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        """Context manager entry."""
        self.create_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
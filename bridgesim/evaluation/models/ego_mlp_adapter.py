"""
Model adapter for EgoStatusMLP model from bridgesim.modelzoo.navsim.

EgoStatusMLP is a blind baseline that only uses ego vehicle state
(velocity, acceleration, driving command) without any sensor input.
"""

import torch
import numpy as np
from typing import Dict, Any

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD


class EgoStatusMLPAdapter(BaseModelAdapter):
    """
    Adapter for EgoStatusMLP model from bridgesim.modelzoo.navsim.

    This is a blind baseline that ignores all sensors and only uses
    ego vehicle state (velocity, acceleration, driving command).
    """

    def __init__(self, checkpoint_path: str, hidden_dim: int = 512, **kwargs):
        """
        Initialize EgoStatusMLP adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            hidden_dim: Hidden layer dimension (default: 512)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_poses = 8  # 4 seconds at 0.5s interval

    def load_model(self):
        """Load EgoStatusMLP model from checkpoint."""
        print("Loading EgoStatusMLP model...")

        # Build the MLP architecture
        # Input: 8 (velocity(2) + acceleration(2) + driving_command(4))
        # Output: num_poses * 3 (x, y, heading for each pose)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.num_poses * 3),
        )

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefix (agent._mlp. -> empty)
        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._mlp.', '')
            clean_sd[new_key] = v

        self.model.load_state_dict(clean_sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

        print("EgoStatusMLP model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """EgoStatusMLP doesn't use cameras (blind model)."""
        return {}

    def _get_driving_command_onehot(self, command: int) -> np.ndarray:
        """Convert driving command to one-hot encoding using NAVSIM_CMD_MAPPING."""
        return NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD).copy()

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """
        Prepare input for EgoStatusMLP model.

        Only uses ego state, ignores images entirely.
        """
        # Get velocity in local frame
        velocity = ego_state['velocity'][:2]  # (vx, vy) in global frame
        heading = ego_state['heading']

        # Rotate velocity to local frame
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity  # (vx_local, vy_local)

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if np.linalg.norm(vel_local) < 0.5:
            vel_local[0] = KICKOFF_SPEED

        # Get acceleration in local frame
        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0])

        # Get driving command one-hot
        command = ego_state.get('command', 1)  # Default to forward
        cmd_onehot = self._get_driving_command_onehot(command)

        # Combine: [velocity(2), acceleration(2), driving_command(4)] = 8
        ego_status = np.concatenate([vel_local, acc_local, cmd_onehot]).astype(np.float32)

        # Convert to tensor
        ego_status_tensor = torch.from_numpy(ego_status).unsqueeze(0).to(self.device)

        return {"ego_status": ego_status_tensor}

    def run_inference(self, model_input: Any) -> Any:
        """Run EgoStatusMLP model inference."""
        with torch.no_grad():
            ego_status = model_input["ego_status"]
            poses = self.model(ego_status)
            # Reshape to (batch, num_poses, 3)
            trajectory = poses.reshape(-1, self.num_poses, 3)
        return {"trajectory": trajectory}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse EgoStatusMLP output."""
        # Extract trajectory: (num_poses, 3) -> (x, y, heading)
        trajectory = model_output["trajectory"][0].cpu().numpy()

        # Swap columns: model outputs [forward, lateral] but evaluator expects [lateral, forward]
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])

        return {'trajectory': traj_swapped}

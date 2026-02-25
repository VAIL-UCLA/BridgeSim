"""
Model adapter for DiffusionDrive model from bridgesim.modelzoo.navsim.
DiffusionDrive uses diffusion-based trajectory prediction.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict


from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS


class DiffusionDriveAdapter(BaseModelAdapter):
    """
    Adapter for DiffusionDrive model from bridgesim.modelzoo.navsim.

    DiffusionDrive uses diffusion-based trajectory generation with TransFuser backbone.
    """

    def __init__(self, checkpoint_path: str, plan_anchor_path: str = None,
                 scorer=None, num_groups: int = 1, bev_calibrator=None, **kwargs):
        """
        Initialize DiffusionDrive adapter.

        Args:
            checkpoint_path: Path to checkpoint (.ckpt file)
            plan_anchor_path: Path to plan anchor numpy file
            scorer: Optional BaseTrajectoryScorer instance for candidate selection.
                    When set, uses forward_inference_scaling instead of regular forward.
            num_groups: Number of groups for trajectory candidate generation.
                        Total candidates = num_groups * 20. Only used when scorer is set.
            bev_calibrator: Optional TransfuserBEVCalibrator for domain adaptation.
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        self.plan_anchor_path = plan_anchor_path
        self.scorer = scorer
        self.num_groups = num_groups
        self.bev_calibrator = bev_calibrator
        self._current_frame_id = 0

    def load_model(self):
        """Load DiffusionDrive model from checkpoint."""
        print("Loading DiffusionDrive model...")

        # Initialize config
        self.config = TransfuserConfig()

        # Override plan_anchor_path if provided
        if self.plan_anchor_path:
            self.config.plan_anchor_path = self.plan_anchor_path

        # Initialize model
        self.model = V2TransfuserModel(self.config)

        # Load checkpoint
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefixes if trained with Lightning/DDP
        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"[DD LOAD] Missing keys: {len(missing_keys)}")
            print(f"[DD LOAD] Sample missing: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[DD LOAD] Unexpected keys: {len(unexpected_keys)}")
            print(f"[DD LOAD] Sample unexpected: {unexpected_keys[:5]}")

        self.model.to(self.device)
        self.model.eval()

        # Attach BEV calibrator if provided
        if self.bev_calibrator is not None:
            print("Loading BEV calibrator for DiffusionDrive...")
            self.bev_calibrator.load_model()
            self.model.bev_calibrator = self.bev_calibrator

        print("DiffusionDrive model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """DiffusionDrive uses 3 cameras (left, front, right) stitched together."""
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0')}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for DiffusionDrive model.
        DiffusionDrive expects a stitched panoramic image (left + front + right) resized to 1024x256.
        """
        # Get individual camera images
        cam_l0 = images_dict.get('CAM_L0')
        cam_f0 = images_dict.get('CAM_F0')
        cam_r0 = images_dict.get('CAM_R0')

        # Create dummy images if missing
        dummy_h, dummy_w = 1080, 1920
        if cam_l0 is None:
            cam_l0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_f0 is None:
            cam_f0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_r0 is None:
            cam_r0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)

        # Crop images to ensure proper stitching (matching navsim transfuser_features.py)
        h, w = cam_f0.shape[:2]
        crop_tb = int(28 * h / 1080)  # Scale crop based on actual image height
        crop_lr = int(416 * w / 1920)  # Scale crop based on actual image width

        # Crop: [top:bottom, left:right]
        l0_cropped = cam_l0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_l0[:, crop_lr:-crop_lr]
        f0_cropped = cam_f0[crop_tb:-crop_tb] if crop_tb > 0 else cam_f0
        r0_cropped = cam_r0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_r0[:, crop_lr:-crop_lr]

        # Stitch images horizontally: left + front + right
        stitched_image = np.concatenate([l0_cropped, f0_cropped, r0_cropped], axis=1)

        # Resize to expected size (1024, 256)
        resized_image = cv2.resize(stitched_image, (1024, 256))

        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()

        # Normalize (RGB values 0-255 to 0-1)
        tensor_image = tensor_image / 255.0

        return tensor_image

    def _create_lidar_bev(self) -> torch.Tensor:
        """Create dummy LiDAR BEV representation."""
        lidar_bev = torch.zeros(
            self.config.lidar_seq_len,
            self.config.lidar_resolution_height,
            self.config.lidar_resolution_width,
            dtype=torch.float32
        )
        return lidar_bev

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        """
        Create status feature tensor.
        Format: [command(4), velocity(2), acceleration(2)]
        """
        # Command one-hot
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        # Velocity in local frame
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if np.linalg.norm(vel_local) < 0.5:
            vel_local[0] = KICKOFF_SPEED

        # Acceleration (if available)
        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0])

        # Combine: [cmd(4), vel(2), acc(2)] = 8
        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for DiffusionDrive model."""
        self._current_frame_id = frame_id

        # 1. Preprocess images
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)

        # 2. Create LiDAR BEV
        lidar_feature = self._create_lidar_bev().unsqueeze(0).to(self.device)

        # 3. Get status feature
        command = ego_state.get('command', 3)
        status_feature = self._get_status_feature(ego_state, command).unsqueeze(0).to(self.device)

        inputs = {
            "camera_feature": camera_feature,
            "lidar_feature": lidar_feature,
            "status_feature": status_feature,
        }

        return inputs

    def run_inference(self, model_input: Any) -> Any:
        """Run DiffusionDrive model inference."""
        use_bev_calibrator = self.bev_calibrator is not None
        with torch.no_grad():
            if self.scorer is not None:
                output = self.model.forward_inference_scaling(
                    model_input, num_groups=self.num_groups,
                    use_bev_calibrator=use_bev_calibrator,
                )
            else:
                output = self.model(model_input, targets=None,
                                    use_bev_calibrator=use_bev_calibrator)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse DiffusionDrive output."""
        if self.scorer is not None:
            # Use scorer to select best trajectory from candidates
            result = self.scorer.select_best(
                model_output,
                ego_state=ego_state,
                frame_idx=self._current_frame_id,
            )
            trajectory = result["trajectory"][0].cpu().numpy()  # (T, 3)
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            parsed = {
                'trajectory': traj_swapped,
                'best_idx': result["best_idx"][0].item(),
                'num_candidates': model_output["all_candidates"].shape[1],
            }

            # Include all candidates and scores for visualization
            all_cands = model_output["all_candidates"][0].cpu().numpy()  # (N, 8, 3)
            # Swap columns per candidate: [forward, lateral] -> [lateral, forward]
            cands_swapped = np.stack([
                np.column_stack([c[:, 1], c[:, 0]]) for c in all_cands
            ])  # (N, 8, 2)
            parsed['trajectory_coarse'] = cands_swapped
            parsed['coarse_scores'] = result["scores"][0].cpu().numpy()  # (N,)

            return parsed

        # Existing behavior: extract trajectory from output
        trajectory = model_output["trajectory"][0].cpu().numpy()  # (T, 3) -> (x, y, heading)

        print(f"[DD DEBUG] Raw trajectory shape: {trajectory.shape}")
        print(f"[DD DEBUG] Raw trajectory first 3 points: {trajectory[:3]}")

        # Swap columns: model outputs [forward, lateral] but evaluator expects [lateral, forward]
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])

        print(f"[DD DEBUG] Swapped trajectory first 3 points: {traj_swapped[:3]}")

        return {'trajectory': traj_swapped}

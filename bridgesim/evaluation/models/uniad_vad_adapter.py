"""
Model adapter for UniAD and VAD models.
Both models use similar mmcv-based architecture.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from pyquaternion import Quaternion

# Bench2Drive model imports from modelzoo
# Add bench2drive to path for plugin loading
_bench2drive_root = Path(__file__).resolve().parent.parent.parent / "modelzoo" / "bench2drive"
sys.path.insert(0, str(_bench2drive_root))
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter


# Sensor extrinsics (from uniad_b2d_agent.py / vad_b2d_agent.py)
LIDAR2IMG = {
    'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK': np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
}

LIDAR2CAM = {
    'CAM_FRONT': np.array([[1., 0., 0., 0.], [0., 0., -1., -0.24], [0., 1., 0., -1.19], [0., 0., 0., 1.]]),
    'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331], [0., 0., -1., -0.24], [-0.81915204, 0.57357644, 0., -0.82909407], [0., 0., 0., 1.]]),
    'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331], [0., 0., -1., -0.24], [0.81915204, 0.57357644, 0., -0.82909407], [0., 0., 0., 1.]]),
    'CAM_BACK': np.array([[-1., 0., 0., 0.], [0., 0., -1., -0.24], [0., -1., 0., -1.61], [0., 0., 0., 1.]]),
    'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956], [0., 0., -1., -0.24], [-0.93969262, -0.34202014, 0., -0.49288953], [0., 0., 0., 1.]]),
    'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956], [0., 0., -1., -0.24], [0.93969262, -0.34202014, 0., -0.49288953], [0., 0., 0., 1.]])
}

LIDAR2EGO = np.array([[0., 1., 0., -0.39], [-1., 0., 0., 0.], [0., 0., 1., 1.84], [0., 0., 0., 1.]])

CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


class UniADVADAdapter(BaseModelAdapter):
    """
    Adapter for UniAD and VAD models.

    These models use the same mmcv-based architecture and inference pipeline.
    The main difference is in output format (handled in parse_output).
    """

    def __init__(self, checkpoint_path: str, config_path: str, model_type: str = "uniad",
                 bev_calibrator=None, **kwargs):
        """
        Initialize UniAD/VAD adapter.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
            config_path: Path to config (.py file)
            model_type: "uniad" or "vad"
            bev_calibrator: Optional BEVCalibrator instance for domain adaptation
        """
        super().__init__(checkpoint_path, config_path, **kwargs)
        self.model_type = model_type
        self.pipeline = None
        self.cfg = None
        self.bev_calibrator = bev_calibrator

    def load_model(self):
        """Load UniAD or VAD model."""
        print(f"Loading {self.model_type.upper()} model...")
        self.cfg = Config.fromfile(self.config_path)

        # UniAD-specific config
        if 'motion_head' in self.cfg.model:
            # Convert relative path to absolute path based on modelzoo location
            anchor_path = self.cfg.model['motion_head']['anchor_info_path']
            self.cfg.model['motion_head']['anchor_info_path'] = str(_bench2drive_root / anchor_path)

        # Load plugin if specified
        if hasattr(self.cfg, 'plugin'):
            if self.cfg.plugin:
                import importlib
                if hasattr(self.cfg, 'plugin_dir'):
                    plugin_dir = self.cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(f"Loading plugin: {_module_path}")
                    importlib.import_module(_module_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build and load model
        self.model = build_model(self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, self.checkpoint_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.eval()

        # Build inference pipeline (skip image loading)
        inference_only_pipeline_cfg = []
        for pipeline_cfg in self.cfg.inference_only_pipeline:
            if pipeline_cfg["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                inference_only_pipeline_cfg.append(pipeline_cfg)
        self.pipeline = Compose(inference_only_pipeline_cfg)

        # Load BEV calibrator if provided
        if self.bev_calibrator is not None:
            print("[UniADVADAdapter] Loading BEV calibrator...")
            self.bev_calibrator.load_model()
            print("[UniADVADAdapter] BEV calibrator loaded successfully")

            # Inject BEV calibrator into model so it can be used in forward pass
            self.model.bev_calibrator = self.bev_calibrator
            print("[UniADVADAdapter] BEV calibrator injected into model")

        print(f"{self.model_type.upper()} model loaded successfully.")

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for UniAD/VAD model."""
        results = {}
        results['lidar2img'] = np.stack([LIDAR2IMG[cam] for cam in CAM_NAMES], axis=0)
        results['lidar2cam'] = np.stack([LIDAR2CAM[cam] for cam in CAM_NAMES], axis=0)
        results['img'] = [images[cam] for cam in CAM_NAMES]

        results['folder'] = ' '
        results['scene_token'] = ' '
        results['frame_idx'] = int(frame_id)
        results['timestamp'] = int(frame_id) / 20.0
        results['box_type_3d'], _ = get_box_type('LiDAR')

        # Build can_bus
        ego_theta = ego_state['heading']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        can_bus = np.zeros(18)
        can_bus[0] = ego_state['position'][0]
        can_bus[1] = ego_state['position'][1]
        can_bus[2] = ego_state['position'][2] if len(ego_state['position']) > 2 else 0.0
        can_bus[3:7] = rotation
        can_bus[7] = np.linalg.norm(ego_state['velocity'])
        can_bus[10:13] = ego_state['acceleration']
        can_bus[13:16] = -ego_state['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        results['can_bus'] = can_bus

        # Command
        command = ego_state['command']
        if command < 0:
            command = 4
        results['command'] = command
        results['ego_fut_cmd'] = np.array([command], dtype=np.int64)

        # Ego2world transform
        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]
        lidar2global = ego2world @ LIDAR2EGO
        results['l2g_r_mat'] = lidar2global[0:3, 0:3]
        results['l2g_t'] = lidar2global[0:3, 3]

        # Image shapes
        stacked_imgs = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape

        # Run inference pipeline
        results = self.pipeline(results)

        return results

    def run_inference(self, model_input: Any) -> Any:
        """Run model inference with optional BEV calibration."""
        # Collate into batch form
        input_data_batch = mm_collate_to_batch_form([model_input], samples_per_gpu=1)

        # Move to GPU
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to("cuda")

        # Run inference
        if self.bev_calibrator is None:
            # Standard inference without calibration
            with torch.no_grad():
                output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
            return output_data_batch[0]

        # Apply BEV calibration if enabled
        else:
            with torch.no_grad():
                output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True, use_bev_calibrator=True)
            return output_data_batch[0]

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse UniAD/VAD output."""
        # UniAD format
        if 'planning' in model_output:
            plan_traj = model_output['planning']['result_planning']['sdc_traj'][0].detach().cpu().numpy()

        # VAD format (top-level)
        elif 'ego_fut_preds' in model_output:
            ego_fut_preds = model_output['ego_fut_preds']
            ego_fut_cmd = ego_state['command']
            plan_traj_idx = min(ego_fut_cmd, ego_fut_preds.shape[0] - 1)
            plan_traj = ego_fut_preds[plan_traj_idx].cpu().numpy()
            plan_traj = np.cumsum(plan_traj, axis=0)

        # VAD format (inside pts_bbox)
        elif 'pts_bbox' in model_output and isinstance(model_output['pts_bbox'], dict):
            pts_bbox = model_output['pts_bbox']
            if 'ego_fut_preds' in pts_bbox:
                ego_fut_preds = pts_bbox['ego_fut_preds']

                # Extract command
                ego_fut_cmd_container = pts_bbox['ego_fut_cmd']
                if hasattr(ego_fut_cmd_container, 'data'):
                    ego_fut_cmd_val = ego_fut_cmd_container.data
                else:
                    ego_fut_cmd_val = ego_fut_cmd_container

                if torch.is_tensor(ego_fut_cmd_val):
                    ego_fut_cmd_val = ego_fut_cmd_val.cpu().item() if ego_fut_cmd_val.numel() == 1 else ego_fut_cmd_val.cpu().numpy()[0]
                elif isinstance(ego_fut_cmd_val, np.ndarray):
                    ego_fut_cmd_val = int(ego_fut_cmd_val.flat[0])
                else:
                    ego_fut_cmd_val = int(ego_fut_cmd_val)

                plan_traj_idx = min(ego_fut_cmd_val, ego_fut_preds.shape[0] - 1)
                plan_traj = ego_fut_preds[plan_traj_idx].cpu().numpy()
                plan_traj = np.cumsum(plan_traj, axis=0)
            else:
                raise KeyError(f"Cannot find ego_fut_preds in pts_bbox. Available keys: {pts_bbox.keys()}")
        else:
            raise KeyError(f"Cannot find planning output. Available keys: {model_output.keys()}")

        return {'trajectory': plan_traj}

    def get_trajectory_time_horizon(self) -> float:
        """UniAD/VAD predicts 6 waypoints at 0.5s intervals = 3.0s horizon."""
        return 3.0

    def save_intermediate_outputs(self, model_output: Any, output_path: str):
        """Save intermediate model outputs."""
        output_path = Path(output_path)

        # Save intermediate features if available
        if hasattr(self.model, '_last_bev_embed_test'):
            torch.save(self.model._last_bev_embed_test.cpu(), output_path / "bev_embed.pth")
        if hasattr(self.model, '_last_seg_output_test'):
            torch.save(self.model._last_seg_output_test, output_path / "seg_output.pth")
        if hasattr(self.model, '_last_motion_output_test'):
            torch.save(self.model._last_motion_output_test, output_path / "motion_output.pth")
        if hasattr(self.model, '_last_occ_output_test'):
            torch.save(self.model._last_occ_output_test, output_path / "occ_output.pth")
        if hasattr(self.model, '_last_track_output_test'):
            torch.save(self.model._last_track_output_test, output_path / "track_output.pth")
        if hasattr(self.model, '_last_planning_output_test'):
            torch.save(self.model._last_planning_output_test, output_path / "planning_output.pth")

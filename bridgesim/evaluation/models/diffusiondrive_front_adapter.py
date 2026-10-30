"""
Front-camera-only variant of DiffusionDriveAdapter.

The front-only checkpoint expects a single CAM_F0 view at
(width=960, height=512) with the TransFuser latent path enabled
(``config.latent=True`` replaces the LiDAR input with a learned latent
parameter inside the backbone). See unified_evaluator's
DiffusionDriveFrontOnlyAgent for the upstream feature builder this mirrors.
"""

from typing import Any, Dict

import cv2
import numpy as np
import torch

from bridgesim.evaluation.models.diffusiondrive_adapter import DiffusionDriveAdapter
from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from bridgesim.modelzoo.navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS


class DiffusionDriveFrontAdapter(DiffusionDriveAdapter):
    """DiffusionDrive variant trained with only the front camera (cam_f0)."""

    def load_model(self):
        print("Loading DiffusionDrive (front-only) model...")

        self.config = TransfuserConfig()
        self.config.latent = True
        self.config.camera_width = 960
        self.config.camera_height = 512
        self.config.img_vert_anchors = 16
        self.config.img_horz_anchors = 30
        if self.plan_anchor_path:
            self.config.plan_anchor_path = self.plan_anchor_path

        self.model = V2TransfuserModel(self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"[DD-Front LOAD] Missing keys: {len(missing_keys)} (sample: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"[DD-Front LOAD] Unexpected keys: {len(unexpected_keys)} (sample: {unexpected_keys[:5]})")

        self.model.to(self.device)
        self.model.eval()

        if self.bev_calibrator is not None:
            print("Loading BEV calibrator for DiffusionDrive (front-only)...")
            self.bev_calibrator.load_model()
            self.model.bev_calibrator = self.bev_calibrator

        print("DiffusionDrive (front-only) model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_THIRD_PERSON')}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        cam_f0 = images_dict.get('CAM_F0')
        if cam_f0 is None:
            cam_f0 = np.zeros((1080, 1920, 3), dtype=np.uint8)

        h = cam_f0.shape[0]
        crop_tb = int(28 * h / 1080)
        f0_cropped = cam_f0[crop_tb:-crop_tb] if crop_tb > 0 else cam_f0

        resized = cv2.resize(f0_cropped, (self.config.camera_width, self.config.camera_height))

        tensor_image = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        return tensor_image

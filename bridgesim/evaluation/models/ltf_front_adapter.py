"""
Front-camera-only variant of LTFAdapter (Latent TransFuser).

The front-only checkpoint expects a single CAM_F0 view at
(width=960, height=512). LTF already enables the latent path
(``config.latent=True``); the only deltas vs the multi-camera LTF
adapter are the input resolution, the single-camera feature builder,
and the patch-grid sizes that match the new image shape.
"""

from typing import Any, Dict

import cv2
import numpy as np
import torch

from bridgesim.evaluation.models.ltf_adapter import LTFAdapter
from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_config import TransfuserConfig
from bridgesim.modelzoo.navsim.agents.transfuser.transfuser_model import TransfuserModel
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class LTFFrontAdapter(LTFAdapter):
    """LTF variant trained with only the front camera (cam_f0)."""

    def load_model(self):
        print("Loading LTF (front-only) model...")

        self.config = TransfuserConfig()
        self.config.latent = True
        self.config.camera_width = 960
        self.config.camera_height = 512
        self.config.img_vert_anchors = 16
        self.config.img_horz_anchors = 30
        self.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

        self.model = TransfuserModel(self.trajectory_sampling, self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"[LTF-Front LOAD] Missing keys: {len(missing_keys)} (sample: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"[LTF-Front LOAD] Unexpected keys: {len(unexpected_keys)} (sample: {unexpected_keys[:5]})")

        self.model.to(self.device)
        self.model.eval()

        print("LTF (front-only) model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0',)}

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

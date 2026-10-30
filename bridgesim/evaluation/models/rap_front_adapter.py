"""
Front-camera-only variant of RAPAdapter.

The original RAP-DINO checkpoint was trained with the canonical 4-cam
layout (cam_b0, cam_f0, cam_l0, cam_r0); for the front-only variant
the per-camera ``cams_embeds`` parameter is sliced down to just
cam_f0 at load time.
"""

from typing import Any, Dict

import cv2
import numpy as np
import torch

from bridgesim.evaluation.models.rap_adapter import (
    RAPAdapter,
    IMG_SCALE,
    IMG_MEAN,
    IMG_STD,
)
from bridgesim.modelzoo.navsim.agents.rap_dino.navsim_config import (
    CANONICAL_CAM_ORDER,
    RAPConfig,
)
from bridgesim.modelzoo.navsim.agents.rap_dino.rap_model import RAPModel
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS


# Front-only inference camera list.
FRONT_CAM_ORDER = ['CAM_F0']
FRONT_CAMERAS = ['cam_f0']


class RAPFrontAdapter(RAPAdapter):
    """RAP variant that consumes only the front camera (cam_f0)."""

    def load_model(self):
        print(f"Loading RAP (front-only) model (image_source={self.image_source})...")

        self.config = RAPConfig()
        self.config.b2d = False
        self.config.num_poses = 10
        self.config.trajectory_sampling.num_poses = 10
        self.config.trajectory_sampling.time_horizon = 5.0
        self.config.cameras = list(FRONT_CAMERAS)

        self.model = RAPModel(self.config).to(self.device)
        self.model.progress = 0.0
        self.model.batch_size = 1

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # Strip prefixes if trained with Lightning/DDP
        clean_sd = {k.replace('agent._rap_model.', '').replace('_rap_model.', ''): v
                   for k, v in state_dict.items()}

        # The checkpoint has 4-camera cams_embeds ([4, embed_dims]); keep cam_f0.
        cams_embeds_key = "_backbone.cams_embeds"
        if cams_embeds_key in clean_sd:
            pretrained_embed = clean_sd[cams_embeds_key]
            target_embed = self.model._backbone.cams_embeds
            if pretrained_embed.shape != target_embed.shape:
                if pretrained_embed.shape[0] == len(CANONICAL_CAM_ORDER):
                    try:
                        indices = [CANONICAL_CAM_ORDER.index(c) for c in self.config.cameras]
                        sliced = pretrained_embed[indices]
                        assert sliced.shape == target_embed.shape, (
                            f"Sliced cams_embeds shape {sliced.shape} != target {target_embed.shape}"
                        )
                        clean_sd[cams_embeds_key] = sliced
                        print(
                            f"[RAP-Front] Sliced pretrained cams_embeds "
                            f"{tuple(pretrained_embed.shape)} -> {tuple(sliced.shape)} "
                            f"for cameras={list(self.config.cameras)}"
                        )
                    except ValueError as e:
                        print(
                            f"[RAP-Front] Could not slice cams_embeds (cam not in canonical order): {e}. "
                            f"Dropping the pretrained cams_embeds."
                        )
                        clean_sd.pop(cams_embeds_key)
                else:
                    print(
                        f"[RAP-Front] Pretrained cams_embeds shape {tuple(pretrained_embed.shape)} "
                        f"does not match canonical 4-cam layout; dropping it."
                    )
                    clean_sd.pop(cams_embeds_key)

        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        if missing_keys:
            print(f"[RAP-Front LOAD] Missing keys: {len(missing_keys)} (sample: {missing_keys[:5]})")
        if unexpected_keys:
            print(f"[RAP-Front LOAD] Unexpected keys: {len(unexpected_keys)} (sample: {unexpected_keys[:5]})")

        self.model.eval()

        self._setup_calibration_features_front()

        if self.image_source == "rasterized_3d":
            from bridgesim.evaluation.models.rap_adapter import ScenarioRenderer
            self.renderer = ScenarioRenderer(camera_channel_list=FRONT_CAM_ORDER)
            print("ScenarioRenderer initialized for rasterized_3d mode (front-only).")

        print("RAP (front-only) model loaded successfully.")

    def _setup_calibration_features_front(self):
        """Precompute lidar2img matrices and image shapes for CAM_F0 only.

        CAM_F0 is rendered at 1920x1120 but RAP was trained on 1920x1080, so
        `_preprocess_images` crops 20 rows top and bottom; the principal point
        `cy` is shifted by -20 to match.
        """
        # Import here to keep the parent's monkey-patch ordering intact.
        from bridgesim.evaluation.models.rap_adapter import camera_params

        lidar2imgs = []
        img_shapes = []

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = IMG_SCALE
        S[1, 1] = IMG_SCALE

        for cam_name in FRONT_CAM_ORDER:
            params = camera_params[cam_name]

            s2l = np.eye(4, dtype=np.float32)
            s2l[:3, :3] = params['sensor2lidar_rotation']
            s2l[:3, 3] = params['sensor2lidar_translation']

            l2s = np.linalg.inv(s2l)

            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = params['intrinsics']
            # Account for the [20:-20] crop in _preprocess_images
            K[1, 2] = K[1, 2] - 20.0

            l2i = S @ K @ l2s
            lidar2imgs.append(l2i)

            # Image shape after the crop, IMG_SCALE resize and 32-multiple pad
            h = 448
            w = int(1920 * IMG_SCALE)
            img_shapes.append((h, w, 3))

        self.lidar2img_tensor = torch.from_numpy(np.stack(lidar2imgs)).float().unsqueeze(0).to(self.device)
        self.img_shape_tensor = torch.from_numpy(np.stack(img_shapes)).float().unsqueeze(0).to(self.device)

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Return CAM_F0 extrinsics in the raw NAVSIM convention (x=fwd, y=left),
        as DrivoR does; the evaluator converts them to the MetaDrive camera pose."""
        if self.image_source == "metadrive":
            return {k: NAVSIM_CAM_CONFIGS[k] for k in FRONT_CAM_ORDER}
        else:
            return {}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Front-only preprocessing: CAM_F0, crop [20:-20] (1120 -> 1080 rows,
        as in RAP training), IMG_SCALE resize, normalize, pad to a 32-multiple.
        """
        processed_imgs = []

        for cam_name in FRONT_CAM_ORDER:
            img = images_dict.get(cam_name)
            if img is None:
                img = np.zeros((1120, 1920, 3), dtype=np.uint8)

            if img.shape[0] >= 40:
                img = img[20:-20]

            h, w = img.shape[:2]
            new_h, new_w = int(h * IMG_SCALE), int(w * IMG_SCALE)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            img = img.transpose(2, 0, 1)
            processed_imgs.append(img)

        img_tensor = torch.from_numpy(np.stack(processed_imgs)).float()

        img_tensor = (img_tensor - IMG_MEAN.to(img_tensor.device)) / IMG_STD.to(img_tensor.device)

        _, _, h, w = img_tensor.shape
        pad_h = int(np.ceil(h / 32)) * 32
        pad_w = int(np.ceil(w / 32)) * 32
        if pad_h != h or pad_w != w:
            padding = torch.nn.ZeroPad2d((0, pad_w - w, 0, pad_h - h))
            img_tensor = padding(img_tensor)

        return img_tensor

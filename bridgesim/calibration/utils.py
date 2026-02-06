from __future__ import annotations

import numpy as np
import torch

try:
    import mmcv
except ImportError:  # pragma: no cover - optional dependency
    mmcv = None

# Constants match RAP bev_feature_build normalization.
BEV_MEAN = (123.675, 116.28, 103.53)
BEV_STD = (58.395, 57.12, 57.375)


def rap_preprocess_image(img: np.ndarray) -> torch.Tensor:
    """
    Mirror RAP bev_feature_build preprocessing for a single image:
    - convert to float32
    - normalize with RAP mean/std (no channel swap)
    - scale by 0.4
    - pad to the nearest size divisible by 32
    - return CHW torch tensor
    Reference: ./BridgeSim/RAP/navsim/agents/rap_dino/rap_features.py
    """
    if mmcv is None:
        raise ImportError("mmcv is required for RAP preprocessing but is not installed.")

    arr = img.astype(np.float32, copy=False)
    mean = np.array(BEV_MEAN, dtype=np.float32)
    std = np.array(BEV_STD, dtype=np.float32)
    arr = mmcv.imnormalize(arr, mean, std, to_rgb=False)

    new_w = int(arr.shape[1] * 0.4)
    new_h = int(arr.shape[0] * 0.4)
    arr = mmcv.imresize(arr, (new_w, new_h), return_scale=False)

    arr = mmcv.impad_to_multiple(arr, 32, pad_val=0)
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))  # CHW
    return torch.from_numpy(arr)

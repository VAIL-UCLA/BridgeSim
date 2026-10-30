# Third-party code and weights used by the `dreamstream` branch

The `dreamstream` branch adds front-camera-only variants of several NAVSIM
policies. Their model code under `bridgesim/modelzoo/navsim/agents/` is adapted
from the original projects and keeps their licenses:

| Component | Path | Upstream | License |
|---|---|---|---|
| SparseDriveV2 (incl. deformable-aggregation CUDA op) | `modelzoo/navsim/agents/sparsedrive/` | [swc-17/SparseDriveV2](https://github.com/swc-17/SparseDriveV2) | Apache-2.0 |
| RAP-DINO (front-camera slicing added) | `modelzoo/navsim/agents/rap_dino/` | [vita-epfl/RAP](https://github.com/vita-epfl/RAP) | Apache-2.0 |
| DiffusionDrive | `modelzoo/navsim/agents/diffusiondrive/` | [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive) | MIT, Copyright (c) 2024 HUST Vision Lab |
| DiffusionDriveV2 | `modelzoo/navsim/agents/diffusiondrivev2/` | [hustvl/DiffusionDriveV2](https://github.com/hustvl/DiffusionDriveV2) | MIT, Copyright (c) 2024 HUST Vision Lab |
| DrivoR | `modelzoo/navsim/agents/drivoR/` | [valeoai/DrivoR](https://github.com/valeoai/DrivoR) | Apache-2.0 |
| LTF / TransFuser, NAVSIM common code | `modelzoo/navsim/agents/transfuser/`, `modelzoo/navsim/common/` | [autonomousvision/navsim](https://github.com/autonomousvision/navsim) | Apache-2.0 |

Modifications by the DreamStream authors: front-camera (CAM_F0) input paths,
checkpoint loading for front-camera models, and BridgeSim adapters
(`bridgesim/evaluation/models/*_front_adapter.py`).

## Weights (not included in this repository)

| Weights | Source | Notes |
|---|---|---|
| Front-camera policy checkpoints (DiffusionDrive, DiffusionDriveV2, DrivoR, LTF, SparseDriveV2) | [`UCLA-VAIL/DreamStream`](https://huggingface.co/UCLA-VAIL/DreamStream) `policy/` | Trained by the DreamStream authors on NAVSIM (OpenScene / nuPlan); non-commercial research use, following the dataset terms |
| RAP-DINO NAVSIM v1 | [`Lanl11/RAP_ckpts`](https://huggingface.co/Lanl11/RAP_ckpts) | Official RAP release; includes DINOv3 weights (Meta DINOv3 license) |
| DINOv3 ViT-H+/16 | [`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | Gated; Meta DINOv3 license |
| DINOv2 ViT-S/14 | [`timm/vit_small_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | Apache-2.0 |
| ResNet-34 | [`timm/resnet34.a1_in1k`](https://huggingface.co/timm/resnet34.a1_in1k) | Apache-2.0 |

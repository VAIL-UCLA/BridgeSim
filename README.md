# BridgeSim: Unveiling the OL-CL Gap in End-to-End Autonomous Driving

[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://vail-ucla.github.io/BridgeSim/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![huggingface](https://img.shields.io/badge/HuggingFace-Checkpoints-yellow?logo=huggingface)](https://huggingface.co/sethzhao506ucla/BridgeSim)

[Seth Z. Zhao*](https://sethzhao506.github.io)<sup>1</sup>, [Luobin Wang*](https://scholar.google.com/citations?user=rbmtcYsAAAAJ&hl=en)<sup>2</sup>, [Hongwei Ruan](https://www.linkedin.com/in/hongwei-ruan/)<sup>2</sup>, [Yuxin Bao](www.linkedin.com/in/rebecca-bao-c13752hz)<sup>1</sup>, [Yilan Chen](https://yilanchen6.github.io)<sup>2</sup>, [Ziyang Leng](https://scholar.google.com/citations?user=Lwz4be0AAAAJ&hl=en)<sup>1</sup>, [Abhijit Ravichandran](https://www.linkedin.com/in/rabhijit/)<sup>2</sup>, [Honglin He](https://dhlinv.github.io)<sup>1</sup>, [Zewei Zhou](https://scholar.google.com/citations?user=TzhyHbYAAAAJ&hl=zh-CN)<sup>1</sup>, [Xu Han](https://scholar.google.com/citations?user=Ndgk55IAAAAJ&hl=en)<sup>1</sup>, [Abhishek Peri](https://www.linkedin.com/in/abhishek-peri/)<sup>3</sup>, [Zhiyu Huang](https://mczhi.github.io)<sup>1</sup>, [Pranav Desai](https://www.linkedin.com/in/pndesai2/)<sup>3</sup>, [Henrik Christensen](https://scholar.google.com/citations?user=MA8rI0MAAAAJ&hl=en)<sup>2</sup>, [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/about/)<sup>1</sup>, [Bolei Zhou](https://boleizhou.github.io/)<sup>1</sup>†

<sup>1</sup>UCLA &nbsp;&nbsp; <sup>2</sup>UCSD &nbsp;&nbsp; <sup>3</sup>Qualcomm

\* Equal contribution &nbsp;&nbsp; † Corresponding author

<!-- TODO: add teaser image -->
<!-- ![teaser](assets/bridgesim_teaser.png) -->

BridgeSim is a cross-simulator closed-loop evaluation platform for end-to-end autonomous driving policies, built on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator. It supports evaluating models trained on NavSim and Bench2Drive across multiple real-world datasets (NavSim, Waymo, nuScenes, and more). BridgeSim provides a unified evaluation interface that bridges the gap between training-time datasets and deployment-time environments, enabling fair and reproducible benchmarking across diverse driving scenarios.

## News

- **`2026/04`**: BridgeSim paper and codebase release.

## ✅ Currently Supported Features

- [√] Closed-loop evaluation of NavSim models (DiffusionDrive, DiffusionDriveV2, LTF, TransFuser, DrivoR) on multiple datasets
- [√] Closed-loop evaluation of Bench2Drive models (UniAD, VAD) on multiple datasets
- [√] Closed-loop evaluation of RAP on multiple datasets
- [√] Scenario conversion from OpenScene / NavSim, Bench2Drive, nuScenes, and Waymo to ScenarioNet format
- [√] Open-loop and closed-loop evaluation modes
- [√] Configurable traffic modes: `no_traffic`, `log_replay`, `IDM`

## Data Preparation

### OpenScene / NavSim

```bash
python converters/openscene/convert_openscene_with_filter.py \
    --scene-filter /path/to/scene_filter.yaml \
    --input-dir /path/to/navsim_logs \
    --output-dir /path/to/output \
    --map-root /path/to/maps \
    --num-future-frames-extract 220 \
    --interpolate
```

| Option | Description |
|--------|-------------|
| `--scene-filter` | YAML file specifying which scenes to convert |
| `--input-dir` | Directory containing raw NavSim/OpenScene logs |
| `--output-dir` | Output directory for converted scenarios |
| `--map-root` | Directory containing nuPlan HD maps |
| `--num-future-frames-extract` | Number of future frames to extract |
| `--interpolate` | Interpolate from 2Hz to 10Hz |

### Bench2Drive

```bash
python converters/bench2drive/convert_bench2drive.py \
    /path/to/scenario.tar.gz \
    --hd-map /path/to/Town_HD_map.npz \
    --output-dir /path/to/output
```

### nuScenes

```bash
python -m converters.nuscenes.convert_nuscenes \
    --dataroot /path/to/nuScenes \
    --database_path /path/to/output \
    --split v1.0-mini \
    --num_workers 8
```

### Waymo

```bash
python -m converters.waymo.convert_waymo \
    --raw_data_path /path/to/waymo/tfrecords \
    --database_path /path/to/output \
    --num_workers 8
```

## Installation

### Step 1: Base Docker Image

All model environments are tested against:

```bash
docker pull robinwangucsd/metabench:latest
```

### Step 2: Clone Repository

```bash
git clone https://github.com/VAIL-UCLA/BridgeSim.git
cd BridgeSim
git clone https://github.com/motional/nuplan-devkit.git
```

### Step 3: Per-Model Environment Setup

Each model group requires a different conda environment.

| Model group | Conda env | Python |
|---|---|---|
| DiffusionDrive / DiffusionDriveV2 / LTF / TransFuser / DrivoR | `mdsn` | 3.9 |
| UniAD / VAD | `b2d` | 3.8 |
| RAP | `rap` | 3.9 |

#### NavSim models (DiffusionDrive, DiffusionDriveV2, LTF, TransFuser, DrivoR)

```bash
conda env create -f mdsn.yaml
conda activate mdsn

pip install -e nuplan-devkit/
pip install -e metadrive/.[cuda]
pip install -e .
```

> **Note (headless servers):** If you encounter OpenGL or `GLIBCXX_3.4.xx not found` errors, run:
> ```bash
> mkdir -p /usr/lib/dri
> ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so
> ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 $(conda info --base)/envs/mdsn/lib/libstdc++.so.6
> ```

> **Note (Panda3D model paths):** If Panda3D cannot find pedestrian or traffic cone models, register them manually (replace `<ENV>` with your conda env name, e.g. `mdsn`, `b2d`, or `rap`):
> ```bash
> echo "model-path /path/to/metadrive/metadrive/assets/models/pedestrian" >> /opt/conda/envs/<ENV>/lib/python3.9/site-packages/panda3d/etc/Config.prc
> echo "model-path /path/to/metadrive/metadrive/assets/models/traffic_cone" >> /opt/conda/envs/<ENV>/lib/python3.9/site-packages/panda3d/etc/Config.prc
> ```


#### Bench2Drive models (UniAD, VAD)

```bash
conda env create -f b2d.yml
conda activate b2d

pip install -e nuplan-devkit/
pip install -e metadrive/.[cuda]
pip install -e bridgesim/modelzoo/bench2drive/ --no-build-isolation
pip install -e .
```

> **Note (headless servers):** Same workarounds as above, replacing `mdsn` with `b2d` in the `ln -sf` path.

#### RAP

```bash
conda env create -f rap.yml
conda activate rap

pip install -e metadrive/.[cuda]
pip install -e navsim/
pip install -e .
```

> **Note (HuggingFace access):** RAP uses the `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` model from HuggingFace. You must log in and request access before running:
> 1. Request access at https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
> 2. Log in via CLI:
>    ```bash
>    huggingface-cli login --token hf_xxxxxxxx
>    ```

> **Note (headless servers):** Same workarounds as above, replacing `mdsn` with `rap` in the `ln -sf` path.

### Step 4: Download Checkpoints

Download all model checkpoints from HuggingFace:

```bash
huggingface-cli download sethzhao506ucla/BridgeSim --local-dir ckpts/BridgeSim
```

Expected structure:

```
ckpts/BridgeSim/
├── bench2drive/
│   ├── UniAD/
│   ├── VAD/
│   └── TCP/
└── navsimv2/
    ├── DiffusionDrive/
    ├── DiffusionDriveV2/
    ├── DrivoR/
    ├── LEAD_navsim/
    ├── RAP_DINO/
    ├── transfuser/
    ├── ltf/
    └── ego_status_mlp/
```

### Key evaluator options

| Option | Default | Description |
|--------|---------|-------------|
| `--replan-rate` | 1 | Steps between model inference calls |
| `--eval-frames` | full scenario | Number of frames to evaluate |
| `--ego-replay-frames` | 0 | Frames to replay log ego before model takes over |
| `--traffic-mode` | `log_replay` | `no_traffic`, `log_replay`, or `IDM` |
| `--trajectory-scorer` | None | Inference-time trajectory scorer for DiffusionDrive/V2 |
| `--eval-mode` | `closed_loop` | `closed_loop` or `open_loop` |
| `--image-source` | `metadrive` | Image source for RAP: `metadrive`, `rasterized_3d` |

## Supported Models

| Model | Type | `--model-type` |
|-------|------|----------------|
| UniAD | Bench2Drive | `uniad` |
| VAD | Bench2Drive | `vad` |
| TCP | Bench2Drive | `tcp` |
| TransFuser | NavSim v2 | `transfuser` |
| Latent TransFuser | NavSim v2 | `ltf` |
| DiffusionDrive | NavSim v2 | `diffusiondrive` |
| DiffusionDriveV2 | NavSim v2 | `diffusiondrivev2` |
| DrivoR | NavSim v2 | `drivor` |
| RAP | NavSim v2 | `rap` |
| LEAD (NavSim) | NavSim v2 | `lead_navsim` |
| EgoMLP | NavSim v2 | `ego_mlp` |
| OpenPilot | comma.ai | `openpilot` |
| Alpamayo-R1 | Nvidia | `alpamayo_r1` |

## Acknowledgement

The codebase is built upon [MetaDrive](https://github.com/metadriverse/metadrive) and [ScenarioNet](https://github.com/metadriverse/scenarionet). We also thank the authors of [Bench2Drive](https://arxiv.org/abs/2406.03877), [NavSim](https://github.com/autonomousvision/navsim), [UniAD](https://github.com/OpenDriveLab/UniAD), [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), and [ADV-BMT](https://github.com/Yuxin45/Adv-BMT) for releasing their codebases.

## Citation

If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Third-party components:
- `metadrive/` — Apache 2.0 ([metadriverse/metadrive](https://github.com/metadriverse/metadrive))
- `scenarionet/` — Apache 2.0 ([metadriverse/scenarionet](https://github.com/metadriverse/scenarionet))
- `nuplan-devkit/` — Apache 2.0 ([motional/nuplan-devkit](https://github.com/motional/nuplan-devkit))
- `ADV-BMT/` — see [Yuxin45/Adv-BMT](https://github.com/Yuxin45/Adv-BMT)
- Model weights are subject to their respective original licenses

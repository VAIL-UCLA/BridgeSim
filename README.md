# BridgeSim: Closed-Loop Evaluation for End-to-End Autonomous Driving

BridgeSim is a closed-loop cross-dataset evaluation platform for end-to-end autonomous driving models, built on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator. It supports evaluating models trained on NavSim and Bench2Drive across multiple real-world datasets (NavSim, Waymo, nuScenes, and more).

## Table of Contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Scenario Conversion](#scenario-conversion)
- [Evaluation](#evaluation)
- [Supported Models](#supported-models)
- [References](#references)

---

## Installation

### 1. Base Docker Image

All model environments are tested against:

```bash
docker pull robinwangucsd/metabench:latest
```

### 2. Clone Repository

```bash
git clone https://github.com/VAIL-UCLA/BridgeSim.git
cd BridgeSim
```

### 3. Per-Model Environment Setup

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

#### Bench2Drive models (UniAD, VAD)

```bash
conda env create -f b2d.yaml
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

> **Note (headless servers):** Same workarounds as above, replacing `mdsn` with `rap` in the `ln -sf` path.

---

## Checkpoints

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

---

## Scenario Conversion

Convert driving datasets to ScenarioNet format for MetaDrive evaluation.

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

---

## Evaluation

Run evaluation with `unified_evaluator.py` from the repository root. For batch evaluation over many scenarios see `scripts/evaluator/run_batch_eval.sh`.

### NavSim models (DiffusionDrive, DiffusionDriveV2, LTF, TransFuser, DrivoR)

```bash
python -m bridgesim.evaluation.unified_evaluator \
    --model-type diffusiondrive \
    --checkpoint ckpts/BridgeSim/navsimv2/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS \
    --plan-anchor-path ckpts/BridgeSim/navsimv2/DiffusionDrive/kmeans_navsim_traj_20.npy \
    --scenario-path /path/to/scenario \
    --output-dir /path/to/output \
    --replan-rate 10 \
    --eval-frames 80
```

### Bench2Drive models (UniAD, VAD)

```bash
python -m bridgesim.evaluation.unified_evaluator \
    --model-type uniad \
    --checkpoint ckpts/BridgeSim/bench2drive/UniAD/uniad_base_b2d.pth \
    --config bridgesim/modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
    --scenario-path /path/to/scenario \
    --output-dir /path/to/output \
    --replan-rate 10 \
    --eval-frames 80
```

### RAP

```bash
python -m bridgesim.evaluation.unified_evaluator \
    --model-type rap \
    --checkpoint ckpts/BridgeSim/navsimv2/RAP_DINO/RAP_DINO_navsimv2.ckpt \
    --image-source rasterized_3d \
    --scenario-path /path/to/scenario \
    --output-dir /path/to/output \
    --replan-rate 10 \
    --eval-frames 80
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--replan-rate` | 1 | Steps between model inference calls |
| `--eval-frames` | full scenario | Number of frames to evaluate |
| `--ego-replay-frames` | 0 | Frames to replay log ego before model takes over |
| `--traffic-mode` | `log_replay` | `no_traffic`, `log_replay`, or `IDM` |
| `--trajectory-scorer` | None | Inference-time trajectory scorer for DiffusionDrive/V2 |
| `--eval-mode` | `closed_loop` | `closed_loop` or `open_loop` |
| `--image-source` | `metadrive` | Image source for RAP: `metadrive`, `rasterized_3d` |

---

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

---

## References

- [Bench2Drive](https://arxiv.org/abs/2406.03877)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [MetaDrive](https://github.com/metadriverse/metadrive)
- [NavSim](https://github.com/autonomousvision/navsim)
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)
- [ADV-BMT](https://github.com/Yuxin45/Adv-BMT)

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Third-party components:
- `metadrive/` — Apache 2.0 ([metadriverse/metadrive](https://github.com/metadriverse/metadrive))
- `scenarionet/` — Apache 2.0 ([metadriverse/scenarionet](https://github.com/metadriverse/scenarionet))
- `nuplan-devkit/` — Apache 2.0 ([motional/nuplan-devkit](https://github.com/motional/nuplan-devkit))
- `ADV-BMT/` — see [Yuxin45/Adv-BMT](https://github.com/Yuxin45/Adv-BMT)
- Model weights are subject to their respective original licenses

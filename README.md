# BridgeSim: Unveiling the OL-CL Gap in End-to-End Autonomous Driving
[![website](https://img.shields.io/badge/Website-Explore%20Now-blueviolet?style=flat&logo=google-chrome)](https://vail-ucla.github.io/BridgeSim/)
<!-- [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2509.03704) -->

This is the official implementation of "BridgeSim: Unveiling the OL-CL Gap in End-to-End Autonomous Driving". This repository contains tools for deploying, evaluating, and converting scenarios for E2E driving models from Bench2Drive and NavSim families.

## Table of Contents

- [Installation](#installation)
- [Scenario Conversion](#scenario-conversion)
- [Evaluation](#evaluation)
- [Supported Models](#supported-models)

## Installation

### 1. Clone the Repository and Install MetaDrive

```bash
git clone https://github.com/VAIL-UCLA/BridgeSim.git
cd BridgeSim

# Install MetaDrive simulator
cd metadrive
pip install -e .
cd ..
```

### 2. Create Conda Environment

```bash
# Create the mdsn environment (for most models)
conda env create -f mdsn.yml
conda activate mdsn

# Install the bridgesim package
pip install -e .
```

### 3. Download Checkpoints

Download model checkpoints from Hugging Face:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the model zoo
huggingface-cli download sethzhao506ucla/BridgeSim --local-dir ckpts/BridgeSim
```

Expected checkpoint structure in `ckpts/BridgeSim/`:

```
ckpts/BridgeSim/
├── bench2drive/          # Bench2Drive trained models
│   ├── UniAD/
│   ├── VAD/
│   └── TCP/
└── navsimv2/             # NavSim v2 trained models
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

Convert driving scenarios to ScenarioNet format for evaluation in MetaDrive.

### OpenScene/NavSim Conversion
We have included filter folder under openscene converter, in the main paper NavHard closed-loop evaluations used the following setting:

```bash
# Convert OpenScene scenarios with scene filtering
python converters/openscene/convert_openscene_with_filter.py \
    --scene-filter /path/to/scene_filter.yaml \
    --input-dir /path/to/navsim_logs \
    --output-dir /path/to/output \
    --map-root /path/to/maps \
    --num-future-frames-extract 40 \ #future 20 seconds since navsim sampling is 2hz
    --interpolate
```

**Options:**
| Option | Description |
|--------|-------------|
| `--scene-filter` | YAML file specifying which scenes to convert |
| `--input-dir` | Directory containing raw NavSim/OpenScene logs |
| `--output-dir` | Output directory for converted scenarios |
| `--map-root` | Directory containing nuPlan HD maps |
| `--num-future-frames-extract` | Number of future frames to extract |
| `--interpolate` | Interpolate from 2Hz to 10Hz |

### Bench2Drive Conversion

```bash
python converters/bench2drive/convert_bench2drive.py \
    /path/to/scenario.tar.gz \
    --hd-map /path/to/Town_HD_map.npz \
    --output-dir /path/to/output
```

### nuScenes Conversion

```bash
# Using shell script
bash scripts/converter/convert_nuscenes.sh /path/to/nuScenes /path/to/output v1.0-mini 8

# Or directly with Python
python -m converters.nuscenes.convert_nuscenes \
    --dataroot /path/to/nuScenes \
    --database_path /path/to/output \
    --split v1.0-mini \
    --num_workers 8
```

**nuScenes Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--split` | v1.0-mini | Dataset split (v1.0-mini, v1.0-trainval, v1.0-test) |
| `--render` | - | Generate rendered images after conversion |
| `--map_radius` | 500 | Map extraction radius |

### Waymo Conversion

```bash
# Using shell script
bash scripts/converter/convert_waymo.sh /path/to/waymo/tfrecords /path/to/output 8

# Or directly with Python
python -m converters.waymo.convert_waymo \
    --raw_data_path /path/to/waymo/tfrecords \
    --database_path /path/to/output \
    --num_workers 8
```

**Waymo Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--version` | v1.2 | Waymo dataset version |
| `--start_file_index` | 0 | Start index for file selection |
| `--num_files` | None | Number of files to process (None = all) |

---

## Evaluation

All evaluation scripts should be run from the repository root.

### Single Scenario Evaluation

```bash
# Usage: bash scripts/evaluator/run_eval.sh [model_type] [scenario_path] [gpu_id]
bash scripts/evaluator/run_eval.sh transfuser /path/to/scenario 0
```

### Batch Evaluation

```bash
# Usage: bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root] [options]
bash scripts/evaluator/run_batch_eval.sh uniad /path/to/scenarios

# Resume interrupted evaluation
bash scripts/evaluator/run_batch_eval.sh uniad /path/to/scenarios --resume
```

**Batch Evaluation Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--resume` | - | Resume from previous run |
| `--replan-rate N` | 10 | Replan rate (steps between replanning) |
| `--sim-dt F` | 0.1 | Simulation timestep |
| `--ego-replay-frames N` | 20 | Number of ego replay frames before control |
| `--eval-frames N` | 200 | Number of frames to evaluate |
| `--scorer-type TYPE` | navsim | Scorer type: `legacy` or `navsim` |
| `--eval-mode MODE` | closed_loop | Evaluation mode: `closed_loop` or `open_loop` |
| `--enable-vis` | - | Enable visualization outputs |

### Output Structure

```
outputs/
├── batch_results.json           # Per-scenario results (for resume)
├── aggregated_results.json      # Aggregated statistics
├── logs/
│   └── <scenario>.log           # Individual scenario logs
└── <scenario>/
    └── evaluation_results.json  # Detailed results
```

---

## Supported Models

### Bench2Drive Models

| Model | Adapter |
|-------|---------|
| `uniad` | `bridgesim/evaluation/models/uniad_vad_adapter.py` |
| `vad` | `bridgesim/evaluation/models/uniad_vad_adapter.py` |
| `tcp` | `bridgesim/evaluation/models/tcp_adapter.py` |

### NavSim v2 Models

| Model | Adapter |
|-------|---------|
| `transfuser` | `bridgesim/evaluation/models/transfuser_adapter.py` |
| `ltf` | `bridgesim/evaluation/models/ltf_adapter.py` |
| `diffusiondrive` | `bridgesim/evaluation/models/diffusiondrive_adapter.py` |
| `diffusiondrivev2` | `bridgesim/evaluation/models/diffusiondrivev2_adapter.py` |
| `drivor` | `bridgesim/evaluation/models/drivor_adapter.py` |
| `rap` | `bridgesim/evaluation/models/rap_adapter.py` |
| `lead_navsim` | `bridgesim/evaluation/models/lead_navsim_adapter.py` |
| `ego_mlp` | `bridgesim/evaluation/models/ego_mlp_adapter.py` |
| `alpamayo_r1` | `bridgesim/evaluation/models/alpamayo_r1_adapter.py` |

---

## References

- [Bench2Drive](https://arxiv.org/abs/2406.03877)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [MetaDrive](https://github.com/metadriverse/metadrive)
- [NavSim](https://github.com/autonomousvision/navsim)

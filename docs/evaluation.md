# Evaluation Guide

Full evaluation documentation for all supported models.

## Evaluator Options

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

| Model | Type | `--model-type` | Conda env |
|-------|------|----------------|-----------|
| UniAD | Bench2Drive | `uniad` | `b2d` |
| VAD | Bench2Drive | `vad` | `b2d` |
| TCP | Bench2Drive | `tcp` | `b2d` |
| TransFuser | NavSim v2 | `transfuser` | `mdsn` |
| Latent TransFuser | NavSim v2 | `ltf` | `mdsn` |
| DiffusionDrive | NavSim v2 | `diffusiondrive` | `mdsn` |
| DiffusionDriveV2 | NavSim v2 | `diffusiondrivev2` | `mdsn` |
| DrivoR | NavSim v2 | `drivor` | `mdsn` |
| RAP | NavSim v2 | `rap` | `rap` |
| LEAD (NavSim) | NavSim v2 | `lead_navsim` | `mdsn` |
| EgoMLP | NavSim v2 | `ego_mlp` | `mdsn` |
| OpenPilot | comma.ai | `openpilot` | — |
| Alpamayo-R1 | Nvidia | `alpamayo_r1` | — |

## Single Scenario Evaluation

```bash
bash scripts/evaluator/run_eval.sh [model_type] [scenario_path] [gpu_id]
```

Or directly:

```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type transfuser \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/transfuser/transfuser.pth \
    --scenario-path /path/to/scenarios/YourScenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/transfuser_log_replay
```

## Batch Evaluation

```bash
bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root]

# Resume after interruption
bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root] --resume
```

Or directly:

```bash
cd bridgesim/evaluation

python batch_evaluator.py \
    --model-type transfuser \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/transfuser/transfuser.pth \
    --scenario-root /path/to/scenarios \
    --output-dir ../../outputs/batch_eval/transfuser_log_replay \
    --traffic-mode log_replay \
    --max-workers 1
```

## Model-Specific Commands

### NavSim Models (env: `mdsn`)

#### TransFuser
```bash
python unified_evaluator.py \
    --model-type transfuser \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/transfuser/transfuser.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/transfuser_log_replay
```

#### Latent TransFuser
```bash
python unified_evaluator.py \
    --model-type ltf \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/ltf/ltf.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/ltf_log_replay
```

#### DiffusionDrive
```bash
python unified_evaluator.py \
    --model-type diffusiondrive \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/DiffusionDrive/diffusiondrive.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/diffusiondrive_log_replay
```

#### DiffusionDriveV2
```bash
python unified_evaluator.py \
    --model-type diffusiondrivev2 \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/DiffusionDriveV2/diffusiondrivev2.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/diffusiondrivev2_log_replay
```

#### DrivoR
```bash
python unified_evaluator.py \
    --model-type drivor \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/DrivoR/drivor.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/drivor_log_replay
```

#### LEAD (NavSim)
```bash
python unified_evaluator.py \
    --model-type lead_navsim \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/LEAD_navsim/model_0060.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/lead_log_replay
```

### Bench2Drive Models (env: `b2d`)

#### UniAD
```bash
python unified_evaluator.py \
    --model-type uniad \
    --config ../modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/UniAD/uniad_base_b2d.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/uniad_log_replay
```

#### VAD
```bash
python unified_evaluator.py \
    --model-type vad \
    --config ../modelzoo/bench2drive/adzoo/vad/configs/VAD_base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/VAD/vad_b2d_base.pth \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/vad_log_replay
```

#### TCP
```bash
python unified_evaluator.py \
    --model-type tcp \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/TCP/tcp_b2d.ckpt \
    --planner-type learned \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/tcp_log_replay
```

### RAP (env: `rap`)

```bash
python unified_evaluator.py \
    --model-type rap \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/RAP_DINO/RAP_DINO_navsimv2.ckpt \
    --image-source metadrive \
    --scenario-path /path/to/scenario \
    --traffic-mode log_replay \
    --output-dir ../../outputs/rap_log_replay
```

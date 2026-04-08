# Evaluation Guide

Full evaluation documentation for all supported models.

## Evaluator Options

| Option | Default | Description |
|--------|---------|-------------|
| `--replan-rate` | 1 | Steps between model execution calls |
| `--eval-frames` | full scenario | Number of frames to evaluate |
| `--ego-replay-frames` | 0 | Frames to replay log ego before model takes over |
| `--traffic-mode` | `log_replay` | `no_traffic`, `log_replay`, or `IDM` |
| `--trajectory-scorer` | None | Inference-time trajectory scorer for each model |
| `--eval-mode` | `closed_loop` | `closed_loop` or `open_loop` |
| `--image-source` | `metadrive` | Default image source: `metadrive`; for RAP we provide `rasterized_3d` as options|

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
| OpenPilot | comma.ai | `openpilot` | `openpilot` |
| Alpamayo-R1 | Nvidia | `alpamayo_r1` | `openpilot` |

## Single Scenario Evaluation


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
cd bridgesim/evaluation

python batch_evaluator.py \
    --model-type transfuser \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/transfuser/transfuser.pth \
    --scenario-root /path/to/scenarios \
    --output-dir ../../outputs/batch_eval/transfuser_log_replay \
    --traffic-mode log_replay \
    --max-workers 1
```
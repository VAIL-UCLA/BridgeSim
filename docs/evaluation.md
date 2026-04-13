# Evaluation Guide

Full evaluation documentation for all supported models.

## Evaluator Options

### `unified_evaluator.py`

| Argument | Default | Description |
|---|---|---|
| `--model-type` | *(required)* | Model to evaluate. Choices: `uniad`, `vad`, `tcp`, `rap`, `lead`, `lead_navsim`, `drivor`, `transfuser`, `ltf`, `egomlp`, `diffusiondrive`, `diffusiondrivev2`, `openpilot`, `alpamayo_r1` |
| `--checkpoint` | *(required)* | Path to model checkpoint file |
| `--config` | `None` | Path to model config file (required for UniAD/VAD) |
| `--scenario-path` | *(required)* | Path to a single converted scenario directory |
| `--output-dir` | `./evaluation_outputs` | Directory to save evaluation outputs |
| `--traffic-mode` | `log_replay` | Traffic behavior: `no_traffic`, `log_replay` (replay logged agents), `IDM` (intelligent driver model) |
| `--eval-mode` | `closed_loop` | `closed_loop` (model drives the ego) or `open_loop` (ego follows ground truth log) |
| `--controller` | `pure_pursuit` | Low-level trajectory tracker: `pure_pursuit` or `pid` |
| `--replan-rate` | `1` | Run model inference every N simulation frames; cached waypoints are consumed between replans |
| `--sim-dt` | `0.1` | Simulation timestep in seconds (0.1 s = 10 Hz) |
| `--ego-replay-frames` | `0` | Number of initial frames to follow the log while still running model inference (warm-up) |
| `--eval-frames` | `None` | Number of frames to score after the replay warm-up ends; `None` = full scenario |
| `--score-start-frame` | `None` | First frame from which metrics are computed; defaults to `ego_replay_frames` |
| `--enable-vis` | off | Save visualization images and top-down views |
| `--save-perframe` / `--no-save-perframe` | on | Save / suppress per-frame numpy outputs (`planning_traj.npy`, etc.) |
| `--plan-anchor-path` | `None` | Path to plan anchor file (DiffusionDrive / V2 only) |
| `--trajectory-scorer` | `None` | Inference-time trajectory selection for DiffusionDrive/V2: `cls`, `learned`, `gt`, `tta` |
| `--num-groups` | model default | Number of candidate groups; total candidates = num_groups × 20 |
| `--num-proposals` | `None` | Truncate candidate list to the first N before scoring |
| `--v2-scorer-checkpoint` | `None` | V2 checkpoint for loading the learned scorer into a V1 model |
| `--enable-bev-calibrator` | off | Apply BEV flow-matching domain adaptation (MetaDrive → Bench2Drive) |
| `--bev-calibrator-checkpoint` | *(built-in path)* | Path to BEV calibrator `.ckpt` file |
| `--bev-sample-steps` | `50` | Euler sampling steps for the BEV flow matching model |
| `--enable-temporal-consistency` | off | Combine PDM scorer with temporal trajectory consistency (DiffusionDriveV2 only) |
| `--temporal-alpha` | `1.5` | Decay base for older trajectory predictions (range 1.0–3.0) |
| `--temporal-lambda` | `0.3` | Blend weight between PDM score and temporal consistency (0 = pure PDM, 1 = pure temporal) |
| `--temporal-max-history` | `8` | Maximum past predictions kept in the temporal buffer |
| `--temporal-sigma` | `5.0` | Position normalisation factor in metres for temporal consistency |
| `--consensus-temperature` | `1.0` | Softmax temperature when building the consensus trajectory |

### `batch_evaluator.py`

All arguments from `unified_evaluator.py` are supported. Additional batch-specific arguments:

| Argument | Default | Description |
|---|---|---|
| `--scenario-root` | *(required)* | Root directory containing all converted scenario subdirectories |
| `--max-workers` | `1` | Number of parallel worker processes (1 = sequential) |
| `--resume` | off | Skip scenarios that already have a completed result file |

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
| EgoMLP | NavSim v2 | `egomlp` | `mdsn` |
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
    --eval-mode closed_loop \
    --enable-bev-calibrator \
    --bev-calibrator-checkpoint ../../ckpts/BridgeSim/bev_calibrator/transfuser.ckpt \
    --bev-sample-steps 10 \
    --controller pure_pursuit \
    --replan-rate 5 \
    --ego-replay-frames 20 \
    --eval-frames 80 \
    --output-dir ../../outputs/transfuser_single \
    --enable-vis \
```

## Batch Evaluation

```bash
cd bridgesim/evaluation

python batch_evaluator.py \
    --model-type transfuser \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/transfuser/transfuser.pth \
    --scenario-root /path/to/scenarios \
    --traffic-mode log_replay \
    --eval-mode closed_loop \
    --enable-bev-calibrator \
    --bev-calibrator-checkpoint ../../ckpts/BridgeSim/bev_calibrator/transfuser.ckpt \
    --bev-sample-steps 10 \
    --controller pure_pursuit \
    --replan-rate 5 \
    --ego-replay-frames 20 \
    --eval-frames 80 \
    --max-workers 4 \
    --resume \
    --output-dir ../../outputs/batch_eval/transfuser \
```
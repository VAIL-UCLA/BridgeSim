# Unified Closed-Loop Evaluator

A unified evaluation system for autonomous driving models (UniAD, VAD, TCP, RAP, LEAD) on MetaDrive scenarios.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Single Scenario Evaluation](#single-scenario-evaluation)
  - [Batch Evaluation (All Scenarios)](#batch-evaluation-all-scenarios)
  - [Test Batch Evaluation (3 Scenarios)](#test-batch-evaluation-3-scenarios)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [1. Single Scenario Evaluation](#1-single-scenario-evaluation)
  - [2. Batch Evaluation](#2-batch-evaluation)
  - [3. Different Models](#3-different-models)
- [Output Format](#output-format)
- [Configuration Options](#configuration-options)
  - [Traffic Modes](#traffic-modes)
  - [Visualization](#visualization)
  - [Flow Matching](#flow-matching)
- [Consistency with Original Evaluators](#consistency-with-original-evaluators)
- [Adding New Models](#adding-new-models)
- [Troubleshooting](#troubleshooting)
  - [GPU Memory Issues](#gpu-memory-issues)
  - [Import Errors](#import-errors)
  - [Missing Dependencies](#missing-dependencies)
  - [Scenario Not Found](#scenario-not-found)
- [Performance](#performance)
- [Known Issues](#known-issues)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Style](#code-style)
  - [Contributing](#contributing)
- [References](#references)
- [Support](#support)

## Features

- ✅ **Multi-Model Support:** UniAD, VAD, TCP, RAP, LEAD
- ✅ **Consistent Evaluation:** Matches offline_evaluator_md_1.py behavior
- ✅ **Batch Processing:** Evaluate multiple scenarios automatically
- ✅ **Resume Capability:** Continue interrupted evaluations
- ✅ **Result Aggregation:** JSON output with complete statistics
- ✅ **Modular Design:** Easy to add new models via adapters
- ✅ **Self-Contained:** All dependencies bundled in utils/ for easy deployment

## Quick Start

### Single Scenario Evaluation

```bash
# From repository root
conda activate mdsn

# Run single scenario evaluation
bash scripts/evaluator/run_eval.sh [model_type] [scenario_path] [gpu_id]

# Example
bash scripts/evaluator/run_eval.sh transfuser /path/to/scenario 0
```

### Batch Evaluation (All Scenarios)

```bash
# From repository root
conda activate mdsn

# Run batch evaluation
bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root]

# Example
bash scripts/evaluator/run_batch_eval.sh uniad /path/to/scenarios

# Resume after interruption
bash scripts/evaluator/run_batch_eval.sh uniad /path/to/scenarios --resume
```

## Directory Structure

```
unified_closedloop_eval/
├── README.md                          # This file
├── BATCH_EVALUATION.md               # Detailed batch evaluation guide
├── CONSISTENCY_UPDATES.md            # Changes made for consistency
│
├── unified_evaluator.py              # Main single-scenario evaluator
├── batch_evaluator.py                # Batch evaluation script
│
├── core/                             # Core evaluation logic
│   ├── base_evaluator.py            # Base evaluator class
│   └── environment_manager.py        # Environment management
│
├── models/                           # Model adapters
│   ├── base_adapter.py              # Base adapter interface
│   ├── uniad_vad_adapter.py         # UniAD/VAD adapter
│   ├── tcp_adapter.py               # TCP adapter
│   ├── rap_adapter.py               # RAP adapter
│   └── lead_adapter.py              # LEAD adapter
│
├── utils/                            # Utility modules
│   ├── controller_md.py             # PID controller for vehicle control
│   └── statistics_manager_md.py     # Statistics manager for metrics
│
└── scripts/                          # Convenient scripts
    ├── run_eval.sh                  # Single scenario evaluation
    ├── run_batch_eval.sh            # Batch eval (shell)
    ├── run_batch_eval_python.sh     # Batch eval (Python, recommended)
    └── test_batch_eval.sh           # Test on 3 scenarios
```

## Usage

### 1. Single Scenario Evaluation

Evaluate a specific scenario with direct Python command:

```bash
# Navigate to the evaluation directory
cd bridgesim/evaluation

# Run evaluation with UniAD model
python unified_evaluator.py \
    --model-type uniad \
    --config ../modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/UniAD/uniad_base_b2d.pth \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/uniad_log_replay
```

**Available options:**
- `--model-type`: Model to use (uniad, vad, tcp, rap, lead)
- `--config`: Path to model config file (required for UniAD/VAD)
- `--checkpoint`: Path to model checkpoint file
- `--scenario-path`: Path to scenario directory
- `--traffic-mode`: Traffic mode (log_replay, no_traffic, IDM)
- `--output-dir`: Directory to save evaluation results
- `--enable-vis`: Enable visualization (optional)
- `--no-save-perframe`: Disable per-frame outputs to save disk space (optional)

### 2. Batch Evaluation

See [BATCH_EVALUATION.md](BATCH_EVALUATION.md) for detailed documentation.

**Quick start:**
```bash
# From repository root
bash scripts/evaluator/run_batch_eval.sh [model_type] [scenario_root]
```

**Or use Python directly:**
```bash
cd bridgesim/evaluation

python batch_evaluator.py \
    --model-type uniad \
    --config ../modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/UniAD/uniad_base_b2d.pth \
    --scenario-root /path/to/scenarios \
    --output-dir ../../outputs/batch_eval/uniad_log_replay \
    --traffic-mode log_replay \
    --max-workers 1
```

**Configuration:** Edit the script to change:
- `--model-type`: Model to use (uniad, vad, tcp, rap, lead)
- `--config`: Path to model config file
- `--checkpoint`: Path to model checkpoint
- `--scenario-root`: Root directory containing all scenarios
- `--output-dir`: Directory to save all evaluation results
- `--traffic-mode`: Traffic mode (log_replay, no_traffic, IDM)
- `--max-workers`: Number of parallel workers (default: 1)
- `--resume`: Resume from previous run (optional flag)

### 3. Different Models

The evaluator supports multiple models via the `--model-type` parameter:

#### UniAD
```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type uniad \
    --config ../modelzoo/bench2drive/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/UniAD/uniad_base_b2d.pth \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/uniad_log_replay
```

#### VAD
```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type vad \
    --config ../modelzoo/bench2drive/adzoo/vad/configs/VAD_base_e2e_b2d.py \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/VAD/vad_b2d_base.pth \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/vad_log_replay
```

#### TCP
```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type tcp \
    --checkpoint ../../ckpts/BridgeSim/bench2drive/TCP/tcp_b2d.ckpt \
    --planner-type learned \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/tcp_log_replay
```
**Note:** TCP supports `--planner-type learned` or `--planner-type reactive`

#### RAP
```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type rap \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/RAP_DINO/RAP_DINO_navsimv2.ckpt \
    --image-source rgb_camera \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/rap_log_replay
```
**Note:** RAP supports `--image-source rgb_camera` or `--image-source metadrive_camera`

#### LEAD
```bash
cd bridgesim/evaluation

python unified_evaluator.py \
    --model-type lead_navsim \
    --checkpoint ../../ckpts/BridgeSim/navsimv2/LEAD_navsim/model_0060.pth \
    --scenario-path /path/to/scenarios/YourScenarioName \
    --traffic-mode log_replay \
    --output-dir ../../outputs/lead_log_replay
```

## Output Format

After evaluation, each scenario produces:

```
output_dir/ScenarioName/
├── evaluation_results.json          # Statistics and metrics
├── 00000/                           # Per-frame outputs
│   ├── planning_traj.npy           # Planned trajectory
│   ├── bev_embed.pth               # BEV features
│   ├── seg_output.pth              # Segmentation
│   ├── occ_output.pth              # Occupancy
│   └── track_output.pth            # Tracking
├── 00001/
└── ...
```

**evaluation_results.json** contains:
- Infractions (collisions, traffic violations, etc.)
- Route completion metrics
- Scenario metadata

## Configuration Options

### Traffic Modes

- `no_traffic`: No other vehicles
- `log_replay`: Replay recorded traffic (default)
- `IDM`: Intelligent Driver Model traffic

### Visualization

Enable visualization (not implemented yet):
```bash
--enable-vis
```

### Flow Matching

Enable flow matching (not implemented yet):
```bash
--enable-flow-matching
```

## Consistency with Original Evaluators

This unified evaluator is **fully consistent** with `offline_evaluator_md_1.py`.

Key consistency points:
- ✅ Same control calculation (ego-frame transformation)
- ✅ Same PID controller behavior
- ✅ Same action format for MetaDrive
- ✅ Same termination handling
- ✅ Same statistics management

See [CONSISTENCY_UPDATES.md](CONSISTENCY_UPDATES.md) for details.

## Adding New Models

To add a new model:

1. **Create adapter** in `models/`:
   ```python
   from models.base_adapter import BaseModelAdapter

   class MyModelAdapter(BaseModelAdapter):
       def load_model(self):
           # Load your model
           pass

       def run_inference(self, model_input):
           # Run inference
           pass
   ```

2. **Register in evaluator:**
   Edit `unified_evaluator.py` to add your model type to `create_model_adapter()`

3. **Test:**
   ```bash
   python unified_evaluator.py \
       --model-type mymodel \
       --checkpoint path/to/checkpoint \
       --scenario-path path/to/scenario \
       --output-dir path/to/output
   ```

## Troubleshooting

### GPU Memory Issues

If you get OOM errors:
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=2
bash run_eval.sh

# Check GPU usage
nvidia-smi
```

### Import Errors

Make sure you're in the correct conda environment:
```bash
conda activate mdsn
```

And the paths are correct relative to the leaderboard directory.

### Missing Dependencies

Some models require specific dependencies:
- **UniAD/VAD**: mmcv, mmdet3d
- **RAP**: mmengine
- **All**: MetaDrive, torch, numpy, cv2

### Scenario Not Found

Ensure the scenario path points to the scenario directory (not the parent):
```
✓ Correct: /path/to/scenarios/Accident_Town03_Route101_Weather23
✗ Wrong:   /path/to/scenarios
```

## Performance

Typical performance on L40S GPU:
- **Single scenario:** 1-3 minutes (depends on length)
- **1000 scenarios:** 16-50 hours
- **GPU memory:** ~3-4 GB per evaluation

## Known Issues

1. **CUDA device selection with MetaDrive:**
   - MetaDrive cameras expect device 0
   - Use `CUDA_VISIBLE_DEVICES` to remap GPUs
   - Don't use multiple evaluations on same GPU simultaneously

2. **Path resolution:**
   - Some configs use relative paths
   - Run from the correct directory or use absolute paths

## Development

### Running Tests

```bash
# Test single scenario
cd scripts/
bash run_eval.sh

# Test batch (3 scenarios)
bash test_batch_eval.sh
```

### Code Style

- Follow existing patterns
- Add docstrings to new functions
- Keep consistency with offline_evaluator_md_1.py

### Contributing

To contribute:
1. Test your changes on multiple scenarios
2. Ensure consistency with original evaluators
3. Update documentation
4. Add examples

## References

- [Bench2Drive Paper](https://arxiv.org/abs/2406.03877)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [MetaDrive](https://github.com/metadriverse/metadrive)

## Support

For issues or questions:
1. Check the documentation (BATCH_EVALUATION.md, CONSISTENCY_UPDATES.md)
2. Review scenario logs in `output_dir/logs/`
3. Test with a single scenario first
4. Check GPU availability and memory

# DreamStream: Closed-Loop Evaluation with Generated Observations

This branch (`dreamstream`) adds the BridgeSim side of
[DreamStream](https://arxiv.org/abs/2609.26792): BridgeSim runs the physics and
traffic, renders a **traffic layout** for the front camera, and a DreamStream video-model
server turns the layouts into camera frames that the driving policy consumes.

```
policy ──plan──▶ BridgeSim (MetaDrive, IDM traffic) ──layouts──▶ DreamStream server
   ▲                                                                   │
   └──────────────────────── generated CAM_F0 frames ◀─────────────────┘
```

The video model (inference server) lives in the
[DreamStream repository](https://github.com/VAIL-UCLA/DreamStream) (`DreamStream/`
folder); this branch only contains the simulator and the client.

## What this branch adds

- `bridgesim/evaluation/core/dreamstream_evaluator.py`: DreamStream backend
  (`--use-world-model`). It renders a layout for every simulation frame and
  sends one block per replan to the server.
- `bridgesim/evaluation/core/batch_client.py`: HTTP client for the server.
- Front-camera-only (CAM_F0) policies: `drivor_front`, `diffusiondrive_front`,
  `diffusiondrivev2_front`, `ltf_front`, `rap_front`, `sparsedrivev2_front`.
- `scripts/dreamstream/run_navhard.sh`: batch evaluation.

## 1. Installation

Follow the main [README](../README.md#step-1-install) to create the BridgeSim
environment (MetaDrive, nuplan-devkit, `pip install -e .`). Then build the
CUDA op used by SparseDriveV2:

```bash
cd bridgesim/modelzoo/navsim/agents/sparsedrive/ops
python setup.py build_ext --inplace
cd -
```

## 2. Policy checkpoints

The front-camera checkpoints are hosted on Hugging Face. Download them into
`ckpts/DreamStream/policy` (the default location; set `DREAMSTREAM_POLICY_ROOT`
to use another directory):

```bash
# DreamStream front-camera policies + k-means anchors
hf download UCLA-VAIL/DreamStream --include "policy/*" --include "config.json" --local-dir ckpts/DreamStream

# RAP-DINO: the official NAVSIM v1 checkpoint (4-camera; sliced to CAM_F0 at load time)
hf download Lanl11/RAP_ckpts RAP_DINO_navsimv1.ckpt --local-dir ckpts/DreamStream/policy

# SparseDriveV2 image-backbone init (timm ResNet-34)
hf download timm/resnet34.a1_in1k pytorch_model.bin --local-dir /tmp/resnet34
cp /tmp/resnet34/pytorch_model.bin ckpts/DreamStream/policy/aux/sparsedrive/resnet34.bin
```

Expected layout:

```
ckpts/DreamStream/policy/
├── diffusiondrive_front_only_epoch_99_step_33300.ckpt
├── diffusiondrivev2_epoch19_step6660.ckpt
├── drivor_front_Nav2_10epochs.pth
├── ltf_epoch99_step_11100.ckpt
├── RAP_DINO_navsimv1.ckpt
├── sparsedrivev2_epoch_9_step_6650.ckpt
└── aux/
    ├── kmeans_navsim_traj_20.npy               # DiffusionDrive / V2 plan anchors
    └── sparsedrive/
        ├── resnet34.bin
        └── kmeans/{path_1024.npy, velocity_256.npy, trajectory_1024_256.npz}
```

## 3. Navhard scenarios

DreamStream's closed-loop benchmark (Navhard-Base) uses the 421 scenarios of
NAVSIM `navhard_two_stage`. Download the NAVSIM test logs, maps and sensor blobs
following the [NAVSIM instructions](https://github.com/autonomousvision/navsim/blob/main/docs/install.md),
then convert them:

```bash
python converters/openscene/convert_openscene_with_filter.py \
    --scene-filter converters/openscene/filter/navhard_two_stage.yaml \
    --input-dir /path/to/navsim_logs/test \
    --output-dir converted_scenarios/navhard \
    --map-root /path/to/maps \
    --num-future-frames-extract 40 \
    --interpolate
```

This yields 421 scenario directories (see
`converters/openscene/converted_list/navhard421.txt`).

The DreamStream evaluator reads the logged front-camera images; point `NAVSIM_DATA_ROOT` at the
camera root:

```bash
export NAVSIM_DATA_ROOT=/path/to/navsim/sensor_blobs/test
```

## 4. Start the DreamStream server

In the DreamStream repository (see its `DreamStream/README.md`):

```bash
python serve.py --num-gpus 1      # serves on http://127.0.0.1:8790
```

Each server keeps one session's KV
cache, so it can serve one scenario at a time; to evaluate scenarios in
parallel, start one server per worker.

## 5. Evaluate

Single scenario:

```bash
python bridgesim/evaluation/unified_evaluator.py \
    --model-type diffusiondrive_front \
    --checkpoint ckpts/DreamStream/policy/diffusiondrive_front_only_epoch_99_step_33300.ckpt \
    --plan-anchor-path ckpts/DreamStream/policy/aux/kmeans_navsim_traj_20.npy \
    --scenario-path converted_scenarios/navhard/<scenario> \
    --traffic-mode IDM --ego-replay-frames 20 --replan-rate 5 --eval-frames 80 \
    --use-world-model --server-url http://127.0.0.1:8790 \
    --output-dir outputs/dreamstream --enable-vis
```

All 421 Navhard scenarios with the paper protocol:

```bash
bash scripts/dreamstream/run_navhard.sh diffusiondrive_front converted_scenarios/navhard
# parallel, one server per worker:
bash scripts/dreamstream/run_navhard.sh diffusiondrive_front converted_scenarios/navhard \
    --max-workers 2 --server-urls http://127.0.0.1:8790,http://127.0.0.1:8791
```

### DreamStream-specific options

| Option | Default | Description |
|---|---|---|
| `--use-world-model` | off | Use DreamStream observations instead of MetaDrive RGB |
| `--server-url` | `http://127.0.0.1:8790` | DreamStream server (`unified_evaluator.py`) |
| `--server-urls` | `http://127.0.0.1:8790` | Comma-separated servers, one per worker (`batch_evaluator.py`) |
| `--dreamstream-prompt` | `a driving scene` | Text prompt sent to the server |
| `--first-frame-path` / `--first-frame-dir` | — | Custom anchor frame instead of the last ego-replay image (e.g. an edited weather variant); the batch form expects `<dir>/<scenario>/original_cam_f0.png` |
| `--seed` | 0 | Seeds Python/NumPy/PyTorch so stochastic policies are reproducible |

With `--enable-vis`, each scenario folder also contains `cam_f0_<scenario>.mp4`
(generated frames with the plan overlaid), `condition_<scenario>.mp4` (layouts)
and `bev_<scenario>.mp4`.

## Environment variables

| Variable | Default | Used for |
|---|---|---|
| `NAVSIM_DATA_ROOT` | — (required) | NAVSIM camera images (`sensor_blobs/test`) |
| `NUSCENES_DATA_ROOT` | — | nuScenes `samples/CAM_FRONT`, for nuScenes scenarios |
| `DREAMSTREAM_POLICY_ROOT` | `ckpts/DreamStream/policy` | Front-camera policy checkpoints |
| `BRIDGESIM_CKPT_ROOT` | `ckpts/BridgeSim` | Original BridgeSim checkpoints |

## Citation

```bibtex
@inproceedings{leng2026dreamstream,
  title     = {{DreamStream: Towards Policy-Oriented Generative Simulation for End-to-End Driving}},
  author    = {Leng, Ziyang and Mo, Sicheng and Zhao, Seth Z. and Cai, Haoyuan and Zeng, Yu and McAllister, Rowan and Zhou, Bolei},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

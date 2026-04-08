# Data Preparation

Scenario conversion instructions for all supported datasets.

## OpenScene / NavSim (full options)

```bash
python converters/openscene/convert_openscene_with_filter.py \
    --scene-filter /path/to/scene_filter.yaml \
    --input-dir /path/to/navsim_logs \
    --output-dir /path/to/output \
    --map-root /path/to/maps \
    --num-future-frames-extract <N> \
    --interpolate
```

| Option | Description |
|--------|-------------|
| `--scene-filter` | YAML file specifying which scenes to convert. Available filters: `converters/openscene/filter/navhard_two_stage.yaml`, `converters/openscene/filter/navtest.yaml` |
| `--input-dir` | Directory containing raw NavSim/OpenScene logs |
| `--output-dir` | Output directory for converted scenarios |
| `--map-root` | Directory containing nuPlan HD maps |
| `--num-future-frames-extract` | Number of future frames to extract |
| `--interpolate` | Interpolate from 2Hz to 10Hz |

## Bench2Drive

```bash
python converters/bench2drive/convert_bench2drive.py \
    /path/to/scenario.tar.gz \
    --hd-map /path/to/Town_HD_map.npz \
    --output-dir /path/to/output
```

## nuScenes

```bash
python -m converters.nuscenes.convert_nuscenes \
    --dataroot /path/to/nuScenes \
    --database_path /path/to/output \
    --split v1.0-mini \
    --num_workers 8
```

## Waymo

```bash
python -m converters.waymo.convert_waymo \
    --raw_data_path /path/to/waymo/tfrecords \
    --database_path /path/to/output \
    --num_workers 8
```

## ADV-BMT: Adversarial Scenario Generation

[ADV-BMT](https://github.com/Yuxin45/Adv-BMT) generates adversarial variants of converted scenarios by perturbing background agent trajectories. Run it after converting scenarios.

```bash
cd ADV-BMT
python scripts/ADV-BMT_dataset_generate_long_scenario.py \
    --dir /path/to/converted/scenarios \
    --save_dir /path/to/output \
    --TF_mode all_TF_except_adv \
    --num_mode 1 \
    --window_size 91 \
    --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--dir` | *(required)* | Input folder containing converted scenario `.pkl` files |
| `--save_dir` | *(required)* | Output folder for generated adversarial scenarios |
| `--TF_mode` | `all_TF_except_adv` | Which agents to apply Teacher Forcing to |
| `--num_mode` | `1` | Number of adversarial modes to generate per scenario |
| `--window_size` | `91` | Sliding window size in frames |
| `--seed` | `42` | Random seed for reproducibility |
| `--num_scenario` | all | Limit to the first N scenarios |
| `--start_idx` / `--end_idx` | — | Process a specific index range of scenarios |
| `--batch_id` / `--num_batches` | — | Split work across SLURM array jobs |
| `--parallel` / `--num_workers` | off / `1` | Enable multi-CPU parallel processing |

See `ADV-BMT/scripts/ADV-BMT_dataset_generate_long_scenario.sh` for examples of all processing modes.
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
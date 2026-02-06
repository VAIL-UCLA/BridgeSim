# Bench2Drive to Metadrive Converter

This repository contains tools to convert Bench2Drive scenarios to Metadrive format, focusing on vehicles, pedestrians, and non-ego vehicles.

## Overview

The converter transforms Bench2Drive annotation files (JSON.gz format) into Metadrive-compatible pickle files that can be used for scenario replay and simulation.

### Key Features

- ✅ Converts vehicles and pedestrians to Metadrive format
- ✅ Handles ego vehicle (SDC) extraction
- ✅ Backward interpolation for vehicles spawning after first frame
- ✅ **Road and lane extraction from HD-Map data**
- ✅ **Lane markings and road boundaries processing**
- ✅ **Cached HD-Map loading for faster processing**
- ✅ Generates required output files (dataset_mapping.pkl, dataset_summary.pkl, scenario.pkl)
- ✅ Enhanced visualization tools for scenario verification
- ✅ Filtering of non-vehicle objects (traffic signs, traffic lights)

## Files

- `convert_bench2drive.py` - Main converter script
- `visualize_converted_scenario.py` - Visualization and analysis tool
- `README_bench2drive_converter.md` - This documentation

## Installation Requirements

```bash
conda activate mdsn  # Use the mdsn conda environment
# Required packages: numpy, pickle, gzip, json, pathlib
# For visualization: matplotlib, metadrive (optional)
```

## Usage

### 1. Convert a Scenario

```bash
# Basic conversion (vehicles and pedestrians only)
python tools/convert_bench2drive.py "Bench2Drive/Bench2Drive-mini/ControlLoss_Town11_Route401_Weather11" --output-dir "converted_scenarios"

# With HD-Map data for roads and lanes
python tools/convert_bench2drive.py "Bench2Drive/Bench2Drive-mini/ControlLoss_Town11_Route401_Weather11" --output-dir "converted_scenarios" --hd-map "Bench2Drive/Bench2Drive-Map/Town11_HD_map.npz"

# With custom output directory  
python tools/convert_bench2drive.py "/path/to/scenario" -o "/path/to/output" --hd-map "/path/to/hd_map.npz"
```

### 2. Analyze Converted Scenario

```bash
# Analyze scenario data structure and object tracks
python tools/visualize_converted_scenario.py "converted_scenarios" --mode analyze

# Analyze specific scenario file
python tools/visualize_converted_scenario.py "converted_scenarios" --mode analyze --scenario-file "sd_bench2drive_ControlLoss-Town11-Route401-Weather11.pkl"
```

### 3. Generate Static Plot

```bash
# Create trajectory plot
python tools/visualize_converted_scenario.py "converted_scenarios" --mode plot --output "scenario_plot.png"

# Plot specific scenario
python tools/visualize_converted_scenario.py "converted_scenarios" --mode plot --scenario-file "sd_bench2drive_*.pkl" --output "my_plot.png"

# Enhanced visualization with roads and vehicles (for scenarios with HD-Map data)
python tools/visualize_roads_and_vehicles.py "converted_scenarios/converted_scenarios_0/scenario.pkl" --output "comprehensive_plot.png"
```

### 4. Metadrive Visualization (Optional)

If MetaDrive is installed:

```bash
# Interactive visualization with MetaDrive
python tools/visualize_converted_scenario.py "converted_scenarios" --mode metadrive --max-scenarios 1 --render-frames 200
```

## Input Data Structure

The converter expects Bench2Drive scenarios with this structure:
```
ControlLoss_Town11_Route401_Weather11/
├── anno/
│   ├── 00000.json.gz
│   ├── 00001.json.gz
│   └── ...
├── camera/
├── lidar/
└── other_data/
```

## Output Data Structure

The converter generates:
```
converted_scenarios/
├── dataset_mapping.pkl          # Maps scenario files to dataset
├── dataset_summary.pkl          # Summary of all scenarios
└── sd_bench2drive_ControlLoss-Town11-Route401-Weather11.pkl  # Individual scenario
```

## Conversion Details

### Object Types Supported
- **Vehicles**: Converted to `VEHICLE` type
- **Pedestrians/Walkers**: Converted to `PEDESTRIAN` type  
- **Ego Vehicle**: Special handling as SDC (Self-Driving Car)

### Object Types Filtered Out
- Traffic signs
- Traffic lights
- Static objects without movement

### Data Processing
1. **Frame Processing**: Processes all annotation frames (typically 350 frames at ~20Hz)
2. **Coordinate System**: Preserves world coordinates from CARLA
3. **Interpolation**: Vehicles not present from frame 0 get backward interpolation
4. **Velocity Calculation**: Computed from speed+heading or position differences
5. **Validation**: All interpolated frames marked as valid for Metadrive compatibility

### Key Metadrive Fields Generated
- `tracks`: Object trajectories with position, velocity, dimensions, heading
- `sdc_id`: Self-driving car identifier
- `track_length`: Number of frames (typically 350)
- `ts`: Timestamps array (20Hz = 0.05s intervals)
- `object_summary`: Statistics for each object (distance traveled, validity)
- `number_summary`: Scenario-level statistics

## Example Output

### Scenario with HD-Map Data

```bash
=== Conversion Results ===
INFO: Loaded HD map with 1874 roads
INFO: Processing 1874 roads for map features...
INFO: Extracted 11204 map features
INFO: Conversion completed successfully!

=== Scenario Analysis ===
Scenario ID: ControlLoss-Town11-Route401-Weather11
Track Length: 350 frames
SDC ID: 5363
Total Objects: 12
Object Types: {'VEHICLE': 12}

Map Features: 11,204 total
  - Lanes: 3,166
  - Road Lines: 8,038
  - Types: SolidSolid, Solid, Broken, Center

Track Analysis:
  Object 5405 (VEHICLE):
    Valid frames: 128/350
    Start: (7017.5, -2855.0)
    End: (7107.4, -2863.8)
    Distance traveled: 90.3m
```

## Limitations

- **MetaDrive Integration**: While map features are extracted, MetaDrive's road network construction requires additional processing for full compatibility
- **Traffic Elements**: Traffic lights and signs are not yet supported (planned for future enhancement)
- **Lane Topology**: Complex lane connections and junctions need further development
- **Complex Interpolation**: Uses simple backward interpolation for late-spawning vehicles

## Troubleshooting

### Common Issues

1. **"No such file or directory"**: Check that the scenario path exists and contains an `anno/` subdirectory
2. **"MetaDrive not available"**: Install MetaDrive for interactive visualization, or use static plotting
3. **Empty tracks**: Ensure the annotation files contain `bounding_boxes` data

### Debugging Tips

1. Use `--mode analyze` to inspect converted data structure
2. Check annotation files with: `gunzip -c 00000.json.gz | python -m json.tool`
3. Verify object IDs and types in raw annotations

## Future Enhancements

- [ ] Add support for traffic lights and signs
- [ ] Implement road/lane topology conversion
- [ ] Add support for more complex interpolation methods
- [ ] Support for multiple scenarios in batch processing
- [ ] Integration with HD-Map data from Bench2Drive-Map

## Testing

The converter has been tested on:
- ControlLoss_Town11_Route401_Weather11 scenario
- 350 frame sequences
- Mixed vehicle types with different spawn times
- Successfully generates compatible Metadrive files

## References

- [Bench2Drive Dataset](https://github.com/Thinklab-SJTU/Bench2Drive)
- [MetaDrive](https://github.com/metadriverse/metadrive)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)
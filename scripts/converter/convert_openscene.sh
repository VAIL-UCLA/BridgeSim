#!/bin/bash

# OpenScene/NavSim scenario conversion script
# Usage: bash convert_openscene.sh [input_dir] [output_dir] [map_root]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration - modify these paths as needed
INPUT_DIR="${1:-/home/seth/note_workspace/BridgeSim/test_navsim_logs/test}"
OUTPUT_DIR="${2:-/home/seth/note_workspace/BridgeSim/converted_scenarios/navhard}"
MAP_ROOT="${3:-/home/seth/note_workspace/BridgeSim/maps}"
SCENE_FILTER="${4:-/home/seth/note_workspace/BridgeSim/converters/openscene/filter/navhard_two_stage.yaml}"

# Check required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MAP_ROOT" ]; then
    echo "Usage: bash convert_openscene.sh [input_dir] [output_dir] [map_root] [scene_filter (optional)]"
    echo ""
    echo "Arguments:"
    echo "  input_dir    - Directory containing raw NavSim/OpenScene logs"
    echo "  output_dir   - Output directory for converted scenarios"
    echo "  map_root     - Directory containing nuPlan HD maps"
    echo "  scene_filter - (Optional) YAML file specifying which scenes to convert"
    exit 1
fi

echo "============================================================"
echo "OpenScene Conversion"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Maps:   $MAP_ROOT"
[ -n "$SCENE_FILTER" ] && echo "Filter: $SCENE_FILTER"
echo "============================================================"

# Build command
CMD=(python "${REPO_ROOT}/converters/openscene/convert_openscene_with_filter.py"
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --map-root "$MAP_ROOT"
    --num-future-frames-extract 40
    --interpolate)

[ -n "$SCENE_FILTER" ] && CMD+=(--scene-filter "$SCENE_FILTER")

"${CMD[@]}"

#!/bin/bash

# OpenScene/NavSim scenario conversion script
# Usage: bash convert_openscene.sh [input_dir] [output_dir] [map_root]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration - modify these paths as needed
INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"
MAP_ROOT="${3:-}"
SCENE_FILTER="${4:-}"  # Optional: path to scene filter YAML

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
if [ -n "$SCENE_FILTER" ]; then
    python "${REPO_ROOT}/converters/openscene/convert_openscene_with_filter.py" \
        --scene-filter "$SCENE_FILTER" \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --map-root "$MAP_ROOT" \
        --num-future-frames-extract 220 \
        --interpolate
else
    python "${REPO_ROOT}/converters/openscene/convert_openscene.py" \
        --input "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --map_root "$MAP_ROOT" \
        --interpolate
fi

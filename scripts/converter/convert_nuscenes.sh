#!/bin/bash
# Convert nuScenes dataset to ScenarioNet format
#
# Usage: ./convert_nuscenes.sh <dataroot> <output_path> [split] [num_workers]
#
# Arguments:
#   dataroot:     Path to nuScenes dataset root
#   output_path:  Output directory for converted scenarios
#   split:        Dataset split (default: v1.0-mini)
#                 Options: v1.0-mini, v1.0-trainval, v1.0-test,
#                          mini_train, mini_val, train, train_val, val
#   num_workers:  Number of parallel workers (default: 8)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataroot> <output_path> [split] [num_workers]"
    echo ""
    echo "Arguments:"
    echo "  dataroot:     Path to nuScenes dataset root"
    echo "  output_path:  Output directory for converted scenarios"
    echo "  split:        Dataset split (default: v1.0-mini)"
    echo "                Options: v1.0-mini, v1.0-trainval, v1.0-test"
    echo "                         mini_train, mini_val, train, train_val, val"
    echo "  num_workers:  Number of parallel workers (default: 8)"
    exit 1
fi

DATAROOT="$1"
OUTPUT_PATH="$2"
SPLIT="${3:-v1.0-mini}"
NUM_WORKERS="${4:-8}"

echo "Converting nuScenes dataset..."
echo "  Input: $DATAROOT"
echo "  Output: $OUTPUT_PATH"
echo "  Split: $SPLIT"
echo "  Workers: $NUM_WORKERS"

cd "$REPO_ROOT"

python -m converters.nuscenes.convert_nuscenes \
    --dataroot "$DATAROOT" \
    --database_path "$OUTPUT_PATH" \
    --split "$SPLIT" \
    --num_workers "$NUM_WORKERS" \
    --overwrite

echo "Conversion complete!"

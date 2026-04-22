#!/bin/bash
# Convert Waymo Motion Dataset to ScenarioNet format
#
# Usage: ./convert_waymo.sh <raw_data_path> <output_path> [num_workers]
#
# Arguments:
#   raw_data_path: Directory containing Waymo tfrecord files
#   output_path:   Output directory for converted scenarios
#   num_workers:   Number of parallel workers (default: 8)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <raw_data_path> <output_path> [num_workers]"
    echo ""
    echo "Arguments:"
    echo "  raw_data_path: Directory containing Waymo tfrecord files"
    echo "  output_path:   Output directory for converted scenarios"
    echo "  num_workers:   Number of parallel workers (default: 8)"
    exit 1
fi

RAW_DATA_PATH="$1"
OUTPUT_PATH="$2"
NUM_WORKERS="${3:-8}"

echo "Converting Waymo dataset..."
echo "  Input: $RAW_DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  Workers: $NUM_WORKERS"

cd "$REPO_ROOT"

python -m converters.waymo.convert_waymo \
    --raw_data_path "$RAW_DATA_PATH" \
    --database_path "$OUTPUT_PATH" \
    --num_workers "$NUM_WORKERS" \
    --overwrite

echo "Conversion complete!"

#!/bin/bash
#
# This script automates the conversion of multiple Bench2Drive scenarios
# by using the fast local node disk (/tmp) for all heavy I/O operations
# to avoid PVC (network storage) slowness.

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Source and Destination Directories ---
SCENARIO_DIR="/workspace/Bench2Drive-base"
MAP_DIR="/workspace/Bench2Drive-Map"
FINAL_OUTPUT_DIR="/workspace/converted_scenarios"
CONVERTER_SCRIPT="./convert_bench2drive.py"

# --- Temporary Local Disk Directories ---
# Create a unique temporary base directory on the fast local disk
TEMP_BASE_DIR=$(mktemp -d /tmp/b2d-conversion.XXXXXX)
TEMP_MAP_DIR="$TEMP_BASE_DIR/maps"
TEMP_SCENARIO_DIR="$TEMP_BASE_DIR/scenarios"
TEMP_OUTPUT_DIR="$TEMP_BASE_DIR/output"

# --- Cleanup Trap ---
# This function will run on EXIT (success or failure) to clean up the temp dir
cleanup() {
    echo "Cleaning up temporary directory: $TEMP_BASE_DIR"
    rm -rf "$TEMP_BASE_DIR"
}
trap cleanup EXIT

# --- Pre-flight Checks ---
if [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo "Error: Converter script not found at $CONVERTER_SCRIPT"
    exit 1
fi
if [ ! -d "$SCENARIO_DIR" ]; then
    echo "Error: Scenario directory not found at $SCENARIO_DIR"
    exit 1
fi
if [ ! -d "$MAP_DIR" ]; then
    echo "Error: Map directory not found at $MAP_DIR"
    exit 1
fi

# --- Setup ---
# Create directory structures
mkdir -p "$FINAL_OUTPUT_DIR"
mkdir -p "$TEMP_MAP_DIR"
mkdir -p "$TEMP_SCENARIO_DIR"
mkdir -p "$TEMP_OUTPUT_DIR"

echo "Temporary directory created at: $TEMP_BASE_DIR"

# 1. Copy all HD Maps to local disk (once)
echo "Copying all HD Maps from PVC to local disk..."
cp "$MAP_DIR"/*.npz "$TEMP_MAP_DIR/"
echo "Map copy complete."

# --- Main Processing Loop ---
# Find all scenario files and count them
scenario_files=("$SCENARIO_DIR"/*.tar.gz)
total_scenarios=${#scenario_files[@]}
echo "Found $total_scenarios scenarios to convert."

current_scenario=0
for scenario_path in "${scenario_files[@]}"; do
    current_scenario=$((current_scenario + 1))
    scenario_filename=$(basename "$scenario_path")
    scenario_basename=$(basename "$scenario_filename" .tar.gz)
    
    echo "----------------------------------------------------------------------"
    echo "Processing scenario $current_scenario of $total_scenarios: $scenario_filename"

    # --- Determine Map Path ---
    if [[ "$scenario_filename" =~ _(Town[0-9]+(HD)?)_ ]]; then
        town_name="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not extract town name from '$scenario_filename'. Skipping."
        continue
    fi

    if [ "$town_name" == "Town10" ]; then
        map_filename="Town10HD_HD_map.npz"
    else
        map_filename="${town_name}_HD_map.npz"
    fi

    temp_map_path="$TEMP_MAP_DIR/$map_filename"

    if [ ! -f "$temp_map_path" ]; then
        echo "Warning: Local map file not found at '$temp_map_path'. Skipping."
        continue
    fi

    # --- Define Local Paths ---
    temp_scenario_path="$TEMP_SCENARIO_DIR/$scenario_filename"
    temp_scenario_output_dir="$TEMP_OUTPUT_DIR/$scenario_basename"

    # --- Execute Strategy ---
    # 1. Copy scenario .tar.gz to local disk
    echo "  1. Copying scenario to local disk..."
    cp "$scenario_path" "$temp_scenario_path"

    # 2. Run conversion using local disk for all I/O
    echo "  2. Running conversion on local disk..."
    echo "     - Scenario: $temp_scenario_path"
    echo "     - HD Map: $temp_map_path"
    echo "     - Output Dir: $temp_scenario_output_dir"
    
    python3 "$CONVERTER_SCRIPT" "$temp_scenario_path" \
        --hd-map "$temp_map_path" \
        --output-dir "$temp_scenario_output_dir"

    # 3. Copy the resulting folder back to the PVC
    echo "  3. Copying converted data from local disk to PVC..."
    cp -r "$temp_scenario_output_dir" "$FINAL_OUTPUT_DIR/"

    # 4. Clean up this scenario's temp files
    echo "  4. Cleaning up local temp files for this scenario..."
    rm "$temp_scenario_path"
    rm -rf "$temp_scenario_output_dir"

    echo "Successfully converted $scenario_filename"
done

echo "----------------------------------------------------------------------"
echo "Batch conversion complete. All $total_scenarios scenarios processed."
# The 'trap' will now automatically clean up $TEMP_BASE_DIR and the maps
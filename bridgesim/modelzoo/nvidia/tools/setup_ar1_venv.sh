#!/bin/bash
# Setup script for Alpamayo-R1 venv
# Creates ar1_venv inside bridgesim/modelzoo/nvidia/alpamayo/ and installs all dependencies.
#
# Usage:
#   bash setup_ar1_venv.sh

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ALPAMAYO_DIR="$( cd "$SCRIPT_DIR/../alpamayo" && pwd )"
VENV_DIR="${VENV_DIR:-$ALPAMAYO_DIR/ar1_venv}"
PYTHON="${PYTHON:-python3.12}"

echo "============================================================"
echo "Alpamayo-R1 venv setup"
echo "  venv:     $VENV_DIR"
echo "  python:   $PYTHON"
echo "  package:  $ALPAMAYO_DIR"
echo "============================================================"

# 1. Create venv
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "[1/5] Creating venv..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/5] Venv already exists, skipping creation."
fi

PIP="$VENV_DIR/bin/pip"

# 2. Upgrade pip + install setuptools
echo "[2/5] Upgrading pip and installing setuptools..."
"$PIP" install --upgrade pip setuptools

# 3. Install torch (must come before flash-attn)
echo "[3/5] Installing torch==2.8.0 and torchvision..."
"$PIP" install torch==2.8.0 torchvision

# 4. Install flash-attn
echo "[4/5] Installing flash-attn..."
if [ -n "${FLASH_WHEEL:-}" ]; then
    echo "  Using provided wheel: $FLASH_WHEEL"
    "$PIP" install "$FLASH_WHEEL"
else
    # Try the pre-built wheel for cu128 + torch2.8 + cp312 first (fast, no compilation)
    FLASH_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
    echo "  Trying pre-built wheel: $FLASH_WHEEL_URL"
    if "$PIP" install "$FLASH_WHEEL_URL"; then
        echo "  Pre-built wheel installed successfully."
    else
        echo "  Pre-built wheel failed, falling back to build from source (this will take a while)..."
        "$PIP" install numpy psutil
        "$PIP" install flash-attn --no-build-isolation
    fi
fi

# 5. Install alpamayo_r1 and remaining deps
echo "[5/5] Installing alpamayo_r1 package and remaining dependencies..."
"$PIP" install "$ALPAMAYO_DIR"

echo "============================================================"
echo "Done! Venv is ready at: $VENV_DIR"
echo ""
echo "To use a custom path, set:"
echo "  export ALPAMAYO_PYTHON=$VENV_DIR/bin/python"
echo "============================================================"

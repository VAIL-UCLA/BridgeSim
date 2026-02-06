#!/bin/bash

# --- CONFIGURATION & SAFETY ---
# Stop script on any error
set -e
# Print commands as they execute so you can see progress in logs
set -x

# Function to find conda hook
find_conda() {
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        echo "Conda definition not found. Ensure 'conda' is in your PATH or update this script."
        exit 1
    fi
}

# Initialize Conda
find_conda
echo "Conda initialized."

# Save the starting directory (Assumed to be the root containing Bench2Drive and RAP)
ROOT_DIR=$(pwd)

# ==========================================
# PART 1: MDSN ENVIRONMENT (Bench2Drive)
# ==========================================
echo ">>> Starting Bench2Drive / MDSN setup..."

# Activate existing environment
conda activate mdsn

# Navigate to Bench2DriveZoo
# Assumes folder structure: ./Bench2Drive/Bench2DriveZoo
cd Bench2Drive/Bench2DriveZoo

# The Numpy Dance (As requested)
pip install --upgrade numpy==1.20.0
pip install ninja packaging
pip install -v -e .
# Re-upgrade numpy as specified
pip install --upgrade numpy==1.21.6

# Setup Team Code Checkpoints
cd ../team_code
mkdir -p ckpts
cd ckpts

# Download uniad_base_b2d.pth
# Using -o to handle the output filename cleanly
echo "Downloading uniad_base_b2d.pth..."
curl -L -o uniad_base_b2d.pth "https://huggingface.co/rethinklab/Bench2DriveZoo/resolve/main/uniad_base_b2d.pth?download=true"

# Setup ScenarioNet
cd ../../../scenarionet
pip install -e .

# Setup Root dependencies
cd ../
pip install 'tensorflow[and-cuda]'
git pull

echo ">>> Bench2Drive Setup Complete."

# ==========================================
# PART 2: RAP SETUP
# ==========================================
echo ">>> Starting RAP Setup..."

# Return to root directory (BridgeSim/Root) to ensure clean relative paths
cd "$ROOT_DIR"

# Create RAP environment
# Using -y to bypass "Proceed [y/n]?" prompt
conda create -n rap python=3.9 pip -y
conda activate rap

cd RAP
pip install -e .

# Install Torch with CUDA 12.1 support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 -U

pip install 'tensorflow[and-cuda]'
pip install --upgrade numpy==1.26.0
pip install openmim

# MMCV Setup
# Uninstalling explicit confirms to ensure clean slate for MIM
pip uninstall mmcv -y || true  # '|| true' prevents script crash if mmcv isn't installed
mim install mmcv==2.1.0

pip install transformers==4.56.0

# RAP Checkpoints
mkdir -p ckpts && cd ckpts
echo "Downloading RAP_DINO_waymo_seed2.ckpt..."
curl -L -o RAP_DINO_waymo_seed2.ckpt "https://huggingface.co/Lanl11/RAP_ckpts/resolve/main/RAP_DINO_waymo_seed2.ckpt?download=true"

# Return to BridgeSim root context to clone nuplan
cd ../../

# Nuplan Devkit
if [ -d "nuplan-devkit" ]; then
    echo "nuplan-devkit already exists, entering directory..."
    cd nuplan-devkit
else
    git clone https://github.com/motional/nuplan-devkit.git 
    cd nuplan-devkit
fi
pip install -e .

cd ../

# Waymo and MetaDrive
pip install waymo-open-dataset-tf-2-12-0

cd metadrive
pip install -e .[cuda]
pip install cupy-cuda12x
pip install cuda-python

cd ../scenarionet
pip install -e .

# Final cd ..
cd ..

echo ">>> ALL TASKS COMPLETED SUCCESSFULLY."
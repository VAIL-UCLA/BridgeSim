# Use the correct NVIDIA CUDA base image to match the project's requirements
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Switch to root user for installations
USER root

# Proxy settings, only necessary if behind a corporate proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

# Install essential system dependencies, including GCC-9 for compatibility
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    rsync \
    vim \
    unzip \
    htop \
    tmux \
    tar \
    ca-certificates \
    xorg \
    libjpeg-dev \
    libpng-dev \
    libpng16-16 \
    libjpeg-turbo8 \
    libtiff5 \
    libomp5 \
    libice6 \
    libsm6 \
    libxaw7 \
    libxkbfile1 \
    libxmu6 \
    libxpm4 \
    libxt6 \
    libsdl2-2.0 \
    x11-common \
    x11-xkb-utils \
    xkb-data \
    xserver-xorg \
    libvulkan1 \
    vulkan-tools \
    python3-dev \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    gcc-9 \
    g++-9 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_ALWAYS_YES=true
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:$PATH"

# Create a non-root user and the work directory
RUN useradd -ms /bin/bash abhijit && \
    mkdir -p /home/abhijit/Work && \
    chown -R abhijit:abhijit /home/abhijit

RUN apt-get update && \
    apt-get install -y sudo && \
    adduser abhijit sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

# Switch to the new non-root user for the rest of the build
USER abhijit
WORKDIR /home/abhijit/Work

# --- Manually Install CARLA 0.9.15 ---
ENV CARLA_ROOT /home/abhijit/Work/carla
RUN mkdir -p $CARLA_ROOT && \
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz -O carla.tar.gz && \
    tar -xzf carla.tar.gz -C $CARLA_ROOT && \
    rm carla.tar.gz

RUN cd $CARLA_ROOT/Import && \
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz && \
    cd $CARLA_ROOT && \
    bash ImportAssets.sh

# --- Project Specific Setup ---

# Set up build argument for the GitHub token
ARG GITHUB_TOKEN

# Clone the private repository
RUN if [ -z "$GITHUB_TOKEN" ]; then echo "Error: GITHUB_TOKEN is not set." && exit 1; fi && \
    git clone https://oauth2:${GITHUB_TOKEN}@github.com/sethzhao506/BridgeSim.git

WORKDIR /home/abhijit/Work/BridgeSim

# Set environment variables for the project
ENV SCENARIO_RUNNER_ROOT="/home/abhijit/Work/BridgeSim/Bench2Drive/scenario_runner"
ENV LEADERBOARD_ROOT="/home/abhijit/Work/BridgeSim/Bench2Drive/leaderboard"
ENV TEAM_CODE_ROOT="/home/abhijit/Work/BridgeSim/Bench2Drive/team_code"
ENV ZOO_ROOT="/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2DriveZoo"
ENV VQA_GEN=1
ENV STRICT_MODE=1

# Create the conda environment
RUN conda create -n b2d python=3.7 numpy networkx scipy six requests -y

# Set the shell to use conda run for subsequent commands
SHELL ["conda", "run", "-n", "b2d", "/bin/bash", "-c"]

# Install Python requirements
RUN pip install -r ${SCENARIO_RUNNER_ROOT}/requirements.txt && \
    pip install -r ${LEADERBOARD_ROOT}/requirements.txt && \
    pip install -r ${ZOO_ROOT}/requirements.txt

# Install PyTorch with the correct CUDA version
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Add necessary paths to PYTHONPATH
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${TEAM_CODE_ROOT}:${PYTHONPATH}"

# Install remaining local packages
# Use the compatible GCC-9 compiler for these packages
RUN pip install ninja packaging
RUN pip install -U "huggingface-hub[cli]"

# --- Data and Model Downloading ---
# NOTE: These steps will significantly increase image size.
# It is recommended to mount this data at runtime instead.

# Download model checkpoints
WORKDIR /home/abhijit/Work/BridgeSim/Bench2Drive/Bench2DriveZoo
#RUN mkdir ckpts && \
#    huggingface-cli download rethinklab/Bench2DriveZoo --repo-type model --local-dir ckpts --local-dir-use-symlinks False
RUN git lfs install && \
    git clone https://huggingface.co/rethinklab/Bench2DriveZoo ckpts

# Set up symlinks
WORKDIR /home/abhijit/Work/BridgeSim/Bench2Drive
RUN mkdir -p team_code && \
    ln -s ../Bench2DriveZoo/team_code/* ./team_code/

RUN git lfs install && \
#    git clone https://huggingface.co/datasets/rethinklab/Bench2Drive Bench2Drive-mini && \
    git clone https://huggingface.co/datasets/rethinklab/Bench2Drive-Map Bench2Drive-Map

# Download the mini dataset
RUN mkdir Bench2Drive-mini
WORKDIR /home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini
#RUN huggingface-cli download --resume-download --repo-type dataset rethinklab/Bench2Drive --include "*.tar.gz" --local-dir Bench2Drive-mini --local-dir-use-symlinks False
RUN wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/HardBreakRoute_Town01_Route30_Weather3.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/Accident_Town03_Route156_Weather0.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/YieldToEmergencyVehicle_Town04_Route165_Weather7.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/ConstructionObstacle_Town05_Route68_Weather8.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/ParkedObstacle_Town10HD_Route371_Weather7.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/ControlLoss_Town11_Route401_Weather11.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/AccidentTwoWays_Town12_Route1444_Weather0.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/OppositeVehicleTakingPriority_Town13_Route600_Weather2.tar.gz && \
    wget https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/VehicleTurningRoute_Town15_Route443_Weather1.tar.gz


# Download the map dataset
#RUN mkdir Bench2Drive-Map && \
#    huggingface-cli download --repo-type dataset rethinklab/Bench2Drive-Map --local-dir Bench2Drive-Map --local-dir-use-symlinks False

# Set final working directory
WORKDIR /home/abhijit/Work/BridgeSim

# Default command to start an interactive shell
CMD ["/bin/bash"]
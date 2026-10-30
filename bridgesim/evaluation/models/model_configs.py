"""
Model configuration file - maps each adapter to its checkpoint path.

Checkpoint Base Path: ckpts/BridgeSim (relative to repo root; override with
    the BRIDGESIM_CKPT_ROOT environment variable)
Front-camera policy Base Path: ckpts/DreamStream/policy (override with the
    DREAMSTREAM_POLICY_ROOT environment variable). This is the layout produced by
    `hf download UCLA-VAIL/DreamStream --include "policy/*"
    --local-dir ckpts/DreamStream`; see docs/dreamstream.md.
Config Base Path: bridgesim/modelzoo/bench2drive (relative to repo root)

Directory Structure:
- bench2drive/: Contains UniAD, VAD, TCP checkpoints (Bench2Drive trained)
- navsimv2/: Contains DiffusionDrive, DiffusionDriveV2, DrivoR, LEAD, RAP, TransFuser checkpoints (NavSim trained)
"""

import os
from pathlib import Path

# Base paths (relative to this file's directory)
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent  # bridgesim/evaluation/models -> repo root
CKPT_BASE = str(Path(os.environ.get(
    "BRIDGESIM_CKPT_ROOT", _REPO_ROOT / "ckpts/BridgeSim")).resolve())
FRONT_ONLY_BASE = str(Path(os.environ.get(
    "DREAMSTREAM_POLICY_ROOT", _REPO_ROOT / "ckpts/DreamStream/policy")).resolve())
_MODELZOO_BENCH2DRIVE = _THIS_DIR.parent.parent / "modelzoo" / "bench2drive"

# Model checkpoint configurations
# Models trained on Bench2Drive (bench2drive directory)
BENCH2DRIVE_MODELS = {
    "uniad": {
        "checkpoint": os.path.join(CKPT_BASE, "bench2drive/UniAD/uniad_base_b2d.pth"),
        "config": str(_MODELZOO_BENCH2DRIVE / "adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py"),
        "description": "UniAD base model trained on Bench2Drive",
    },
    "vad": {
        "checkpoint": os.path.join(CKPT_BASE, "bench2drive/VAD/vad_b2d_base.pth"),
        "config": str(_MODELZOO_BENCH2DRIVE / "adzoo/vad/configs/VAD_base_e2e_b2d.py"),
        "description": "VAD base model trained on Bench2Drive",
    },
    "tcp": {
        "checkpoint": os.path.join(CKPT_BASE, "bench2drive/TCP/tcp_b2d.ckpt"),
        "config": None,  # TCP doesn't use external config
        "description": "TCP model trained on Bench2Drive",
    },
}

# Models trained on NavSim v2 (navsimv2 directory)
NAVSIMV2_MODELS = {
    "diffusiondrive": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS"),
        "config": None,
        "description": "DiffusionDrive model trained on NavSim v2",
    },
    "diffusiondrivev2": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/DiffusionDriveV2/diffusiondrivev2_sel.ckpt"),
        "config": None,
        "plan_anchor_path": os.path.join(CKPT_BASE, "navsimv2/DiffusionDriveV2/kmeans_navsim_traj_20.npy"),
        "description": "DiffusionDriveV2 with selection head trained on NavSim v2",
    },
    "drivor": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/DrivoR/drivor_Nav2_10epochs.pth"),
        "config": None,
        "description": "DrivoR model trained on NavSim v2",
    },
    "drivor_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "drivor_front_Nav2_10epochs.pth"),
        "config": None,
        "description": "DrivoR front-camera-only model trained on NavSim v2",
    },
    "lead_navsim": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/LEAD_navsim/model_0060.pth"),
        "config": None,
        "description": "LEAD NavSim (LTFv6) 4-camera model trained on NavSim v2",
    },
    "rap": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/RAP_DINO/RAP_DINO_navsimv2.ckpt"),
        "config": None,
        "description": "RAP with DINO backbone trained on NavSim v2",
    },
    "transfuser": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/transfuser/transfuser_seed_0.ckpt"),
        "config": None,
        "description": "TransFuser model trained on NavSim v2",
    },
    "ltf": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/ltf/ltf_seed_0.ckpt"),
        "config": None,
        "description": "LTF (Latent TransFuser) image-only model trained on NavSim v2",
    },
    "ego_mlp": {
        "checkpoint": os.path.join(CKPT_BASE, "navsimv2/ego_status_mlp/ego_status_mlp_seed_0.ckpt"),
        "config": None,
        "description": "EgoStatusMLP blind baseline (no sensors, only ego state) trained on NavSim v2",
    },
    "alpamayo_r1": {
        "checkpoint": "nvidia/Alpamayo-R1-10B",
        "config": None,
        "description": "Alpamayo-R1 VLA model"
    },
    # Front-camera-only variants used by DreamStream (checkpoints live under
    # FRONT_ONLY_BASE; RAP is the official 4-camera RAP-DINO NAVSIM v1 checkpoint).
    "diffusiondrive_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "diffusiondrive_front_only_epoch_99_step_33300.ckpt"),
        "config": None,
        "plan_anchor_path": os.path.join(FRONT_ONLY_BASE, "aux/kmeans_navsim_traj_20.npy"),
        "description": "DiffusionDrive front-camera-only",
    },
    "diffusiondrivev2_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "diffusiondrivev2_epoch19_step6660.ckpt"),
        "config": None,
        "plan_anchor_path": os.path.join(FRONT_ONLY_BASE, "aux/kmeans_navsim_traj_20.npy"),
        "description": "DiffusionDriveV2 front-camera-only",
    },
    "ltf_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "ltf_epoch99_step_11100.ckpt"),
        "config": None,
        "description": "LTF (Latent TransFuser) front-camera-only",
    },
    "rap_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "RAP_DINO_navsimv1.ckpt"),
        "config": None,
        "description": "RAP-DINO front-camera-only (official 4-camera checkpoint)",
    },
    "sparsedrivev2_front": {
        "checkpoint": os.path.join(FRONT_ONLY_BASE, "sparsedrivev2_epoch_9_step_6650.ckpt"),
        "config": None,
        "description": "SparseDriveV2 front-camera-only",
    },
}

# Other models
COMMAAI_MODELS = {
    "openpilot": {
        "checkpoint": os.path.join(CKPT_BASE, "commaai/openpilot"),
        "config": None,
        "description": "OpenPilot model trained by CommaAI",
    },
}

# Combined model configurations
MODEL_CONFIGS = {**BENCH2DRIVE_MODELS, **NAVSIMV2_MODELS, **COMMAAI_MODELS}


def get_checkpoint_path(model_type: str) -> str:
    """
    Get the checkpoint path for a given model type.

    Args:
        model_type: Model type (e.g., 'uniad', 'vad', 'tcp', 'rap', etc.)

    Returns:
        Absolute path to the model checkpoint

    Raises:
        ValueError: If model type is not recognized
    """
    model_type = model_type.lower()
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]["checkpoint"]


def get_config_path(model_type: str) -> str:
    """
    Get the config path for a given model type.

    Args:
        model_type: Model type (e.g., 'uniad', 'vad', etc.)

    Returns:
        Path to the model config file, or None if no config is needed

    Raises:
        ValueError: If model type is not recognized
    """
    model_type = model_type.lower()
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]["config"]


def get_model_info(model_type: str) -> dict:
    """
    Get full model configuration info for a given model type.

    Args:
        model_type: Model type (e.g., 'uniad', 'vad', etc.)

    Returns:
        Dictionary with checkpoint, config, and description

    Raises:
        ValueError: If model type is not recognized
    """
    model_type = model_type.lower()
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type]


def is_bench2drive_model(model_type: str) -> bool:
    """Check if a model is trained on Bench2Drive."""
    return model_type.lower() in BENCH2DRIVE_MODELS


def is_navsimv2_model(model_type: str) -> bool:
    """Check if a model is trained on NavSim v2."""
    return model_type.lower() in NAVSIMV2_MODELS


def list_available_models() -> list:
    """List all available model types."""
    return list(MODEL_CONFIGS.keys())


def print_model_summary():
    """Print a summary of all available models and their checkpoint paths."""
    print("=" * 80)
    print("Model Configurations Summary")
    print("=" * 80)
    print(f"\nCheckpoint Base: {CKPT_BASE}")

    print("\n--- Bench2Drive Models (bench2drive/) ---")
    for name, config in BENCH2DRIVE_MODELS.items():
        print(f"  {name}:")
        print(f"    Checkpoint: {config['checkpoint']}")
        if config['config']:
            print(f"    Config: {config['config']}")
        print(f"    Description: {config['description']}")

    print("\n--- NavSim v2 Models (navsimv2/) ---")
    for name, config in NAVSIMV2_MODELS.items():
        print(f"  {name}:")
        print(f"    Checkpoint: {config['checkpoint']}")
        if config['config']:
            print(f"    Config: {config['config']}")
        print(f"    Description: {config['description']}")

    print("\n--- CommaAI Models (commaai/) ---")
    for name, config in COMMAAI_MODELS.items():
        print(f"  {name}:")
        print(f"    Checkpoint: {config['checkpoint']}")
        if config['config']:
            print(f"    Config: {config['config']}")
        print(f"    Description: {config['description']}")

    print("=" * 80)


if __name__ == "__main__":
    print_model_summary()
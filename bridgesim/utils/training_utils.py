"""
Training utilities for BridgeSim.

This module provides shared training utilities used across calibration and model training.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


# =============================================================================
# Constants
# =============================================================================

# Constants match RAP bev_feature_build normalization
BEV_MEAN = (123.675, 116.28, 103.53)
BEV_STD = (58.395, 57.12, 57.375)


# =============================================================================
# Utility Functions
# =============================================================================


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

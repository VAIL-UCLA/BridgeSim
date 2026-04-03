"""
Inference-only Lightning module for UniAD BEV flow matching calibrator.
Used to load checkpoints trained for MetaDrive -> Bench2Drive domain adaptation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl

from .flow_matching_model import FlowMatchingModel


class FlowMatchingLit(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        lr_scheduler_name: str = "customized",
        warmup_steps: int = 500,
        lr_t_max: int = 1000,
        ema_decay: float = 0.9999,
        visualize: bool = False,
        viz_dir: Optional[Path] = None,
        viz_score_thresh: float = 0.2,
        config_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        dit_variant: str = "DiT-B/2",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ema_decay = ema_decay

        self.flow_model = FlowMatchingModel(dit_variant=dit_variant)
        self.ema_flow_model = deepcopy(self.flow_model)
        for p in self.ema_flow_model.parameters():
            p.requires_grad_(False)

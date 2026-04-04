"""
Inference-only Lightning module for TransFuser BEV flow matching calibrator.
Used to load checkpoints trained for MetaDrive -> NavSim domain adaptation.
"""

from __future__ import annotations

from copy import deepcopy

import pytorch_lightning as pl

from .flow_matching_model import FlowMatchingModel

TRANSFUSER_BEV_CHANNELS = 512
TRANSFUSER_BEV_HW = 8


class FlowMatchingLit(pl.LightningModule):
    """PyTorch Lightning module for TransFuser BEV flow matching."""

    def __init__(
        self,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        lr_scheduler_name: str = "customized",
        warmup_steps: int = 500,
        lr_t_max: int = 1000,
        ema_decay: float = 0.9999,
        dit_variant: str = "DiT-S/2",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ema_decay = ema_decay

        self.flow_model = FlowMatchingModel(
            default_bev_hw=(TRANSFUSER_BEV_HW, TRANSFUSER_BEV_HW),
            bev_channels=TRANSFUSER_BEV_CHANNELS,
            dit_variant=dit_variant,
        )

        self.ema_flow_model = deepcopy(self.flow_model)
        for p in self.ema_flow_model.parameters():
            p.requires_grad_(False)

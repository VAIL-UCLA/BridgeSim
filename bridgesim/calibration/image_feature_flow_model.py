"""
Flow Matching model for non-square image features.

This wraps DiTNonSquare for flow matching on image features with shape (C, H, W)
where H != W (e.g., 64 channels, 64 height, 256 width from TransFuser stem).

Notes:
- DiTNonSquare forward expects (B, C, H, W) and a timestep vector t in [0, 1].
- psi and dt_psi implement the flow-matching interpolation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from dit_nonsquare import DiTNonSquare_models


class ImageFeatureFlowMatchingModel(nn.Module):
    """
    Flow matching model for non-square image features.

    Input shape: (B, C, H, W) where H != W
    Output shape: same as input
    """
    def __init__(
        self,
        *,
        feature_height: int = 128,
        feature_width: int = 512,
        feature_channels: int = 64,
        dit_variant: str = "DiT-B/4",
        learn_sigma: bool = False,
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
    ) -> None:
        """
        Args:
            feature_height: Height of the feature map (e.g., 128).
            feature_width: Width of the feature map (e.g., 512).
            feature_channels: Number of channels in the feature map (e.g., 64).
            dit_variant: Key into DiTNonSquare_models. Available: DiT-{S,B,L,XL}/{4,8}
            learn_sigma: Whether DiT predicts both mean and sigma; usually False.
            sigma_min: Minimum sigma for flow matching.
            sigma_max: Maximum sigma for flow matching.
        """
        super().__init__()
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.feature_channels = feature_channels
        self.learn_sigma = learn_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if dit_variant not in DiTNonSquare_models:
            raise ValueError(f"Unknown DiT variant '{dit_variant}'. Available: {list(DiTNonSquare_models.keys())}")
        self.dit_variant = dit_variant
        self.dit_builder = DiTNonSquare_models[dit_variant]

        cfg = SimpleNamespace(
            height=feature_height,
            width=feature_width,
            channels=feature_channels,
            learn_sigma=learn_sigma,
        )
        self.dit: nn.Module = self.dit_builder(config=cfg)

    def psi(self, t: torch.Tensor, noise: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Interpolation used for flow-matching: psi(t, noise, x1).

        Args:
            t: Timestep tensor (B,)
            noise: Starting state (source domain features) (B, C, H, W)
            x1: Target state (target domain features) (B, C, H, W)

        Returns:
            Interpolated features at time t
        """
        t_expanded = self.expand_t(t, noise)
        return (t_expanded * (self.sigma_min / self.sigma_max - 1) + 1) * noise + t_expanded * x1

    def dt_psi(self, t: torch.Tensor, noise: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of psi w.r.t. t (target velocity).

        Args:
            t: Timestep tensor (B,)
            noise: Starting state (source domain features) (B, C, H, W)
            x1: Target state (target domain features) (B, C, H, W)

        Returns:
            Velocity target for flow matching
        """
        _ = self.expand_t(t, noise)  # shapes are validated in expand_t
        return (self.sigma_min / self.sigma_max - 1) * noise + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Expand timestep tensor to match x dimensions."""
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity at time t.

        Args:
            x_t: Interpolated features (B, C, H, W).
            t: Timesteps (B,) sampled in the training loop.

        Returns:
            Velocity prediction from DiT, shaped like x_t.
        """
        bsz, channels, h, w = x_t.shape

        if channels != self.feature_channels:
            raise ValueError(
                f"DiT initialized for channels={self.feature_channels}; got {channels}."
            )
        if h != self.feature_height or w != self.feature_width:
            raise ValueError(
                f"DiT initialized for (H,W)=({self.feature_height},{self.feature_width}); got ({h},{w})."
            )

        self.dit = self.dit.to(x_t.device)

        if t.dim() != 1 or t.shape[0] != bsz:
            raise ValueError(f"t must be shape (B,), got {t.shape} for batch {bsz}")

        null_indicator = torch.zeros_like(t, dtype=torch.long)
        dit_out = self.dit(x_t, t=t, null_indicator=null_indicator)
        pred = dit_out[0] if isinstance(dit_out, (list, tuple)) else dit_out
        return pred

    def sample(
        self,
        src_features: torch.Tensor,
        *,
        sample_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
        step_size_type: str = "step_in_dt",
    ) -> torch.Tensor:
        """
        Euler sampler for flow matching.

        Args:
            src_features: Source features (B, C, H, W) treated as the starting state.
            sample_steps: Number of Euler steps between t_start and t_end.
            t_start: Start of integration window.
            t_end: End of integration window.
            step_size_type: "step_in_dt" or "step_in_dsigma" (currently only dt is used).

        Returns:
            Transformed features at t_end.
        """
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive.")

        x = src_features
        bsz = x.shape[0]
        device = x.device

        times_for_step = torch.linspace(t_start, t_end, sample_steps + 1, device=device)
        times_for_eval = torch.linspace(t_start, t_end, sample_steps, device=device)

        for i in range(sample_steps):
            t_i = times_for_eval[i]
            t_vec = t_i.repeat(bsz)
            v_pred = self.forward(x, t_vec)
            if step_size_type == "step_in_dt":
                step_size = times_for_step[i + 1] - times_for_step[i]
            elif step_size_type == "step_in_dsigma":
                # Placeholder: no sigma schedule; falls back to dt
                step_size = times_for_step[i + 1] - times_for_step[i]
            else:
                raise ValueError(f"Unknown step_size_type {step_size_type}")
            x = x + v_pred * step_size

        return x

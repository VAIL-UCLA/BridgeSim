"""
Flow Matching model that wraps a DiT backbone for BEV-to-BEV alignment.

Notes:
- DiT forward expects `(B, C, H, W)` and a timestep vector `t` sampled in the training loop (Uniform[0, 1]).
- `psi` and `Dt_psi` mirror the CrossFlow flow-matching interpolation; they are shared so the Lightning loop can build
  `x_t` and targets consistently.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn

from dit import DiT_models


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        *,
        default_bev_hw: Tuple[int, int] = (200, 200),
        bev_channels: int = 256,
        dit_variant: str = "DiT-B/2",
        learn_sigma: bool = False,
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
    ) -> None:
        """
        Args:
            default_bev_hw: Fallback BEV spatial size (H, W) if it cannot be inferred.
            dit_variant: Key into dit.DiT_models. Kept configurable for future tuning.
            learn_sigma: Whether DiT predicts both mean and sigma; usually False here.
            patch_size: DiT patch size; keep small for large BEV grids.
        """
        super().__init__()
        self.default_bev_hw = default_bev_hw
        self.default_bev_channels = bev_channels
        self.learn_sigma = learn_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if dit_variant not in DiT_models:
            raise ValueError(f"Unknown DiT variant '{dit_variant}'. Available: {list(DiT_models.keys())}")
        self.dit_variant = dit_variant
        self.dit_builder = DiT_models[dit_variant]

        latent_size = default_bev_hw[0]
        # TODO: replace cfg with task-specific DiT config/weights when available.
        cfg = SimpleNamespace(latent_size=latent_size, channels=bev_channels, learn_sigma=self.learn_sigma)
        self.dit: Optional[nn.Module] = self.dit_builder(config=cfg)
        self._latent_hw: Optional[int] = latent_size
        self._channels: Optional[int] = bev_channels

    def _infer_hw(self, tokens: int) -> int:
        h = int(math.isqrt(tokens))
        if h * h != tokens:
            raise ValueError(f"Cannot reshape BEV tokens={tokens} into a square grid.")
        return h

    def _to_bchw(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Accepts (H*W, B, C), (B, H*W, C), or (B, C, H, W) and returns (B, C, H, W).
        """
        if bev.dim() == 4:
            # Already in (B, C, H, W) format
            return bev
        if bev.dim() == 3:
            # Check if it's (H*W, B, C) or (B, H*W, C)
            # Heuristic: if first dimension is much larger than second, it's (H*W, B, C)
            # Otherwise, if second dimension is large (like 40000 for 200x200), it's (B, H*W, C)
            if bev.shape[1] > bev.shape[0] and bev.shape[1] > 1000:
                # Format: (B, H*W, C) from precomputed cache
                bsz, tokens, channels = bev.shape
                h = self._infer_hw(tokens)
                w = h
                # Reshape: (B, H*W, C) -> (B, C, H, W)
                bev = bev.permute(0, 2, 1).reshape(bsz, channels, h, w)
                return bev
            else:
                # Format: (H*W, B, C) from live extraction
                tokens, bsz, channels = bev.shape
                h = self._infer_hw(tokens)
                w = h
                # Reshape: (H*W, B, C) -> (B, C, H, W)
                bev = bev.permute(1, 2, 0).reshape(bsz, channels, h, w)
                return bev
        raise ValueError(f"Unsupported BEV shape {bev.shape}")

    def reshape_bev(self, bev: torch.Tensor) -> torch.Tensor:
        return self._to_bchw(bev)

    def psi(self, t: torch.Tensor, noise: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Interpolation used for flow-matching: psi(t, noise, x1).
        """
        t_expanded = self.expand_t(t, noise)
        return (t_expanded * (self.sigma_min / self.sigma_max - 1) + 1) * noise + t_expanded * x1

    def dt_psi(self, t: torch.Tensor, noise: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Time derivative of psi w.r.t. t (target velocity).
        """
        _ = self.expand_t(t, noise)  # shapes are validated in expand_t
        return (self.sigma_min / self.sigma_max - 1) * noise + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
        Args:
            x_t: Interpolated/noised BEV (H*W, B, C) or (B, C, H, W).
            t: Timesteps (B,) sampled in the training loop.
        Returns:
            Velocity prediction from DiT, shaped like `x_t`.
        """
        x_bchw = self._to_bchw(x_t)
        bsz, channels, h, w = x_bchw.shape
        if channels != self._channels or h != self._latent_hw or w != self._latent_hw:
            raise ValueError(
                f"DiT initialized for (C,H,W)=({self._channels},{self._latent_hw},{self._latent_hw}); "
                f"got ({channels},{h},{w}). Update FlowMatchingModel config before training."
            )
        if self.dit is None:
            raise RuntimeError("DiT backbone was not initialized.")
        self.dit = self.dit.to(x_bchw.device)

        if t.dim() != 1 or t.shape[0] != bsz:
            raise ValueError(f"t must be shape (B,), got {t.shape} for batch {bsz}")
        null_indicator = torch.zeros_like(t, dtype=torch.long)

        dit_out = self.dit(x_bchw, t=t, null_indicator=null_indicator)
        pred = dit_out[0] if isinstance(dit_out, (list, tuple)) else dit_out
        return pred

    def sample(
        self,
        src_bev: torch.Tensor,
        *,
        sample_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
        step_size_type: str = "step_in_dt",
    ) -> torch.Tensor:
        """
        Euler sampler for flow matching, inspired by CrossFlow's ODEEulerFlowMatchingSolver.

        Args:
            src_bev: Source BEV embedding (H*W, B, C) or (B, C, H, W) treated as the starting state.
            sample_steps: Number of Euler steps between t_start and t_end.
            t_start: Start of integration window.
            t_end: End of integration window.
            step_size_type: "step_in_dt" or "step_in_dsigma" (currently only dt is used).
        """
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive.")
        x = self.reshape_bev(src_bev)
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

"""
BEV Calibrator for domain adaptation from MetaDrive to Bench2Drive.

Uses flow matching to calibrate BEV features extracted from images,
improving model performance by adapting MetaDrive representations to Bench2Drive domain.

Reference files:
- /calibration/precompute_bev_cache.py - BEV extraction from images
- /calibration/test_bev_flow_sampling.py:L194 - sampling function
- /calibration/train_uniad_bev.py:L715 - occupancy/detection heads
"""

import sys
import torch
from pathlib import Path
from typing import Optional

# Add calibration directory to path (evaluation/ is at top level, calibration/ is sibling)
CALIBRATION_DIR = Path(__file__).resolve().parent.parent.parent / "calibration"
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_DIR))

try:
    from train_bev_flow_pl import FlowMatchingLit
except ImportError as e:
    raise ImportError(f"Cannot import FlowMatchingLit. Make sure calibration directory exists at {CALIBRATION_DIR}") from e


class BEVCalibrator:
    """
    BEV domain adaptation calibrator using flow matching.

    This calibrator takes BEV features extracted from MetaDrive images and
    transforms them to the Bench2Drive domain using a trained flow matching model.
    """

    def __init__(self,
                 checkpoint_path: str,
                 sample_steps: int = 50,
                 use_ema: bool = True,
                 device: str = "cuda"):
        """
        Initialize BEV calibrator.

        Args:
            checkpoint_path: Path to flow matching checkpoint (.ckpt file)
            sample_steps: Number of Euler sampling steps for flow matching
            use_ema: Use EMA weights if available
            device: Device to run on (cuda/cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.device = torch.device(device)

        self.lit_model = None
        self.sampler_model = None

        print(f"[BEVCalibrator] Initialized with checkpoint: {checkpoint_path}")
        print(f"[BEVCalibrator] Sample steps: {sample_steps}, Use EMA: {use_ema}")

    def load_model(self):
        """Load flow matching model from checkpoint."""
        if self.lit_model is not None:
            print("[BEVCalibrator] Model already loaded, skipping.")
            return

        print(f"[BEVCalibrator] Loading checkpoint: {self.checkpoint_path}")

        # Load PyTorch Lightning checkpoint
        self.lit_model = FlowMatchingLit.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        self.lit_model = self.lit_model.to(self.device)
        self.lit_model.eval()

        # Choose sampler (EMA or normal)
        if self.use_ema and hasattr(self.lit_model, 'ema_flow_model'):
            print("[BEVCalibrator] Using EMA model for sampling")
            self.sampler_model = self.lit_model.ema_flow_model
        else:
            print("[BEVCalibrator] Using non-EMA model for sampling")
            self.sampler_model = self.lit_model.flow_model

        self.sampler_model = self.sampler_model.to(self.device)
        self.sampler_model.eval()

        # Freeze all parameters
        for p in self.sampler_model.parameters():
            p.requires_grad_(False)

        print("[BEVCalibrator] Model loaded successfully")

    @torch.no_grad()
    def calibrate(self, bev_embed: torch.Tensor) -> torch.Tensor:
        """
        Calibrate BEV embedding from MetaDrive domain to Bench2Drive domain.

        Args:
            bev_embed: BEV embedding tensor from UniAD model
                      Expected shape: (H*W, B, C) format

        Returns:
            Calibrated BEV embedding in same format as input (H*W, B, C)
        """
        if self.sampler_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Store original properties
        original_shape = bev_embed.shape
        original_device = bev_embed.device
        original_format_is_hwbc = (bev_embed.dim() == 3 and bev_embed.shape[0] > bev_embed.shape[1])

        # Move to calibrator device
        bev_embed = bev_embed.to(self.device)

        # Apply flow matching - sampler_model.sample() returns in (B, C, H, W) format
        calibrated_bev = self.sampler_model.sample(bev_embed, sample_steps=self.sample_steps)

        # Convert back to original format if needed
        if original_format_is_hwbc:
            # Convert from (B, C, H, W) back to (H*W, B, C)
            bsz, channels, h, w = calibrated_bev.shape
            calibrated_bev = calibrated_bev.reshape(bsz, channels, h * w).permute(2, 0, 1)

        # Move back to original device
        calibrated_bev = calibrated_bev.to(original_device)

        # Verify output shape matches input
        if calibrated_bev.shape != original_shape:
            print(f"[BEVCalibrator] Warning: Output shape {calibrated_bev.shape} != input shape {original_shape}")

        return calibrated_bev

    @torch.no_grad()
    def calibrate_with_reshape(self, bev_embed: torch.Tensor) -> torch.Tensor:
        """
        Calibrate BEV embedding and reshape to (B, C, H, W) format.

        This method handles the reshaping needed when working with UniAD BEV features,
        similar to test_bev_flow_sampling.py:L195-196.

        Args:
            bev_embed: BEV embedding tensor from UniAD model

        Returns:
            Calibrated BEV embedding in (B, C, H, W) format
        """
        if self.sampler_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        original_device = bev_embed.device
        bev_embed = bev_embed.to(self.device)

        # Apply flow matching
        calibrated_bev = self.sampler_model.sample(bev_embed, sample_steps=self.sample_steps)

        # Reshape to (B, C, H, W)
        calibrated_bev_bchw = self.lit_model.flow_model.reshape_bev(calibrated_bev)

        # Move back to original device
        calibrated_bev_bchw = calibrated_bev_bchw.to(original_device)

        return calibrated_bev_bchw

    def __repr__(self):
        return (f"BEVCalibrator(checkpoint={self.checkpoint_path}, "
                f"sample_steps={self.sample_steps}, use_ema={self.use_ema}, "
                f"device={self.device})")


def create_bev_calibrator(checkpoint_path: Optional[str] = None,
                         sample_steps: int = 50,
                         use_ema: bool = True,
                         device: str = "cuda") -> Optional[BEVCalibrator]:
    """
    Factory function to create BEV calibrator.

    Args:
        checkpoint_path: Path to flow matching checkpoint. If None, returns None.
        sample_steps: Number of Euler sampling steps
        use_ema: Use EMA weights if available
        device: Device to run on

    Returns:
        BEVCalibrator instance or None if checkpoint_path is None
    """
    if checkpoint_path is None:
        return None

    checkpoint_path = str(checkpoint_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"BEV calibrator checkpoint not found: {checkpoint_path}")

    calibrator = BEVCalibrator(
        checkpoint_path=checkpoint_path,
        sample_steps=sample_steps,
        use_ema=use_ema,
        device=device
    )

    return calibrator

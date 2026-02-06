"""
TransFuser BEV Calibrator for domain adaptation from MetaDrive to NavSim.

Uses flow matching to calibrate TransFuser-style BEV features (fused_features)
extracted from the backbone, improving model performance by adapting
MetaDrive representations to NavSim domain.

This calibrator is designed for DiffusionDriveV2 and other models using
the TransFuser backbone with BEV shape (B, 512, 8, 8).

Reference files:
- /calibration/precompute_transfuser_bev_cache.py - BEV extraction from backbone
- /calibration/test_transfuser_bev_flow.py - evaluation and sampling
- /calibration/train_transfuser_bev_flow.py - flow matching training
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
    from train_transfuser_bev_flow import FlowMatchingLit
except ImportError as e:
    raise ImportError(
        f"Cannot import FlowMatchingLit from train_transfuser_bev_flow. "
        f"Make sure calibration directory exists at {CALIBRATION_DIR}"
    ) from e


class TransfuserBEVCalibrator:
    """
    TransFuser BEV domain adaptation calibrator using flow matching.

    This calibrator takes fused_features (BEV) extracted from the TransFuser
    backbone and transforms them from MetaDrive domain to NavSim domain
    using a trained flow matching model.

    BEV format: (B, 512, 8, 8) - batch, channels, height, width
    """

    def __init__(
        self,
        checkpoint_path: str,
        sample_steps: int = 50,
        use_ema: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize TransFuser BEV calibrator.

        Args:
            checkpoint_path: Path to flow matching checkpoint (.ckpt file)
            sample_steps: Number of Euler sampling steps for flow matching
            use_ema: Use EMA weights if available (recommended)
            device: Device to run on (cuda/cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.device = torch.device(device)

        self.lit_model = None
        self.sampler_model = None

        print(f"[TransfuserBEVCalibrator] Initialized with checkpoint: {checkpoint_path}")
        print(f"[TransfuserBEVCalibrator] Sample steps: {sample_steps}, Use EMA: {use_ema}")

    def load_model(self):
        """Load flow matching model from checkpoint."""
        if self.lit_model is not None:
            print("[TransfuserBEVCalibrator] Model already loaded, skipping.")
            return

        print(f"[TransfuserBEVCalibrator] Loading checkpoint: {self.checkpoint_path}")

        # Load PyTorch Lightning checkpoint
        self.lit_model = FlowMatchingLit.load_from_checkpoint(
            self.checkpoint_path,
            map_location=self.device
        )
        self.lit_model = self.lit_model.to(self.device)
        self.lit_model.eval()

        # Choose sampler (EMA or normal)
        if self.use_ema and hasattr(self.lit_model, 'ema_flow_model') and self.lit_model.ema_flow_model is not None:
            print("[TransfuserBEVCalibrator] Using EMA model for sampling")
            self.sampler_model = self.lit_model.ema_flow_model
        else:
            print("[TransfuserBEVCalibrator] Using non-EMA model for sampling")
            self.sampler_model = self.lit_model.flow_model

        self.sampler_model = self.sampler_model.to(self.device)
        self.sampler_model.eval()

        # Freeze all parameters
        for p in self.sampler_model.parameters():
            p.requires_grad_(False)

        print("[TransfuserBEVCalibrator] Model loaded successfully")

    @torch.no_grad()
    def calibrate(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """
        Calibrate BEV feature from MetaDrive domain to NavSim domain.

        This method applies flow matching to transform the fused_features
        from TransFuser backbone to the target domain.

        Args:
            bev_feature: BEV feature tensor from TransFuser backbone
                        Shape: (B, 512, 8, 8) - batch, channels, height, width

        Returns:
            Calibrated BEV feature in same format (B, 512, 8, 8)
        """
        if self.sampler_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Store original properties
        original_shape = bev_feature.shape
        original_device = bev_feature.device

        # Move to calibrator device
        bev_feature = bev_feature.to(self.device)

        # Apply flow matching sampling
        # The sampler expects (B, C, H, W) format which matches our input
        calibrated_bev = self.sampler_model.sample(
            bev_feature,
            sample_steps=self.sample_steps
        )

        # Move back to original device
        calibrated_bev = calibrated_bev.to(original_device)

        # Verify output shape matches input
        if calibrated_bev.shape != original_shape:
            print(
                f"[TransfuserBEVCalibrator] Warning: Output shape {calibrated_bev.shape} "
                f"!= input shape {original_shape}"
            )

        return calibrated_bev

    def __repr__(self):
        return (
            f"TransfuserBEVCalibrator(checkpoint={self.checkpoint_path}, "
            f"sample_steps={self.sample_steps}, use_ema={self.use_ema}, "
            f"device={self.device})"
        )


def create_transfuser_bev_calibrator(
    checkpoint_path: Optional[str] = None,
    sample_steps: int = 50,
    use_ema: bool = True,
    device: str = "cuda",
) -> Optional[TransfuserBEVCalibrator]:
    """
    Factory function to create TransFuser BEV calibrator.

    Args:
        checkpoint_path: Path to flow matching checkpoint. If None, returns None.
        sample_steps: Number of Euler sampling steps
        use_ema: Use EMA weights if available
        device: Device to run on

    Returns:
        TransfuserBEVCalibrator instance or None if checkpoint_path is None
    """
    if checkpoint_path is None:
        return None

    checkpoint_path = str(checkpoint_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"TransFuser BEV calibrator checkpoint not found: {checkpoint_path}"
        )

    calibrator = TransfuserBEVCalibrator(
        checkpoint_path=checkpoint_path,
        sample_steps=sample_steps,
        use_ema=use_ema,
        device=device,
    )

    return calibrator
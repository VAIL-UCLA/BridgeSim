import torch
from typing import Dict, Any

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer


class ConfidenceScorer(BaseTrajectoryScorer):
    """
    Scorer that uses the model's confidence scores (poses_cls) to select the
    best trajectory candidate. This is only applicable to DiffusionDrive v1,
    which produces classification logits for each trajectory mode.

    For DiffusionDrive v2, confidence_scores will be None and this scorer
    will raise an error.
    """

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select the trajectory with the highest confidence score.

        Args:
            model_output: dict from forward_inference_scaling

        Returns:
            dict with "trajectory", "scores", "best_idx"
        """
        scores = model_output.get("confidence_scores")
        if scores is None:
            raise ValueError(
                "Confidence scores not available. "
                "This scorer only works with DiffusionDrive v1."
            )

        candidates = model_output["all_candidates"]  # (B, N, 8, 3)
        bs = candidates.shape[0]
        device = candidates.device

        best_idx = scores.argmax(dim=-1)  # (B,)
        best_traj = candidates[
            torch.arange(bs, device=device), best_idx
        ]  # (B, 8, 3)

        return {
            "trajectory": best_traj,
            "scores": scores,
            "best_idx": best_idx,
        }

import torch
from typing import Dict, Any

from bridgesim.evaluation.scorers.base_scorer import BaseTrajectoryScorer


class ClsScorer(BaseTrajectoryScorer):
    """
    Baseline scorer that selects the best trajectory using the model's own
    classification logits (confidence scores).

    During inference scaling, the model generates N candidate trajectories and
    assigns a confidence score to each. This scorer simply picks the candidate
    with the highest score — equivalent to the model's default selection, but
    applied within the multi-candidate inference scaling framework.

    This serves as a baseline for comparing against more sophisticated external
    scorers (e.g. rule-based or learned scorers).

    Note: requires the model to output per-trajectory confidence scores.
    """

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select the trajectory candidate with the highest confidence score.

        Args:
            model_output: dict containing:
                - "all_candidates": (B, N, T, 3) tensor of N trajectory candidates
                - "confidence_scores": (B, N) per-candidate classification scores

        Returns:
            dict with:
                - "trajectory": (B, T, 3) selected trajectory
                - "scores": (B, N) confidence scores
                - "best_idx": (B,) index of selected candidate
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

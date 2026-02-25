from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTrajectoryScorer(ABC):
    """
    Base class for trajectory candidate scorers.

    Scorers receive the output of forward_inference_scaling (which contains
    all trajectory candidates and model-computed scores) and select the best
    trajectory using a scoring strategy.
    """

    @abstractmethod
    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select the best trajectory from candidates.

        Args:
            model_output: dict from forward_inference_scaling containing:
                - "all_candidates": (B, N, 8, 3) all trajectory candidates
                - "confidence_scores": (B, N) or None — poses_cls from v1
                - "coarse_scores": (B, N) or None — learned coarse scorer from v2
                - "scorer_context": dict of intermediate features for external scorers
            **kwargs: scorer-specific arguments (e.g., ego_state, frame_idx)

        Returns:
            dict with:
                - "trajectory": (B, 8, 3) selected best trajectory
                - "scores": (B, N) all candidate scores
                - "best_idx": (B,) index of selected candidate
        """
        pass

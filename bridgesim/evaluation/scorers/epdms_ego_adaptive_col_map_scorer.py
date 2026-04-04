"""
EPDMS Ego Adaptive Collision + Map Awareness Scorer — Ablation: collision + map rewards.

Subclasses EPDMSEgoAdaptiveColScorer and overrides only the scoring formula.

Scoring formula:
  COL × DAC × DDC × TLC

Used for ablation study to measure the combined contribution of collision
avoidance (COL) and map awareness (DAC, DDC, TLC) before adding comfort
rewards (EP, LK, HC, EC).
"""

from bridgesim.evaluation.scorers.epdms_ego_adaptive_col_scorer import EPDMSEgoAdaptiveColScorer


class EPDMSEgoAdaptiveColMapScorer(EPDMSEgoAdaptiveColScorer):
    """Ablation scorer: collision + map awareness. Score = COL × DAC × DDC × TLC."""

    @staticmethod
    def _metrics_to_score(metrics: dict) -> float:
        return metrics['col'] * metrics['dac'] * metrics['ddc'] * metrics['tlc']

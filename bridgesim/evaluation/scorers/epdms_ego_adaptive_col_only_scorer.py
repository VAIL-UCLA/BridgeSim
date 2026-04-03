"""
EPDMS Ego Adaptive Collision-Only Scorer — Ablation: collision reward only.

Subclasses EPDMSEgoAdaptiveColScorer and overrides only the scoring formula.

Scoring formula:
  COL

Used for ablation study to measure the contribution of the collision reward
component in isolation, before adding map awareness (DAC, DDC, TLC) and
comfort rewards (EP, LK, HC, EC).
"""

from bridgesim.evaluation.scorers.epdms_ego_adaptive_col_scorer import EPDMSEgoAdaptiveColScorer


class EPDMSEgoAdaptiveColOnlyScorer(EPDMSEgoAdaptiveColScorer):
    """Ablation scorer: collision reward only. Score = COL."""

    @staticmethod
    def _metrics_to_score(metrics: dict) -> float:
        return metrics['col']

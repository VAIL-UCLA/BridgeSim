"""
EPDMS Ego Adaptive Comfort-Only Scorer — Ablation: comfort reward only.

Scoring formula:
  (2×HC + 2×EC) / 4
"""

from bridgesim.evaluation.scorers.epdms_ego_adaptive_col_scorer import EPDMSEgoAdaptiveColScorer

W_HISTORY_COMFORT = 2.0
W_EXTENDED_COMFORT = 2.0
W_TOTAL = W_HISTORY_COMFORT + W_EXTENDED_COMFORT


class EPDMSEgoAdaptiveComfortOnlyScorer(EPDMSEgoAdaptiveColScorer):
    """Ablation scorer: comfort only. Score = (2×HC + 2×EC) / 4."""

    @staticmethod
    def _metrics_to_score(metrics: dict) -> float:
        return (W_HISTORY_COMFORT * metrics['hc'] + W_EXTENDED_COMFORT * metrics['ec']) / W_TOTAL

"""
EPDMS Ego Adaptive Comfort + Map Awareness Scorer — Ablation.

Scoring formula:
  (DAC × DDC × TLC) × (2×HC + 2×EC) / 4
"""

from bridgesim.evaluation.scorers.epdms_ego_adaptive_col_scorer import EPDMSEgoAdaptiveColScorer

W_HISTORY_COMFORT = 2.0
W_EXTENDED_COMFORT = 2.0
W_TOTAL = W_HISTORY_COMFORT + W_EXTENDED_COMFORT


class EPDMSEgoAdaptiveComfortMapScorer(EPDMSEgoAdaptiveColScorer):
    """Ablation scorer: comfort + map awareness. Score = (DAC×DDC×TLC) × (2×HC+2×EC)/4."""

    @staticmethod
    def _metrics_to_score(metrics: dict) -> float:
        multi_prod = metrics['dac'] * metrics['ddc'] * metrics['tlc']
        weighted_sum = W_HISTORY_COMFORT * metrics['hc'] + W_EXTENDED_COMFORT * metrics['ec']
        return multi_prod * weighted_sum / W_TOTAL

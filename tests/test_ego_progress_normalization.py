"""Ego-progress normalisation, across the three BridgeSim scorers.

NAVSIM (arXiv:2406.15349) defines ego progress as a RATIO to an achievable
upper bound, not an absolute distance:

    "The ego progress subscore represents the agent progress along the route
    center as a ratio to an approximated safe upper bound [...] The final ratio
    is clipped to [0, 1] while discarding low or negative progress scores if
    the upper bound is below 5 meters."

These tests pin the two variants BridgeSim needs:

* ``EPDMSScorer``  -- reported metric, one trajectory scored against a
  reference bound (NAVSIM's agent-evaluation path).
* ``GTScorer`` / ``TTAScorer`` -- candidate rankers, normalised against the
  best candidate in the batch (NAVSIM's proposal-scoring path).

The scorer modules are loaded directly from source: importing them through
their packages executes ``bridgesim/evaluation/utils/__init__.py``, which pulls
in metadrive and panda3d. Nothing under test needs either.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_namespace_stubs():
    """Register package stubs so submodule imports resolve without __init__."""
    packages = {
        "bridgesim": _REPO_ROOT / "bridgesim",
        "bridgesim.evaluation": _REPO_ROOT / "bridgesim" / "evaluation",
        "bridgesim.evaluation.scorers": _REPO_ROOT / "bridgesim" / "evaluation" / "scorers",
        "bridgesim.evaluation.utils": _REPO_ROOT / "bridgesim" / "evaluation" / "utils",
    }
    for name, path in packages.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module


_install_namespace_stubs()

EPDMS = importlib.import_module("bridgesim.evaluation.utils.epdms_scorer_md")
GT = importlib.import_module("bridgesim.evaluation.scorers.GT_scorer")
TTA = importlib.import_module("bridgesim.evaluation.scorers.TTA_scorer")

THRESHOLD = 5.0


# ----------------------------------------------------------------------
# Shared contract: all three scorers agree on the threshold
# ----------------------------------------------------------------------

def test_all_scorers_share_the_navsim_threshold():
    """A single threshold, so the three scorers stay mutually comparable."""
    assert EPDMS.PROGRESS_DISTANCE_THRESHOLD_M == THRESHOLD
    assert GT.PROGRESS_DISTANCE_THRESHOLD_M == THRESHOLD
    assert TTA.PROGRESS_DISTANCE_THRESHOLD_M == THRESHOLD


# ----------------------------------------------------------------------
# EPDMSScorer -- reported metric, reference-normalised
# ----------------------------------------------------------------------

def test_progress_is_a_ratio_to_the_reference():
    """Half the reference's progress scores half the points."""
    assert EPDMS.EPDMSScorer._normalize_progress(10.0, 1.0, 20.0) == pytest.approx(0.5)


def test_beating_the_reference_saturates_at_one():
    """The denominator is max(agent, reference), so outperforming caps at 1.0.

    It must not exceed 1.0, and it must not be clipped down to the reference
    ratio either -- an agent 25% faster than the bound still scores exactly 1.0.
    """
    assert EPDMS.EPDMSScorer._normalize_progress(25.0, 1.0, 20.0) == pytest.approx(1.0)


def test_disqualified_trajectory_earns_no_progress():
    """multi_prod == 0 masks progress: a collision cannot buy progress credit."""
    assert EPDMS.EPDMSScorer._normalize_progress(30.0, 0.0, 20.0) == 0.0


def test_standstill_frame_is_not_penalised():
    """The regression this change exists for.

    At a standstill launch neither the agent nor the reference can cover a
    meaningful distance, so progress cannot discriminate and must go neutral.
    Under the previous fixed 30 m normalisation this frame scored 10.5/30 =
    0.35 and dragged the whole frame's EPDMS down for physics no trajectory
    could beat.
    """
    assert EPDMS.EPDMSScorer._normalize_progress(4.0, 1.0, 4.5) == 1.0
    # The old behaviour, pinned so a regression is unambiguous.
    assert min(4.0 / 30.0, 1.0) == pytest.approx(0.1333, abs=1e-4)


def test_standstill_does_not_absolve_a_disqualified_trajectory():
    """Below threshold, a collided trajectory still scores 0, not the neutral 1."""
    assert EPDMS.EPDMSScorer._normalize_progress(4.0, 0.0, 4.5) == 0.0


def test_threshold_boundary_is_strict():
    """The bound must EXCEED the threshold; exactly 5 m is still 'no progress'."""
    assert EPDMS.EPDMSScorer._normalize_progress(5.0, 1.0, 5.0) == 1.0
    just_over = EPDMS.EPDMSScorer._normalize_progress(2.5, 1.0, 5.0 + 1e-6)
    assert just_over == pytest.approx(0.5, abs=1e-6)


def test_zero_progress_against_a_moving_reference_scores_zero():
    """Stopping when the reference drove 20 m is a real progress failure."""
    assert EPDMS.EPDMSScorer._normalize_progress(0.0, 1.0, 20.0) == 0.0


def test_unsafe_reference_progress_is_not_a_valid_bound():
    """A human who violated a hard constraint must not set the bar.

    NAVSIM's bound is PDM-Closed progress "without collisions or off-road
    driving". A logged driver who gained 30 m by running a red light is not a
    safe upper bound, and an agent that legally covered 12 m must not be
    scored 0.4 against it. ``score_frame`` drops such a reference to 0.0,
    which is exercised here through the normalisation contract.
    """
    unsafe_bound_dropped = EPDMS.EPDMSScorer._normalize_progress(12.0, 1.0, 0.0)
    assert unsafe_bound_dropped == pytest.approx(1.0)
    # Had the unsafe 30 m been kept, the agent would have been punished:
    assert EPDMS.EPDMSScorer._normalize_progress(12.0, 1.0, 30.0) == pytest.approx(0.4)


def test_no_reference_available_falls_back_to_self_normalisation():
    """With no reference (bound 0), a moving agent is its own bound => 1.0.

    This is the degenerate case where the human trajectory is missing or
    truncated at the end of a log. It must not crash or divide by zero.
    """
    assert EPDMS.EPDMSScorer._normalize_progress(12.0, 1.0, 0.0) == pytest.approx(1.0)
    assert EPDMS.EPDMSScorer._normalize_progress(0.0, 1.0, 0.0) == 1.0


# ----------------------------------------------------------------------
# GTScorer -- candidate ranker, batch-relative
# ----------------------------------------------------------------------

def test_batch_best_candidate_scores_one_and_others_are_proportional():
    raw = np.array([0.0, 4.0, 7.0, 10.0])
    multi = np.ones(4)
    ep = GT.GTScorer._normalize_batch_progress(raw, multi)
    assert ep[3] == pytest.approx(1.0)
    assert ep == pytest.approx([0.0, 0.4, 0.7, 1.0])


def test_disqualified_candidate_cannot_set_the_batch_bound():
    """A collided candidate that drove furthest must not rescale the others.

    Without masking, the 40 m collided candidate would become the denominator
    and push the best legal candidate down to 0.25.
    """
    raw = np.array([40.0, 10.0, 8.0])
    multi = np.array([0.0, 1.0, 1.0])
    ep = GT.GTScorer._normalize_batch_progress(raw, multi)
    assert ep[0] == 0.0
    assert ep[1] == pytest.approx(1.0)
    assert ep[2] == pytest.approx(0.8)


def test_batch_below_threshold_goes_neutral():
    """Every candidate stationary => progress cannot rank them, so all 1.0."""
    raw = np.array([0.1, 0.5, 2.0])
    multi = np.ones(3)
    ep = GT.GTScorer._normalize_batch_progress(raw, multi)
    assert ep == pytest.approx([1.0, 1.0, 1.0])


def test_batch_below_threshold_still_zeroes_disqualified():
    raw = np.array([0.1, 2.0])
    multi = np.array([0.0, 1.0])
    ep = GT.GTScorer._normalize_batch_progress(raw, multi)
    assert ep[0] == 0.0
    assert ep[1] == 1.0


def test_batch_normalisation_never_leaves_the_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 12))
        raw = rng.uniform(0.0, 60.0, size=n)
        multi = (rng.uniform(size=n) > 0.3).astype(float)
        ep = GT.GTScorer._normalize_batch_progress(raw, multi)
        assert np.all(ep >= 0.0) and np.all(ep <= 1.0)


# ----------------------------------------------------------------------
# TTAScorer -- shared denominator across comparison stages
# ----------------------------------------------------------------------

def test_tta_shared_denominator_preserves_ordering():
    """Two trajectories compared on one bound rank by actual progress."""
    worse = TTA.TTAScorer._normalize_progress(6.0, 1.0, 12.0)
    better = TTA.TTAScorer._normalize_progress(12.0, 1.0, 12.0)
    assert worse == pytest.approx(0.5)
    assert better == pytest.approx(1.0)
    assert better > worse


def test_tta_per_call_denominators_would_be_incomparable():
    """Why the denominator is threaded through rather than recomputed.

    Scored against itself, a 6 m trajectory looks identical to a 12 m one.
    This is the bug the shared ``progress_norm_m`` exists to prevent, and it is
    pinned here so nobody 'simplifies' the parameter away.
    """
    self_normalised_worse = TTA.TTAScorer._normalize_progress(6.0, 1.0, 6.0)
    self_normalised_better = TTA.TTAScorer._normalize_progress(12.0, 1.0, 12.0)
    assert self_normalised_worse == self_normalised_better  # both 1.0 -- useless


def test_tta_metrics_to_score_uses_the_shared_bound():
    """``_metrics_to_score`` must fill ``ep`` from raw metres, not read a stale value."""
    metrics = {
        "col": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
        "ep": 0.0, "ep_raw_m": 6.0, "lk": 1.0, "hc": 1.0, "ec": 1.0,
    }
    score = TTA.TTAScorer._metrics_to_score(metrics, progress_norm_m=12.0)
    assert metrics["ep"] == pytest.approx(0.5)
    expected = (TTA.W_PROGRESS * 0.5 + TTA.W_LANE_KEEPING
                + TTA.W_HISTORY_COMFORT + TTA.W_EXTENDED_COMFORT) / TTA.W_TOTAL
    assert score == pytest.approx(expected)


def test_tta_disqualified_scores_zero_regardless_of_progress():
    metrics = {
        "col": 0.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
        "ep": 0.0, "ep_raw_m": 30.0, "lk": 1.0, "hc": 1.0, "ec": 1.0,
    }
    assert TTA.TTAScorer._metrics_to_score(metrics, progress_norm_m=30.0) == 0.0
    assert metrics["ep"] == 0.0

"""Tests for PDM-Closed helper classes."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from bridgesim.evaluation.models.pdm_closed_adapter import (
    _IDMController,
    _KinematicBicycleSimulator,
    _CenterlineExtractor,
    _PDMClosedScorer,
    PDMClosedAdapter,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_scenario(n_frames=100, speed_mps=0.5, heading=0.0):
    positions = np.zeros((n_frames, 3))
    positions[:, 0] = np.arange(n_frames) * speed_mps * math.cos(heading)
    positions[:, 1] = np.arange(n_frames) * speed_mps * math.sin(heading)
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": positions,
                    "valid":    np.ones(n_frames, dtype=bool),
                    "heading":  np.full(n_frames, heading),
                },
            }
        },
        "dynamic_map_states": {},
        "length": n_frames,
    }


def _make_ego_state(x=0.0, y=0.0, heading=0.0, speed=5.0):
    return {
        "position":         np.array([x, y, 0.0]),
        "heading":          heading,
        "speed":            speed,
        "velocity":         np.array([speed * math.cos(heading),
                                      speed * math.sin(heading), 0.0]),
        "acceleration":     np.zeros(3),
        "angular_velocity": np.zeros(3),
        "waypoint":         np.array([x + 20.0, y]),
        "command":          2,
    }


def _make_straight_path(length=60.0, spacing=0.5, heading=0.0):
    n  = int(length / spacing) + 1
    ts = np.linspace(0, length, n)
    return np.stack([ts * math.cos(heading), ts * math.sin(heading)], axis=1)


# ===========================================================================
# _IDMController tests
# ===========================================================================

def test_idm_free_road_accelerates():
    idm = _IDMController(v_desired=8.33, a_max=2.0)
    a = idm.compute_acceleration(ego_speed=0.0, gap=math.inf, rel_speed=0.0)
    assert a == pytest.approx(idm.a_max)


def test_idm_at_desired_speed_no_leader():
    idm = _IDMController(v_desired=8.33, a_max=2.0)
    a = idm.compute_acceleration(ego_speed=8.33, gap=math.inf, rel_speed=0.0)
    assert abs(a) < 1e-9


def test_idm_close_leader_decelerates():
    idm = _IDMController(v_desired=8.33, a_max=2.0, b=3.0)
    a = idm.compute_acceleration(ego_speed=5.0, gap=0.5, rel_speed=5.0)
    assert a < -1.0


def test_idm_output_clamped_to_b():
    idm = _IDMController(b=3.0)
    a = idm.compute_acceleration(ego_speed=10.0, gap=0.01, rel_speed=10.0)
    assert a >= -idm.b - 1e-9


def test_idm_output_clamped_to_a_max():
    idm = _IDMController(a_max=2.0)
    a = idm.compute_acceleration(ego_speed=0.0, gap=math.inf, rel_speed=-10.0)
    assert a <= idm.a_max + 1e-9


def test_idm_find_leader_ahead():
    idm = _IDMController()
    agents = [{"position": np.array([10.0, 0.0]), "velocity": np.zeros(2)}]
    gap, rel = idm.find_leading_vehicle(0.0, 0.0, 0.0, 5.0, agents)
    assert gap < math.inf and rel > 0.0


def test_idm_find_leader_behind_ignored():
    idm = _IDMController()
    agents = [{"position": np.array([-10.0, 0.0]), "velocity": np.zeros(2)}]
    gap, rel = idm.find_leading_vehicle(0.0, 0.0, 0.0, 5.0, agents)
    assert gap == math.inf


def test_idm_find_leader_outside_corridor():
    idm = _IDMController(corridor_half_width=1.5)
    agents = [{"position": np.array([10.0, 5.0]), "velocity": np.zeros(2)}]
    gap, rel = idm.find_leading_vehicle(0.0, 0.0, 0.0, 5.0, agents)
    assert gap == math.inf


def test_idm_find_closest_of_multiple():
    idm = _IDMController()
    agents = [
        {"position": np.array([30.0, 0.0]), "velocity": np.zeros(2)},
        {"position": np.array([10.0, 0.0]), "velocity": np.zeros(2)},
        {"position": np.array([20.0, 0.0]), "velocity": np.zeros(2)},
    ]
    gap, _ = idm.find_leading_vehicle(0.0, 0.0, 0.0, 5.0, agents)
    assert gap < 10.0


def test_idm_no_agents_returns_inf():
    idm = _IDMController()
    gap, rel = idm.find_leading_vehicle(0.0, 0.0, 0.0, 5.0, [])
    assert gap == math.inf and rel == 0.0


def test_idm_find_leader_heading_north():
    idm = _IDMController()
    # Ego faces north (π/2); agent is directly north at (0, 10)
    agents = [{"position": np.array([0.0, 10.0]), "velocity": np.zeros(2)}]
    gap, rel = idm.find_leading_vehicle(0.0, 0.0, math.pi / 2, 5.0, agents)
    assert gap < math.inf and rel > 0.0


def test_idm_negative_speed_treated_as_zero():
    idm = _IDMController(v_desired=8.33, a_max=2.0)
    a = idm.compute_acceleration(ego_speed=-1.0, gap=math.inf, rel_speed=0.0)
    a_zero = idm.compute_acceleration(0.0, math.inf, 0.0)
    assert isinstance(a, float) and a == pytest.approx(a_zero)


def test_idm_zero_gap_no_exception():
    idm = _IDMController()
    a = idm.compute_acceleration(ego_speed=5.0, gap=0.0, rel_speed=5.0)
    assert isinstance(a, float) and a <= 0.0


# ===========================================================================
# _KinematicBicycleSimulator tests
# ===========================================================================

def test_sim_output_shape():
    sim = _KinematicBicycleSimulator(n_output_steps=8)
    idm = _IDMController()
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=5.0), [], idm)
    assert traj.shape == (8, 2)


def test_sim_moves_forward_east():
    sim = _KinematicBicycleSimulator(n_output_steps=8)
    idm = _IDMController(v_desired=8.33)
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=5.0), [], idm)
    assert traj[-1, 0] > traj[0, 0]
    assert all(traj[i, 0] <= traj[i + 1, 0] for i in range(len(traj) - 1))


def test_sim_moves_forward_north():
    sim = _KinematicBicycleSimulator(n_output_steps=8)
    idm = _IDMController(v_desired=8.33)
    path = _make_straight_path(heading=math.pi / 2)
    ego  = _make_ego_state(heading=math.pi / 2, speed=5.0)
    traj = sim.simulate(path, ego, [], idm)
    assert traj[-1, 1] > traj[0, 1]


def test_sim_stops_for_close_leader():
    sim  = _KinematicBicycleSimulator(n_output_steps=8)
    idm  = _IDMController(v_desired=8.33, a_max=2.0, b=3.0)
    agents = [{"position": np.array([3.0, 0.0]),
               "velocity": np.zeros(2), "heading": 0.0, "type": "VEHICLE"}]
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=5.0), agents, idm)
    total_dist = float(np.linalg.norm(traj[-1] - traj[0]))
    assert total_dist < 6.0


def test_sim_accelerates_from_rest():
    sim = _KinematicBicycleSimulator(n_output_steps=8)
    idm = _IDMController(v_desired=8.33, a_max=2.0)
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=0.0), [], idm)
    assert float(np.linalg.norm(traj[-1] - traj[0])) > 10.0


def test_sim_follows_curved_path():
    sim = _KinematicBicycleSimulator(n_output_steps=8)
    idm = _IDMController(v_desired=5.0)
    n   = 100
    angles = np.linspace(0, math.pi / 2, n)
    R  = 20.0
    path = np.stack([R * np.cos(angles), R * np.sin(angles - math.pi / 2) + R], axis=1)
    path -= path[0]
    ego = _make_ego_state(speed=3.0, heading=0.0)
    traj = sim.simulate(path, ego, [], idm)
    assert traj[-1, 1] > 0.0
    # Also verify trajectory tracked the arc (ends near the reference path)
    final_to_path_dists = np.linalg.norm(path - traj[-1], axis=1)
    assert float(np.min(final_to_path_dists)) < 16.0


def test_sim_very_short_path_no_exception():
    sim  = _KinematicBicycleSimulator(n_output_steps=8)
    idm  = _IDMController()
    traj = sim.simulate(np.array([[0.0, 0.0], [1.0, 0.0]]),
                        _make_ego_state(speed=2.0), [], idm)
    assert traj.shape == (8, 2)


def test_sim_zero_speed_no_crash():
    sim  = _KinematicBicycleSimulator(n_output_steps=8)
    idm  = _IDMController(v_desired=8.33, a_max=2.0)
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=0.0), [], idm)
    assert traj.shape == (8, 2) and np.all(np.isfinite(traj))


def test_sim_waypoints_finite():
    sim  = _KinematicBicycleSimulator(n_output_steps=8)
    idm  = _IDMController()
    traj = sim.simulate(_make_straight_path(), _make_ego_state(speed=5.0), [], idm)
    assert np.all(np.isfinite(traj))


# ===========================================================================
# _CenterlineExtractor tests
# ===========================================================================

def test_centerline_returns_array():
    ext = _CenterlineExtractor(lookahead_m=20.0)
    cl = ext.extract_centerline(_make_scenario(), "ego", np.zeros(3), 0)
    assert cl is not None and cl.ndim == 2 and cl.shape[1] == 2
    assert len(cl) >= 2


def test_centerline_lookahead_respected():
    ext = _CenterlineExtractor(lookahead_m=10.0, resample_spacing=0.5)
    cl = ext.extract_centerline(_make_scenario(), "ego", np.zeros(3), 0)
    total = float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))
    assert 8.0 <= total <= 12.0


def test_centerline_uniform_spacing():
    ext = _CenterlineExtractor(lookahead_m=20.0, resample_spacing=1.0)
    cl = ext.extract_centerline(_make_scenario(), "ego", np.zeros(3), 0)
    seg_lengths = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    assert np.allclose(seg_lengths, seg_lengths[0], atol=0.15)


def test_generate_candidates_count():
    ext = _CenterlineExtractor()
    cl = ext.extract_centerline(_make_scenario(), "ego", np.zeros(3), 0)
    assert len(ext.generate_candidates(cl)) == len(ext.lateral_offsets)


def test_generate_candidates_offset_direction():
    """Left offset (+Y) > centre > right offset (-Y) for eastward path."""
    ext = _CenterlineExtractor(lateral_offsets=(-1.0, 0.0, 1.0))
    cl = ext.extract_centerline(_make_scenario(), "ego", np.zeros(3), 0)
    cands = ext.generate_candidates(cl)
    assert np.all(cands[2][:, 1] > cands[1][:, 1])
    assert np.all(cands[0][:, 1] < cands[1][:, 1])


def test_centerline_mid_scenario_frame():
    sd = _make_scenario(n_frames=200, speed_mps=1.0)
    ext = _CenterlineExtractor(lookahead_m=20.0)
    start_pos = sd["tracks"]["ego"]["state"]["position"][50]
    cl = ext.extract_centerline(sd, "ego", start_pos, 50)
    assert cl is not None
    assert np.linalg.norm(cl[0] - start_pos[:2]) < 2.0


def test_centerline_returns_none_for_single_valid_frame():
    sd = _make_scenario(n_frames=5)
    sd["tracks"]["ego"]["state"]["valid"][:] = False
    sd["tracks"]["ego"]["state"]["valid"][2] = True
    ext = _CenterlineExtractor(lookahead_m=20.0)
    assert ext.extract_centerline(sd, "ego", np.zeros(3), 0) is None


def test_centerline_at_last_frame():
    sd = _make_scenario(n_frames=10)
    ext = _CenterlineExtractor(lookahead_m=20.0)
    cl = ext.extract_centerline(sd, "ego", np.zeros(3), 9)
    assert cl is None


def test_generate_candidates_none_centerline():
    ext = _CenterlineExtractor()
    assert ext.generate_candidates(None) == []


def test_generate_candidates_single_point():
    ext = _CenterlineExtractor()
    assert ext.generate_candidates(np.zeros((1, 2))) == []


def test_centerline_all_invalid_frames():
    sd = _make_scenario(n_frames=20)
    sd["tracks"]["ego"]["state"]["valid"][:] = False
    ext = _CenterlineExtractor(lookahead_m=10.0)
    assert ext.extract_centerline(sd, "ego", np.zeros(3), 0) is None


# ===========================================================================
# _PDMClosedScorer tests — using Shapely lane stubs (no MetaDrive)
# ===========================================================================

from shapely.geometry import box as shapely_box


class _StubLane:
    """Minimal lane stub matching the MetaDrive lane API used by _PDMClosedScorer."""

    def __init__(self, x0, y0, x1, y1):
        self.length = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        self._p0 = np.array([x0, y0], dtype=float)
        self._p1 = np.array([x1, y1], dtype=float)
        self.index = ("road", "lane", "0")

    def local_coordinates(self, pos):
        tang = self._p1 - self._p0
        norm = np.linalg.norm(tang)
        if norm < 1e-9:
            return 0.0, 0.0
        tang /= norm
        d = pos - self._p0
        return float(np.dot(d, tang)), float(np.dot(d, np.array([-tang[1], tang[0]])))

    def heading_at(self, s):
        tang = self._p1 - self._p0
        norm = np.linalg.norm(tang)
        return tang / norm if norm > 1e-9 else np.array([1.0, 0.0])

    def distance(self, pos):
        s, r = self.local_coordinates(pos)
        s_c = max(0.0, min(s, self.length))
        tang = (self._p1 - self._p0) / np.linalg.norm(self._p1 - self._p0)
        closest = self._p0 + s_c * tang
        return float(np.linalg.norm(pos - closest))


def _make_scorer_no_lanes():
    return _PDMClosedScorer(
        all_lanes=[],
        all_lane_bounds=np.empty((0, 4)),
        world_to_sim_offset=np.zeros(2),
    )


def _make_scorer_with_lane(x0=0, y0=-2, x1=30, y1=2):
    poly  = shapely_box(x0, y0, x1, y1)
    lane  = _StubLane(x0, 0, x1, 0)
    lanes = [(lane, poly)]
    bounds = np.array([[x0, y0, x1, y1]])
    return _PDMClosedScorer(
        all_lanes=lanes,
        all_lane_bounds=bounds,
        world_to_sim_offset=np.zeros(2),
    )


def _make_east_traj(n=8, spacing=2.5, y=0.0):
    xs = np.arange(1, n + 1) * spacing
    return np.stack([xs, np.full(n, y)], axis=1)


# --- Degenerate inputs ---

def test_scorer_no_lanes_returns_valid_score():
    s = _make_scorer_no_lanes().score(_make_east_traj(), set(), [])
    assert 0.0 <= s <= 1.0


def test_scorer_empty_trajectory_returns_zero():
    assert _make_scorer_no_lanes().score(np.empty((0, 2)), set(), []) == 0.0


# --- EP (progress) ---

def test_scorer_ep_increases_with_distance():
    s_short = _make_scorer_no_lanes().score(_make_east_traj(spacing=0.5), set(), [])
    s_long  = _make_scorer_no_lanes().score(_make_east_traj(spacing=3.0), set(), [])
    assert s_long > s_short


def test_scorer_ep_capped():
    scorer = _make_scorer_no_lanes()
    traj = _make_east_traj(spacing=20.0)
    assert scorer._score_ep(traj) == pytest.approx(1.0)


# --- COL (collision) ---

def test_scorer_col_collision_drops_score_to_zero():
    scorer = _make_scorer_no_lanes()
    traj   = _make_east_traj(spacing=2.5)
    agents = [{"position": np.array([2.5, 0.0]),
               "velocity": np.zeros(2), "heading": 0.0, "type": "VEHICLE"}]
    assert scorer.score(traj, set(), agents) == 0.0


def test_scorer_col_distant_agent_ok():
    scorer = _make_scorer_no_lanes()
    agents = [{"position": np.array([100.0, 0.0]),
               "velocity": np.zeros(2), "heading": 0.0, "type": "VEHICLE"}]
    assert scorer.score(_make_east_traj(), set(), agents) > 0.0


def test_scorer_col_moving_agent_away_no_collision():
    scorer = _make_scorer_no_lanes()
    agents = [{"position": np.array([10.0, 0.0]),
               "velocity": np.array([20.0, 0.0]),
               "heading": 0.0, "type": "VEHICLE"}]
    assert scorer.score(_make_east_traj(spacing=2.0), set(), agents) > 0.0


# --- DAC ---

def test_scorer_dac_inside_lane_is_one():
    scorer = _make_scorer_with_lane()
    traj_sim = _make_east_traj(spacing=2.5, y=0.0) + scorer.world_to_sim_offset
    assert scorer._score_dac(traj_sim) == 1.0


def test_scorer_dac_outside_lane_less_than_one():
    scorer = _make_scorer_with_lane()
    traj_sim = _make_east_traj(spacing=2.5, y=10.0) + scorer.world_to_sim_offset
    assert scorer._score_dac(traj_sim) == 0.0


def test_scorer_ddc_wrong_direction_is_zero():
    scorer = _make_scorer_with_lane()
    # Westward trajectory on an east-going lane
    traj_sim = np.array([[20.0 - i * 2.5, 0.0] for i in range(8)])
    headings = scorer._compute_headings(traj_sim)
    assert scorer._score_ddc(traj_sim, headings, dac=1.0) == 0.0


# --- HC (history comfort) ---

def test_scorer_hc_smooth_trajectory_is_one():
    scorer   = _make_scorer_no_lanes()
    traj_sim = _make_east_traj(spacing=2.0) + scorer.world_to_sim_offset
    assert scorer._score_hc(traj_sim) == 1.0


def test_scorer_hc_abrupt_stop_is_zero():
    scorer  = _make_scorer_no_lanes()
    traj = np.array([[2.5, 0.], [5.0, 0.], [7.5, 0.], [10.0, 0.],
                     [10.01, 0.], [10.01, 0.], [10.01, 0.], [10.01, 0.]])
    assert scorer._score_hc(traj + scorer.world_to_sim_offset) == 0.0


# --- LK (lane keeping) ---

def test_scorer_lk_on_centerline_is_one():
    scorer   = _make_scorer_with_lane()
    traj_sim = _make_east_traj(spacing=2.0, y=0.0) + scorer.world_to_sim_offset
    headings = scorer._compute_headings(traj_sim)
    assert scorer._score_lk(traj_sim, headings) == 1.0


def test_scorer_lk_far_off_lane_is_zero():
    scorer   = _make_scorer_with_lane()
    traj_sim = _make_east_traj(spacing=2.0, y=3.0) + scorer.world_to_sim_offset
    headings = scorer._compute_headings(traj_sim)
    assert scorer._score_lk(traj_sim, headings) == 0.0


# --- TLC (traffic light) ---

def test_scorer_tlc_no_red_lights():
    assert _make_scorer_no_lanes().score(_make_east_traj(), set(), []) > 0.0


def test_scorer_tlc_red_light_not_in_path():
    scorer    = _make_scorer_with_lane()
    red_lanes = {"99"}
    assert scorer.score(_make_east_traj(), red_lanes, []) > 0.0


# ===========================================================================
# PDMClosedAdapter tests (no MetaDrive required)
# ===========================================================================

def _make_adapter():
    a = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    return a


def _make_two_vehicle_scenario():
    n = 30
    pos_ego = np.zeros((n, 3))
    pos_ego[:, 0] = np.arange(n) * 0.5
    pos_other = np.zeros((n, 3))
    pos_other[:, 0] = 20.0
    pos_ped = np.zeros((n, 3))
    pos_ped[:, 0] = 15.0
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos_ego,
                    "valid":    np.ones(n, dtype=bool),
                    "heading":  np.zeros(n),
                    "velocity": np.zeros((n, 3)),
                },
            },
            "car_1": {
                "type": "VEHICLE",
                "state": {
                    "position": pos_other,
                    "valid":    np.ones(n, dtype=bool),
                    "heading":  np.zeros(n),
                    "velocity": np.zeros((n, 3)),
                },
            },
            "ped_1": {
                "type": "PEDESTRIAN",
                "state": {
                    "position": pos_ped,
                    "valid":    np.ones(n, dtype=bool),
                    "heading":  np.zeros(n),
                    "velocity": np.zeros((n, 3)),
                },
            },
        },
        "dynamic_map_states": {},
        "length": n,
    }


def _make_tl_scenario(state_at_frame, n_frames=10):
    sd = dict(_make_scenario())
    sd["dynamic_map_states"] = {
        "lane_1": {"state": {"object_state": [state_at_frame] * n_frames}}
    }
    return sd


# --- Lifecycle ---

def test_adapter_load_model_no_exception():
    PDMClosedAdapter(checkpoint_path="none").load_model()


def test_adapter_warns_non_none_checkpoint(capsys):
    a = PDMClosedAdapter(checkpoint_path="/fake/path.pth")
    a.load_model()
    out = capsys.readouterr().out
    assert "ignored" in out.lower() or "Checkpoint" in out


def test_adapter_none_checkpoint_no_warning(capsys):
    a = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    assert "ignored" not in capsys.readouterr().out.lower()


def test_adapter_get_camera_configs():
    configs = _make_adapter().get_camera_configs()
    assert "CAM_F0" in configs
    assert "CAM_THIRD_PERSON" in configs


def test_adapter_get_waypoint_dt():
    assert _make_adapter().get_waypoint_dt() == 0.5


def test_adapter_get_trajectory_time_horizon():
    assert _make_adapter().get_trajectory_time_horizon() == 4.0


# --- parse_output coordinate convention ---

def test_adapter_parse_output_shape():
    a = _make_adapter()
    traj_world = np.stack([np.arange(1, 9, dtype=float), np.zeros(8)], axis=1)
    out = a.parse_output({"trajectory_world": traj_world}, _make_ego_state())
    assert out["trajectory"].shape == (8, 2)


def test_adapter_parse_output_heading_east():
    """Heading 0 (east): +x points ahead → forward > 0, lateral ≈ 0."""
    a = _make_adapter()
    traj_world = np.stack([np.arange(1, 9, dtype=float), np.zeros(8)], axis=1)
    out = a.parse_output({"trajectory_world": traj_world}, _make_ego_state(heading=0.0))
    assert np.all(out["trajectory"][:, 1] > 0)
    assert np.allclose(out["trajectory"][:, 0], 0, atol=1e-6)


def test_adapter_parse_output_heading_north():
    """Heading π/2 (north): 1 m north → forward ≈ 1, lateral ≈ 0."""
    a = _make_adapter()
    traj_world = np.array([[0.0, 1.0]])
    out = a.parse_output({"trajectory_world": traj_world},
                         _make_ego_state(heading=math.pi / 2))
    assert abs(out["trajectory"][0, 1] - 1.0) < 1e-6
    assert abs(out["trajectory"][0, 0]) < 1e-6


def test_adapter_parse_output_finite():
    a = _make_adapter()
    out = a.parse_output({"trajectory_world": np.random.randn(8, 2) * 5}, _make_ego_state())
    assert np.all(np.isfinite(out["trajectory"]))


# --- _extract_agents ---

def test_adapter_extract_agents_excludes_ego():
    a = _make_adapter()
    a._sdc_id = "ego"
    agents = a._extract_agents(_make_two_vehicle_scenario(), 0)
    assert any(ag["position"][0] == 20.0 for ag in agents)
    assert not any(ag["position"][0] == 0.0 for ag in agents)


def test_adapter_extract_agents_excludes_pedestrians():
    a = _make_adapter()
    a._sdc_id = "ego"
    agents = a._extract_agents(_make_two_vehicle_scenario(), 0)
    assert all(ag["type"] != "PEDESTRIAN" for ag in agents)


def test_adapter_extract_agents_frame_out_of_range():
    a = _make_adapter()
    a._sdc_id = "ego"
    assert a._extract_agents(_make_scenario(n_frames=5), 999) == []


def test_adapter_extract_agents_too_far():
    a = _make_adapter()
    a._sdc_id = "ego"
    sd = _make_two_vehicle_scenario()
    sd["tracks"]["car_1"]["state"]["position"][:, 0] = 200.0
    assert len(a._extract_agents(sd, 0)) == 0


def test_adapter_extract_agents_no_velocity_field():
    a = _make_adapter()
    a._sdc_id = "ego"
    sd = _make_two_vehicle_scenario()
    del sd["tracks"]["car_1"]["state"]["velocity"]
    agents = a._extract_agents(sd, 5)
    assert all(np.all(np.isfinite(ag["velocity"])) for ag in agents)


# --- _extract_tl_states ---

def test_adapter_extract_tl_states_stop():
    a = _make_adapter()
    tl = a._extract_tl_states(_make_tl_scenario("LANE_STATE_STOP"), 0)
    assert "lane_1" in tl


def test_adapter_extract_tl_states_go():
    a = _make_adapter()
    tl = a._extract_tl_states(_make_tl_scenario("LANE_STATE_GO"), 0)
    assert "lane_1" not in tl


def test_adapter_extract_tl_states_empty():
    a = _make_adapter()
    assert a._extract_tl_states(_make_scenario(), 0) == set()


def test_adapter_extract_tl_states_out_of_range():
    a = _make_adapter()
    tl = a._extract_tl_states(_make_tl_scenario("LANE_STATE_STOP", n_frames=3), 10)
    assert "lane_1" not in tl


# --- prepare_input + run_inference ---

def test_adapter_prepare_input_keys():
    a = _make_adapter()
    a._sdc_id = "ego"
    inp = a.prepare_input({}, _make_ego_state(), _make_scenario(n_frames=100), 0)
    for key in ("ego_state", "agents", "tl_states", "centerline", "frame_id"):
        assert key in inp


def test_adapter_run_inference_output_shape():
    a = _make_adapter()
    a._sdc_id = "ego"
    inp = a.prepare_input({}, _make_ego_state(), _make_scenario(n_frames=100), 0)
    out = a.run_inference(inp)
    assert np.array(out["trajectory_world"]).shape == (8, 2)


def test_adapter_run_inference_fallback_no_centerline():
    a = _make_adapter()
    out = a.run_inference({
        "ego_state": _make_ego_state(), "agents": [],
        "tl_states": set(), "centerline": None, "frame_id": 0,
    })
    assert np.array(out["trajectory_world"]).shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory_world"]))


# ===========================================================================
# End-to-end pipeline tests (no MetaDrive)
# ===========================================================================

def test_e2e_full_pipeline():
    a   = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    sd  = _make_scenario(n_frames=200, speed_mps=1.0)
    ego = _make_ego_state(speed=5.0)
    out = a.parse_output(a.run_inference(a.prepare_input({}, ego, sd, 0)), ego)
    assert out["trajectory"].shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory"]))


def test_e2e_fallback_all_invalid_tracks():
    a  = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    sd = _make_scenario(n_frames=50)
    sd["tracks"]["ego"]["state"]["valid"][:] = False
    ego = _make_ego_state(speed=3.0)
    out = a.parse_output(a.run_inference(a.prepare_input({}, ego, sd, 0)), ego)
    assert out["trajectory"].shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory"]))


def test_e2e_with_nearby_vehicle():
    a  = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    sd = _make_scenario(n_frames=100)
    n  = 100
    other_pos = np.zeros((n, 3))
    other_pos[:, 0] = 20.0
    sd["tracks"]["car_1"] = {
        "type": "VEHICLE",
        "state": {
            "position": other_pos,
            "valid":    np.ones(n, dtype=bool),
            "heading":  np.zeros(n),
            "velocity": np.zeros((n, 3)),
        },
    }
    ego = _make_ego_state(speed=5.0)
    out = a.parse_output(a.run_inference(a.prepare_input({}, ego, sd, 0)), ego)
    assert out["trajectory"].shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory"]))


def test_e2e_red_traffic_light():
    a  = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    sd = _make_scenario(n_frames=100)
    sd["dynamic_map_states"]["tl_0"] = {
        "state": {"object_state": ["LANE_STATE_STOP"] * 100}
    }
    ego = _make_ego_state(speed=5.0)
    out = a.parse_output(a.run_inference(a.prepare_input({}, ego, sd, 0)), ego)
    assert out["trajectory"].shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory"]))


def test_e2e_mid_scenario_frame():
    a  = PDMClosedAdapter(checkpoint_path="none")
    a.load_model()
    sd = _make_scenario(n_frames=200, speed_mps=1.0)
    pos_50 = sd["tracks"]["ego"]["state"]["position"][50]
    ego = _make_ego_state(x=float(pos_50[0]), y=float(pos_50[1]), speed=3.0)
    out = a.parse_output(a.run_inference(a.prepare_input({}, ego, sd, 50)), ego)
    assert out["trajectory"].shape == (8, 2)
    assert np.all(np.isfinite(out["trajectory"]))

"""
PDM-Closed standalone demo — no MetaDrive or real scenarios required.

Runs three synthetic scenarios through the full planning pipeline and
produces a matplotlib figure showing candidate trajectories and scores.

Usage:
    conda run -n mdsn python tools/demo_pdm_closed.py
    conda run -n mdsn python tools/demo_pdm_closed.py --save demo_pdm_closed.png
"""

import sys, os, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from bridgesim.evaluation.models.pdm_closed_adapter import (
    PDMClosedAdapter,
    _CenterlineExtractor,
    _IDMController,
    _KinematicBicycleSimulator,
    _PDMClosedScorer,
    VEHICLE_LENGTH,
    VEHICLE_WIDTH,
)

def make_scenario(positions, heading=0.0):
    n = len(positions)
    pos = np.zeros((n, 3))
    pos[:, :2] = positions
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos,
                    "valid": np.ones(n, dtype=bool),
                    "heading": np.full(n, heading),
                    "velocity": np.zeros((n, 3)),
                },
            }
        },
        "dynamic_map_states": {},
        "length": n,
    }


def make_ego_state(x=0.0, y=0.0, heading=0.0, speed=5.0):
    return {
        "position": np.array([x, y, 0.0]),
        "heading": heading,
        "speed": speed,
        "velocity": np.array([speed * math.cos(heading), speed * math.sin(heading), 0.0]),
        "acceleration": np.zeros(3),
        "angular_velocity": np.zeros(3),
        "waypoint": np.array([x + 20.0, y]),
        "command": 2,
    }


def run_pipeline(adapter, scenario_data, ego_state, agents=None):
    """Run prepare → infer and return candidates + scores + best trajectory."""
    if agents is None:
        agents = []

    # Build input manually so we can also expose candidates/scores
    sdc_id = scenario_data["metadata"]["sdc_id"]
    adapter._sdc_id = sdc_id

    centerline = adapter._extractor.extract_centerline(
        scenario_data, sdc_id, ego_state["position"], 0
    )
    candidates = adapter._extractor.generate_candidates(centerline)

    tl_states = set()
    scorer = _PDMClosedScorer(
        all_lanes=[],
        all_lane_bounds=np.empty((0, 4)),
        world_to_sim_offset=np.zeros(2),
    )

    simulated, scores = [], []
    for path in candidates:
        traj = adapter._simulator.simulate(path, ego_state, agents, adapter._idm)
        simulated.append(traj)
        scores.append(scorer.score(traj, tl_states, agents))

    best_idx = int(np.argmax(scores))
    return candidates, simulated, scores, best_idx, centerline


def draw_vehicle(ax, x, y, heading, color="steelblue", alpha=1.0, label=None):
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    hl, hw = VEHICLE_LENGTH / 2.0, VEHICLE_WIDTH / 2.0
    corners = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    world_corners = (rot @ corners.T).T + np.array([x, y])
    poly = plt.Polygon(world_corners, closed=True, facecolor=color,
                       edgecolor="black", alpha=alpha, linewidth=0.8, label=label)
    ax.add_patch(poly)
    # heading arrow
    ax.annotate("", xy=(x + cos_h * hl * 0.9, y + sin_h * hl * 0.9),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))


def scenario_straight(adapter):
    """Straight empty road — all candidates score similarly, centre wins."""
    n = 200
    xs = np.linspace(0, 50, n)
    positions = np.stack([xs, np.zeros(n)], axis=1)
    sd = make_scenario(positions)
    ego = make_ego_state(speed=6.0)
    return run_pipeline(adapter, sd, ego), "Straight Road (No Obstacles)", []


def scenario_blocked(adapter):
    """
    Agent sits 1.2 m right of centre at x=12.
    As the ego steers left (negative offset), the agent exits the IDM's
    lateral corridor — IDM stops braking — ego reaches the agent → COL=0.
    As the ego steers right (positive offset), the agent stays in the
    corridor — IDM brakes throughout — ego stops safely → higher score.
    Best candidate is +1.5 m (maximises safe progress, stays right of agent).
    """
    n = 200
    xs = np.linspace(0, 50, n)
    positions = np.stack([xs, np.zeros(n)], axis=1)
    sd = make_scenario(positions)
    ego = make_ego_state(speed=5.0)
    agents = [{
        "position": np.array([12.0, 1.2]),
        "velocity": np.zeros(2),
        "heading": 0.0,
        "type": "VEHICLE",
    }]
    return run_pipeline(adapter, sd, ego, agents), "Right-Side Agent: IDM Corridor Interaction", agents


def scenario_curve(adapter):
    """Quarter-circle arc — planner tracks the curve."""
    n = 200
    angles = np.linspace(0, math.pi / 2, n)
    R = 30.0
    xs = R * np.cos(angles)
    ys = R * np.sin(angles) - R          # shift so start = (R, -R) → (R,0) actually at origin
    xs -= xs[0]
    ys -= ys[0]
    positions = np.stack([xs, ys], axis=1)
    sd = make_scenario(positions)
    ego = make_ego_state(speed=5.0)
    return run_pipeline(adapter, sd, ego), "Curved Road (Quarter Arc)", []

CANDIDATE_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
                    "#1abc9c", "#3498db", "#9b59b6"]
OFFSET_LABELS = ["-1.5 m", "-1.0 m", "-0.5 m", "centre",
                 "+0.5 m", "+1.0 m", "+1.5 m"]


def plot_scenario(ax, result, title, agents):
    candidates, simulated, scores, best_idx, centerline = result

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_facecolor("#f8f9fa")

    # Centerline (GT reference)
    if centerline is not None:
        ax.plot(centerline[:, 0], centerline[:, 1],
                color="gray", lw=1.0, ls="--", alpha=0.6, label="GT centerline")

    # All candidate paths (faint)
    for i, path in enumerate(candidates):
        ax.plot(path[:, 0], path[:, 1],
                color=CANDIDATE_COLORS[i], lw=0.8, alpha=0.25, ls=":")

    # All simulated trajectories with scores
    for i, (traj, score) in enumerate(zip(simulated, scores)):
        lw = 3.5 if i == best_idx else 1.2
        alpha = 1.0 if i == best_idx else 0.45
        zorder = 5 if i == best_idx else 2
        ax.plot(traj[:, 0], traj[:, 1],
                color=CANDIDATE_COLORS[i], lw=lw, alpha=alpha, zorder=zorder,
                label=f"{OFFSET_LABELS[i]}  score={score:.3f}{'  ★' if i == best_idx else ''}")
        # Dots at waypoints
        ax.scatter(traj[:, 0], traj[:, 1],
                   color=CANDIDATE_COLORS[i], s=(28 if i == best_idx else 8),
                   zorder=zorder + 1, alpha=alpha)

    # Ego vehicle
    draw_vehicle(ax, 0, 0, 0.0, color="steelblue", label="Ego (start)")

    # Obstacle vehicles
    for ag in agents:
        draw_vehicle(ax, ag["position"][0], ag["position"][1],
                     ag["heading"], color="#c0392b", alpha=0.85, label="Obstacle")

    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)


def print_scores(title, scores, best_idx, offsets):
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")
    for i, (s, label) in enumerate(zip(scores, offsets)):
        star = " ★ BEST" if i == best_idx else ""
        print(f"  {label:>8}  score = {s:.4f}{star}")
    print()

def main():
    parser = argparse.ArgumentParser(description="PDM-Closed visual demo")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="Save figure to this path instead of showing interactively")
    args = parser.parse_args()

    print("Loading PDM-Closed adapter...")
    adapter = PDMClosedAdapter(checkpoint_path="none")
    adapter.load_model()

    print("Running three synthetic scenarios...\n")

    results = [
        scenario_straight(adapter),
        scenario_blocked(adapter),
        scenario_curve(adapter),
    ]

    for (result, title, agents) in results:
        _, _, scores, best_idx, _ = result
        print_scores(title, scores, best_idx, OFFSET_LABELS)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("PDM-Closed: Candidate Trajectories & Scores\n"
                 "(dashed = GT centerline reference, ★ = selected trajectory)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, (result, title, agents) in zip(axes, results):
        plot_scenario(ax, result, title, agents)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

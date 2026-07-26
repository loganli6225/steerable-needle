"""Render vanilla-RRT runs: SDF background, obstacle rim, explored tree,
final path (if found), and start/goal markers.

Two scenarios:
    1. "open (single circle)" -- the tests/test_rrt.py scenario. Reaches the
       goal in well under a hundred iterations, so the tree stays sparse.
    2. "narrow doorway (bug trap)" -- a wall spanning the full width of the
       world with a small gap, so the only route from start to goal is
       through the gap. There is no easy detour around the sides, so this
       stresses sampling (most random samples/edges never thread the gap)
       and collision-checking near two close boundary segments at once --
       a harder, more informative validation than the open scenario.

Both verify by eye what the five green tests don't: that the tree's
exploration is sensible (branches stop at obstacle boundaries, nothing
leaks through a gap it shouldn't) and that the path is not an absurd
detour. This is also the "before" half of a planned before/after comparison
figure -- vanilla RRT teleports the needle's heading toward each sample, so
paths show sharp corners a real needle cannot follow; a later
curvature-constrained planner's smooth arcs go in the matching panel. Hence
the fixed colours/sizing here, chosen to match that counterpart.

Usage:
    python scripts/demo_rrt.py
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt import RRT, RRTConfig
from needlesim.utils.plotting import plot_sdf_background


@dataclass
class Scenario:
    """Everything needed to build one environment/planner run. Config-driven,
    seeded -- no magic numbers outside this block (see CLAUDE.md)."""

    name: str
    width: float
    height: float
    resolution: float
    circles: list[tuple[float, float, float]]        # (cx, cy, r)
    rects: list[tuple[float, float, float, float]]    # (x0, y0, x1, y1)
    kappa: float
    start: State
    goal: State
    rrt_config: RRTConfig
    output_path: Path


SCENARIOS = [
    Scenario(
        name="open (single circle)",
        width=100.0,
        height=100.0,
        resolution=0.5,
        circles=[(50.0, 50.0, 15.0)],
        rects=[],
        kappa=1.0 / 20.0,
        start=State(20.0, 20.0, math.pi / 2),
        goal=State(20.0, 80.0, math.pi / 2),
        rrt_config=RRTConfig(
            max_iterations=5000, goal_tolerance=3.0, step_dt=0.05, edge_velocity=5.0, seed=1
        ),
        output_path=Path("experiments/results/rrt_kinodynamic_tree.png"),
    ),
    Scenario(
        name="narrow doorway (bug trap)",
        width=100.0,
        height=100.0,
        resolution=0.5,
        circles=[],
        # Full-width wall at y in [45, 53] with an 8mm gap in x (46, 54): the
        # only way from start to goal, no detour around the sides.
        rects=[(0.0, 45.0, 46.0, 53.0), (54.0, 45.0, 100.0, 53.0)],
        kappa=1.0 / 20.0,
        start=State(50.0, 20.0, math.pi / 2),
        goal=State(50.0, 90.0, math.pi / 2),
        rrt_config=RRTConfig(
            max_iterations=20000, goal_tolerance=3.0, step_dt=0.05, edge_velocity=5.0, seed=1
        ),
        output_path=Path("experiments/results/rrt_kinodynamic_tree_hard.png"),
    ),
]


def build_environment(scenario: Scenario) -> GridEnvironment:
    env = GridEnvironment(width=scenario.width, height=scenario.height, resolution=scenario.resolution)
    for cx, cy, r in scenario.circles:
        env.add_circle(cx, cy, r)
    for x0, y0, x1, y1 in scenario.rects:
        env.add_rect(x0, y0, x1, y1)
    env.bake()
    return env


def render(env: GridEnvironment, result, scenario: Scenario) -> None:
    """Z-order: SDF heatmap -> obstacle rim -> tree edges -> path -> markers."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax, im = plot_sdf_background(env, ax=ax)

    nodes = result.nodes
    segments = [
        ((nodes[node.parent].state.x, nodes[node.parent].state.y), (node.state.x, node.state.y))
        for node in nodes[1:]
    ]
    tree_edges = LineCollection(segments, colors="0.3", linewidths=0.5, alpha=0.3, zorder=2)
    ax.add_collection(tree_edges)

    if result.success:
        path_xs = [s.x for s in result.path]
        path_ys = [s.y for s in result.path]
        ax.plot(path_xs, path_ys, color="deeppink", linewidth=2.5, zorder=4, label="path")

    ax.plot(
        scenario.start.x, scenario.start.y,
        marker="o", color="limegreen", markersize=10, zorder=5, label="start",
    )
    ax.plot(
        scenario.goal.x, scenario.goal.y,
        marker="*", color="gold", markersize=16, zorder=5, label="goal",
    )
    goal_circle = Circle(
        (scenario.goal.x, scenario.goal.y),
        scenario.rrt_config.goal_tolerance,
        fill=False,
        edgecolor="gold",
        linestyle="--",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(goal_circle)

    ax.legend(loc="upper right", fontsize=8)

    status = "success" if result.success else "FAILED (no path found -- tree shown for diagnosis)"
    ax.set_title(
        f"Kinodynamic RRT -- {scenario.name}\n"
        f"success={result.success}, nodes={len(result.nodes)}, "
        f"iterations={result.n_iterations} ({status})"
    )

    fig.tight_layout()
    scenario.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scenario.output_path, dpi=150)
    print(f"wrote {scenario.output_path}")


def main() -> None:
    for scenario in SCENARIOS:
        env = build_environment(scenario)
        params = NeedleParams(kappa=scenario.kappa)
        planner = RRT(env, params, scenario.rrt_config)
        result = planner.plan(scenario.start, scenario.goal)
        print(
            f"{scenario.name}: success={result.success}, "
            f"nodes={len(result.nodes)}, iterations={result.n_iterations}"
        )
        render(env, result, scenario)


if __name__ == "__main__":
    main()

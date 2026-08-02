"""Throwaway RRT* eyeball plot -- NOT a deliverable. Confirms the planner
produces a sensible path (smooth Dubins arcs, threads the obstacle) before
trusting it. Delete after looking, or keep in scripts/ if useful.

Run:  python eyeball_rrt_star.py
"""

import math

import matplotlib.pyplot as plt

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State, rollout_variable
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig

# Same scale that works for RRT* (R=5 in a 100mm world).
env = GridEnvironment(100, 100, 0.5)
env.add_circle(50, 50, 15)
env.bake()
params = NeedleParams(kappa=1 / 5)

cfg = RRTStarConfig(max_iterations=4000, gamma=40.0, seed=1)
planner = RRTStar(env, params, cfg)
start = State(20, 20, math.pi / 2)
goal = State(20, 80, math.pi / 2)
result = planner.plan(start, goal)
print("success:", result.success, "| nodes:", len(result.nodes),
      "| cost:", round(result.best_cost, 1) if result.success else "n/a")

fig, ax = plt.subplots(figsize=(7, 7))

# SDF background so the obstacle is visible.
im = ax.imshow(env.sdf, origin="lower", extent=[0, 100, 0, 100],
               cmap="RdBu", vmin=-20, vmax=20, alpha=0.6)
ax.contour(env.sdf, levels=[0], colors="black",
           extent=[0, 100, 0, 100], linewidths=1)

# Tree edges: draw each node's Dubins edge from its parent as actual arcs,
# not straight chords, so the curvature is visible.
for node in result.nodes:
    if node.parent is None or node.edge_from_parent is None:
        continue
    parent_state = result.nodes[node.parent].state
    trace = rollout_variable(parent_state, node.edge_from_parent.controls, params)
    ax.plot([s.x for s in trace], [s.y for s in trace],
            color="gray", alpha=0.25, linewidth=0.5)

# Final path, thick.
if result.success:
    ax.plot([s.x for s in result.path], [s.y for s in result.path],
            color="deeppink", linewidth=2.5, label="RRT* path")

ax.plot(start.x, start.y, "go", markersize=10, label="start")
ax.plot(goal.x, goal.y, "b*", markersize=15, label="goal")
circle = plt.Circle((goal.x, goal.y), cfg.goal_tolerance, color="blue",
                    fill=False, linestyle="--", alpha=0.5)
ax.add_patch(circle)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_title(f"RRT* eyeball: success={result.success}, "
             f"{len(result.nodes)} nodes, cost={round(result.best_cost, 1) if result.success else 'n/a'}")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("rrt_star_eyeball.png", dpi=130)
print("saved rrt_star_eyeball.png")

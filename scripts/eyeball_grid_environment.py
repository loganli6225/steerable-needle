"""Not a test -- this checks three things for the Task 2 grid environment by eye that no assertion fully covers:

1. Concentric bands around the circle, sign flipping at the rim.
   (SDF magnitude/sign is right, not just "close enough" at a handful of
   sample points.)
2. The circle sits where you put it (CX, CY), not transposed.
   (A row/col swap in world_to_grid or add_circle can still pass every test
   above because CX == CY == 50 in the test fixture -- a transpose is
   invisible when the obstacle is on the diagonal. This picture would still
   catch it if you change CX != CY.)
3. The two arcs from test_arc_free_vs_collision visibly straddle the rim:
   b=-1 plows into the circle, b=+1 peels away and clears it.

Usage:
    python scripts/eyeball_grid_environment.py
"""

import math

import matplotlib.pyplot as plt

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout

CX, CY, R = 70.0, 30.0, 15.0

env = GridEnvironment(width=100.0, height=100.0, resolution=0.25)
env.add_circle(CX, CY, R)
env.bake()

params = NeedleParams(kappa=1.0 / 20.0)
start = State(CX - 6.0, CY - 25.0, math.pi / 2)
dt, n_samples = 0.02, 300

hit_trace = rollout(start, [Control(v=5.0, b=-1)] * n_samples, dt, params)
free_trace = rollout(start, [Control(v=5.0, b=+1)] * n_samples, dt, params)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(
    env.sdf,
    origin="lower",
    extent=(0, env.width, 0, env.height),
    cmap="RdBu",
    vmin=-20,
    vmax=20,
)
fig.colorbar(im, ax=ax, label="signed distance [mm]")
ax.contour(
    env.sdf,
    levels=[0.0],
    colors="k",
    linewidths=1.5,
    extent=(0, env.width, 0, env.height),
    origin="lower",
)

ax.plot([s.x for s in hit_trace], [s.y for s in hit_trace], "r-", label="b=-1 (expect collision)")
ax.plot([s.x for s in free_trace], [s.y for s in free_trace], "g-", label="b=+1 (expect clear)")
ax.plot(start.x, start.y, "ko", label="start")
ax.plot(CX, CY, "k+", markersize=12, label="(CX, CY)")

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("SDF sanity check: bands, rim sign flip, arc straddle")

fig.tight_layout()
fig.savefig("grid_environment_check.png", dpi=150)
print("wrote grid_environment_check.png")
plt.show()

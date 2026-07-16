"""Generate the three Task 1 acceptance-test figures.

Runs once you've implemented `step`. These plots are also the first figures
in your writeup, so this script doubles as figure generation.

Usage:
    python scripts/demo_needle_model.py
"""

import matplotlib.pyplot as plt

from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout
from needlesim.utils.plotting import plot_trace

params = NeedleParams(kappa=1.0 / 50.0)
s0 = State(0.0, 0.0, 0.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Constant control -> circle
controls = [Control(v=5.0, b=+1) for _ in range(300)]
plot_trace(rollout(s0, controls, 0.1, params), ax=axes[0])
axes[0].set_title("Constant b: circle of radius 1/kappa")

# 2. Flip -> S-curve
controls = [Control(v=5.0, b=+1) for _ in range(200)]
controls += [Control(v=5.0, b=-1) for _ in range(200)]
plot_trace(rollout(s0, controls, 0.1, params), ax=axes[1])
axes[1].set_title("Flip at midpoint: symmetric S-curve")

# 3. Duty cycle -> reduced effective curvature
controls = [Control(v=5.0, b=(+1 if i % 2 == 0 else -1)) for i in range(600)]
plot_trace(rollout(s0, controls, 0.05, params), ax=axes[2])
axes[2].set_title("50% duty cycle: ~kappa/2")

fig.tight_layout()
fig.savefig("experiments/results/task1_acceptance.png", dpi=150)
print("Saved experiments/results/task1_acceptance.png")

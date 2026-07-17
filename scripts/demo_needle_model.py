"""Generate the Task 1 acceptance-test figures.

Runs once you've implemented `step`. These plots are also the first figures
in your writeup, so this script doubles as figure generation.

Usage:
    python scripts/demo_needle_model.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from needlesim.models.unicycle_needle import Control, NeedleParams, State, rollout
from needlesim.utils.plotting import plot_trace

params = NeedleParams(kappa=1.0 / 50.0)
s0 = State(0.0, 0.0, 0.0)


def duty_cycle_controls(n, p, v=5.0):
    """n controls with b=+1 on a p-fraction of steps, spread evenly."""
    controls = []
    acc = 0.0
    for _ in range(n):
        acc += p
        if acc >= 1.0:
            controls.append(Control(v=v, b=+1))
            acc -= 1.0
        else:
            controls.append(Control(v=v, b=-1))
    return controls


def measure_kappa_eff(p, n=400, dt=0.05):
    trace = rollout(s0, duty_cycle_controls(n, p), dt, params)
    arc_len = sum(
        math.hypot(trace[i + 1].x - trace[i].x, trace[i + 1].y - trace[i].y)
        for i in range(len(trace) - 1)
    )
    dtheta = trace[-1].theta - trace[0].theta
    return dtheta / arc_len


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

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
# 50% duty cycle cancels to a straight line (kappa_eff=0); 75% b=+1 gives
# kappa_eff = kappa*(2*0.75 - 1) = kappa/2.
controls = [Control(v=5.0, b=(-1 if i % 4 == 3 else +1)) for i in range(600)]
plot_trace(rollout(s0, controls, 0.05, params), ax=axes[2])
axes[2].set_title("75/25 duty cycle: kappa_eff = kappa/2")

# 4. kappa_eff vs p: measured emergent curvature against the analytic line
ps = np.linspace(0.0, 1.0, 11)
measured = [measure_kappa_eff(p) for p in ps]
analytic = params.kappa * (2 * ps - 1)
axes[3].plot(ps, analytic, "-", color="gray", label="kappa*(2p-1)")
axes[3].plot(ps, measured, "o", color="C0", label="measured")
axes[3].set_xlabel("p (fraction of steps with b=+1)")
axes[3].set_ylabel("kappa_eff [1/mm]")
axes[3].set_title("Emergent curvature vs duty cycle")
axes[3].legend()

fig.tight_layout()
fig.savefig("experiments/results/task1_acceptance.png", dpi=150)
print("Saved experiments/results/task1_acceptance.png")

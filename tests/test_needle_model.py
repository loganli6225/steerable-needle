"""Acceptance tests for the Task 1 needle model.

These FAIL until you implement `step` in unicycle_needle.py. That is intended:
they are your definition of "done" for Task 1. Run with:  pytest -q

Test 3 (duty cycle) is the important one: the ~kappa/2 effective curvature
should EMERGE from flipping b, not be hardcoded anywhere.
"""

import math

import numpy as np
import pytest

from needlesim.models.unicycle_needle import (
    Control,
    NeedleParams,
    State,
    rollout,
)


def _fit_circle_radius(xs, ys):
    """Algebraic circle fit; returns fitted radius. Helper for test 1."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    A = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    bvec = xs**2 + ys**2
    cx, cy, c = np.linalg.lstsq(A, bvec, rcond=None)[0]
    return math.sqrt(c + cx**2 + cy**2)


def test_constant_control_traces_circle():
    params = NeedleParams(kappa=1.0 / 50.0)
    s0 = State(0.0, 0.0, 0.0)
    dt = 0.1
    controls = [Control(v=5.0, b=+1) for _ in range(300)]
    trace = rollout(s0, controls, dt, params)
    xs = [s.x for s in trace]
    ys = [s.y for s in trace]
    r_fit = _fit_circle_radius(xs, ys)
    r_true = 1.0 / params.kappa
    assert abs(r_fit - r_true) / r_true < 0.02, (
        f"fitted radius {r_fit:.2f} vs expected {r_true:.2f}"
    )


def test_flip_makes_symmetric_s_curve():
    params = NeedleParams(kappa=1.0 / 50.0)
    s0 = State(0.0, 0.0, 0.0)
    dt = 0.1
    n = 200
    controls = [Control(v=5.0, b=+1) for _ in range(n)]
    controls += [Control(v=5.0, b=-1) for _ in range(n)]
    trace = rollout(s0, controls, dt, params)
    # Heading should return near its start after a symmetric flip.
    assert abs(trace[-1].theta - trace[0].theta) < 0.05


def test_duty_cycle_halves_curvature():
    params = NeedleParams(kappa=1.0 / 50.0)
    s0 = State(0.0, 0.0, 0.0)
    dt = 0.05
    controls = [Control(v=5.0, b=(+1 if i % 2 == 0 else -1)) for i in range(600)]
    trace = rollout(s0, controls, dt, params)
    # Net heading change / arc length gives effective curvature.
    arc_len = sum(
        math.hypot(trace[i + 1].x - trace[i].x, trace[i + 1].y - trace[i].y)
        for i in range(len(trace) - 1)
    )
    dtheta = abs(trace[-1].theta - trace[0].theta)
    kappa_eff = dtheta / arc_len if arc_len > 0 else float("inf")
    assert kappa_eff < 0.5 * params.kappa, (
        f"effective curvature {kappa_eff:.4f} should be well below kappa "
        f"{params.kappa:.4f} for a 50% duty cycle"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

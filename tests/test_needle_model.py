"""Acceptance tests for the Task 1 needle model.

These FAIL until you implement `step` in unicycle_needle.py. That is intended:
they are your definition of "done" for Task 1. Run with:  pytest -q

Test 3 (duty cycle) is the important one: kappa_eff = kappa*(2p - 1) should
EMERGE from flipping b, not be hardcoded anywhere.
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


def _duty_cycle_controls(n, p, v=5.0):
    """n controls with b=+1 on a p-fraction of steps, spread evenly rather
    than front- or back-loaded, so the test can't pass by tracing the wrong
    shape with the right net curvature."""
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


def test_constant_control_traces_circle():
    # Smoke test only: 2% tolerance is loose enough that forward Euler and RK4
    # both land comfortably inside it here. For a tolerance that actually
    # discriminates the two, see test_rk4_matches_analytic_quarter_circle.
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
    """The post-flip half of the trace must be a point reflection of the
    pre-flip half about the midpoint: for j steps either side of the flip,
    trace[n+j] should equal 2*trace[n] - trace[n-j] in (x, y).

    (A theta-only check isn't sufficient here: theta_dot doesn't depend on x,
    y, or theta itself, so heading integrates exactly under any integrator,
    including one that never updates x or y at all.)
    """
    params = NeedleParams(kappa=1.0 / 50.0)
    s0 = State(0.0, 0.0, 0.0)
    dt = 0.1
    n = 200
    controls = [Control(v=5.0, b=+1) for _ in range(n)]
    controls += [Control(v=5.0, b=-1) for _ in range(n)]
    trace = rollout(s0, controls, dt, params)
    mid = trace[n]
    for j in range(1, n + 1):
        before, after = trace[n - j], trace[n + j]
        reflected_err = math.hypot(
            after.x - (2 * mid.x - before.x), after.y - (2 * mid.y - before.y)
        )
        assert reflected_err < 1e-6, f"j={j}: point-reflection error {reflected_err:.3e}"


@pytest.mark.parametrize("p", [0.5, 0.625, 0.75, 0.875, 1.0])
def test_duty_cycle_scales_curvature(p):
    """kappa_eff = kappa*(2p - 1) must emerge from the fraction of b=+1 steps;
    it is not computed anywhere in the model itself. p=0.5 -> straight line
    (kappa_eff=0), p=1.0 -> full curvature."""
    params = NeedleParams(kappa=1.0 / 50.0)
    s0 = State(0.0, 0.0, 0.0)
    dt = 0.05
    n = 400
    controls = _duty_cycle_controls(n, p)
    trace = rollout(s0, controls, dt, params)
    arc_len = sum(
        math.hypot(trace[i + 1].x - trace[i].x, trace[i + 1].y - trace[i].y)
        for i in range(len(trace) - 1)
    )
    dtheta = abs(trace[-1].theta - trace[0].theta)
    kappa_eff = dtheta / arc_len if arc_len > 0 else float("inf")
    predicted = params.kappa * (2 * p - 1)
    if predicted < 1e-9:
        assert kappa_eff < 1e-3, (
            f"expected ~straight line at p={p}, got kappa_eff={kappa_eff:.5f}"
        )
    else:
        rel_err = abs(kappa_eff - predicted) / predicted
        assert rel_err < 0.02, (
            f"p={p}: kappa_eff={kappa_eff:.5f} vs predicted {predicted:.5f}"
        )


def test_rk4_matches_analytic_quarter_circle():
    """Convergence check the circle-fit test is too loose to catch: trace a
    quarter circle at increasing step counts and compare the endpoint against
    the closed-form solution. RK4 is ~4th order (error ratio ~16x per
    doubling); forward Euler is only 1st order (~2x) and misses the absolute
    tolerance below by five orders of magnitude at n=200."""
    params = NeedleParams(kappa=0.02)
    v = 5.0
    theta_total = math.pi / 2
    radius = 1.0 / params.kappa
    x_analytic = radius * math.sin(theta_total)
    y_analytic = radius * (1 - math.cos(theta_total))

    def endpoint_error(n):
        dt = theta_total / (v * params.kappa * n)
        controls = [Control(v=v, b=+1) for _ in range(n)]
        trace = rollout(State(0.0, 0.0, 0.0), controls, dt, params)
        end = trace[-1]
        return math.hypot(end.x - x_analytic, end.y - y_analytic)

    err_100 = endpoint_error(100)
    err_200 = endpoint_error(200)
    assert err_200 < 1e-6, f"endpoint error {err_200:.3e} too large at n=200"
    assert err_100 / err_200 > 8, (
        f"convergence ratio {err_100 / err_200:.2f} looks like Euler (~2x), not RK4 (~16x)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

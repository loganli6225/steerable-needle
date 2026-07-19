"""Acceptance tests for the Task 2 grid environment.

All checks are against a single circular obstacle, where every answer is known
in closed form: clearance at a point distance d from the centre of a radius-r
circle is exactly (d - r). These FAIL until the environment is implemented.

Run:  pytest tests/test_grid_environment.py -v
"""

import math

import pytest

from needlesim.models.unicycle_needle import Control, NeedleParams, State
from needlesim.environments.grid_environment import GridEnvironment

CX, CY, R = 50.0, 50.0, 15.0


def make_env(resolution=0.5):
    env = GridEnvironment(width=100.0, height=100.0, resolution=resolution)
    env.add_circle(CX, CY, R)
    env.bake()
    return env


# ---- 1. point collision: inside vs outside ------------------------------


def test_is_free_inside_and_outside():
    env = make_env()
    # Dead centre of the obstacle -> not free.
    assert not env.is_free(State(CX, CY, 0.0))
    # Far corner -> free.
    assert env.is_free(State(10.0, 10.0, 0.0))
    # Just outside the rim (2 mm of clearance) -> free.
    assert env.is_free(State(CX + R + 2.0, CY, 0.0))
    # Just inside the rim -> not free.
    assert not env.is_free(State(CX + R - 2.0, CY, 0.0))


# ---- 2. clearance matches the analytic circle SDF -----------------------


@pytest.mark.parametrize("d", [25.0, 30.0, 40.0])
def test_clearance_matches_analytic(d):
    env = make_env(resolution=0.25)
    # A point at distance d to the right of centre: analytic clearance = d - R.
    state = State(CX + d, CY, 0.0)
    expected = d - R
    got = env.clearance(state)
    assert (
        abs(got - expected) < 2 * env.resolution
    ), f"clearance {got:.3f} vs analytic {expected:.3f} at d={d}"


def test_clearance_negative_inside():
    env = make_env(resolution=0.25)
    # 5 mm inside the rim: analytic signed distance = (R - 5) - R = -5.
    d = R - 5.0
    got = env.clearance(State(CX + d, CY, 0.0))
    assert got < 0, f"expected negative clearance inside obstacle, got {got:.3f}"
    assert abs(got - (d - R)) < 2 * env.resolution


# ---- 3. resolution convergence on the world<->grid transform ------------
# The clearance error against the analytic value should shrink as the grid is
# refined. This is the test that catches a half-cell offset in world_to_grid.


def test_clearance_error_shrinks_with_resolution():
    d = 30.0
    expected = d - R
    errs = []
    for res in [1.0, 0.5, 0.25]:
        env = make_env(resolution=res)
        errs.append(abs(env.clearance(State(CX + d, CY, 0.0)) - expected))
    # Each refinement should not make things worse, and the finest should be
    # meaningfully better than the coarsest (a stuck offset would flatline).
    assert errs[-1] <= errs[0], f"error grew under refinement: {errs}"
    assert (
        errs[-1] < 0.5 * errs[0] + 1e-9
    ), f"error failed to shrink under refinement: {errs}"


# ---- 4. round-trip transform --------------------------------------------


def test_world_grid_roundtrip():
    env = make_env()
    for x, y in [(10.0, 10.0), (50.0, 50.0), (73.3, 21.7)]:
        i, j = env.world_to_grid(x, y)
        xr, yr = env.grid_to_world(i, j)
        # Round-trip should land within half a cell of the original point.
        assert abs(xr - x) <= env.resolution
        assert abs(yr - y) <= env.resolution


# ---- 5. swept arc: grazes vs clears -------------------------------------
# An arc aimed to skim the obstacle should be free at a curvature that bends it
# away and in collision at a curvature that bends it into the circle. Pick
# start/controls so the two cases straddle the rim.


def test_arc_free_vs_collision():
    env = make_env(resolution=0.25)
    params = NeedleParams(kappa=1.0 / 20.0)
    # Start below the obstacle and 6 mm left of centre, heading straight up.
    # A straight shot would pass near the centre; curving b=-1 bends INTO the
    # circle (collides), b=+1 peels away (clears). Short 30 mm arc so neither
    # loops. Spacing v*dt = 0.1 mm <= half-cell 0.125 mm, so the assert passes.
    start = State(CX - 6.0, CY - 25.0, math.pi / 2)

    hit_arc = env.is_arc_free(
        start, Control(v=5.0, b=-1), dt=0.02, n_samples=300, params=params
    )
    free_arc = env.is_arc_free(
        start, Control(v=5.0, b=+1), dt=0.02, n_samples=300, params=params
    )
    assert free_arc and not hit_arc, (
        f"expected b=+1 to clear and b=-1 to collide; "
        f"got free_arc={free_arc}, hit_arc={hit_arc}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

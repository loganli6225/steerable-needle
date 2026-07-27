"""Tests for CCC Dubins steering.

There is NO reference solver to diff against, so these three classes ARE the
correctness argument. The third (execution consistency) is the strongest: it
uses your trusted Task 1 `step` as the oracle for the untrusted geometry.

Run:  pytest tests/test_dubins.py -v
"""

import math

import pytest

from needlesim.models.unicycle_needle import NeedleParams, State, rollout_variable
from needlesim.planning.dubins import dubins_ccc

KAPPA = 1.0 / 20.0  # R = 20 mm
R = 1.0 / KAPPA
V = 5.0
DT = 0.05
PARAMS = NeedleParams(kappa=KAPPA)

# =====================================================================
# CLASS 3 (do this FIRST -- it is the strongest and most general test):
# execution consistency. Whatever the geometry, feeding the returned
# controls to the model must land at the goal.
# =====================================================================


@pytest.mark.parametrize(
    "goal",
    [
        State(10.0, 10.0, math.pi / 2),
        State(-5.0, 15.0, math.pi),
        State(15.0, -8.0, -math.pi / 3),
        State(20.0, 5.0, 0.0),
    ],
)
def test_controls_land_at_goal(goal):
    start = State(0.0, 0.0, 0.0)
    path = dubins_ccc(start, goal, PARAMS, V, DT)
    if path is None:
        pytest.skip("poses not CCC-connectable; covered by reachability test")
    trace = rollout_variable(start, path.controls, PARAMS)
    end = trace[-1]
    # Position within a couple of steps' worth of arc; heading within a few deg.
    assert (
        math.hypot(end.x - goal.x, end.y - goal.y) < 1e-6
    ), f"CCC path ended at ({end.x:.2f},{end.y:.2f}), goal ({goal.x},{goal.y})"
    dtheta = (end.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi
    assert abs(dtheta) < 0.05, f"heading off by {dtheta:.3f} rad"


# =====================================================================
# CLASS 1: analytic length at a hand-computable configuration.
# Pick a config where you can derive the three arc angles by hand and
# thus the exact length. FILL IN the expected value once you have
# derived it -- do not just read it back from your own solver.
# =====================================================================


def test_length_matches_analytic():
    start = State(0.0, 0.0, 0.0)
    goal = State(0.0, 20.0, 0.0)  # only LRL valid; shorter branch wins
    expected_length = 145.878  # = R * 7.29391
    path = dubins_ccc(start, goal, PARAMS, V, DT)
    assert path is not None, "this config should be CCC-connectable"
    assert path.word == "LRL"
    assert (
        abs(path.length - expected_length) < 0.5
    ), f"length {path.length:.3f} vs analytic {expected_length:.3f}"


# =====================================================================
# CLASS 2: the reachability boundary. A CCC path exists only when the
# turning centres are < 4R apart. Sweep separation and confirm the
# None / not-None transition happens at the geometric threshold.
# =====================================================================


def test_reachability_boundary():
    start = State(0.0, 0.0, 0.0)
    # Move the goal far away along +x with matching heading. Far enough and no
    # CCC path exists (would need a straight run). Near enough and it does.
    near = dubins_ccc(start, State(1.5 * R, 0.0, 0.0), PARAMS, V, DT)
    far = dubins_ccc(start, State(10.0 * R, 0.0, 0.0), PARAMS, V, DT)
    assert near is not None, "close, aligned poses should be CCC-connectable"
    assert far is None, "distant poses have no CCC path; should return None"


def test_returns_none_gracefully_not_raise():
    start = State(0.0, 0.0, 0.0)
    # Whatever happens for an unreachable pose, it must be a clean None, not an
    # exception (RRT* calls this in a hot loop and relies on None).
    result = dubins_ccc(start, State(50.0 * R, 0.0, 0.0), PARAMS, V, DT)
    assert result is None


# =====================================================================
# Shorter of the two words: if both RLR and LRL are valid, the returned
# length must be the smaller. (Weak check: just that word is one of the two.)
# =====================================================================


def test_word_is_valid_type():
    start = State(0.0, 0.0, 0.0)
    goal = State(8.0, 8.0, math.pi / 2)
    path = dubins_ccc(start, goal, PARAMS, V, DT)
    if path is not None:
        assert path.word in ("RLR", "LRL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

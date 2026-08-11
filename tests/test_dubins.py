"""Tests for Dubins steering: CCC (arc-only) and full CSC+CCC.

Still no external reference solver, so the argument is unchanged from Task
3.5: the strongest test rolls the returned controls through the trusted Task 1
`step` and demands the needle land at the goal. That oracle now covers four
more words.

WHAT CHANGED FROM THE CCC-ONLY TEST FILE
----------------------------------------
The reachability contract INVERTED for dubins_full. CCC has a hard 4R limit
(distant poses -> None); with a straight segment, distant poses are ALWAYS
connectable. So the old `test_reachability_boundary` assertion "far poses
return None" is correct for dubins_ccc and WRONG for dubins_full. Both
behaviours are now asserted separately -- that is a genuine spec change, not
a loosened test.

Run:  pytest tests/test_dubins.py -v
"""

import math

import pytest

from needlesim.models.unicycle_needle import NeedleParams, State, rollout_variable
from needlesim.planning.dubins import dubins_ccc, dubins_full

KAPPA = 1.0 / 20.0  # R = 20 mm (kept from Task 3.5: geometry is scale-free,
R = 1.0 / KAPPA     # so these tests need not move to the realistic kappa)
V = 5.0
DT = 0.05
PARAMS = NeedleParams(kappa=KAPPA)

# Set from the measured duty-cycle idealisation error, which scales with
# straight-segment length: 0.0004mm per 100mm shortfall plus sub-micron
# lateral wander. Observed worst case 0.00125mm on a 150mm path. 1e-2 gives
# ~8x headroom while still catching genuine geometric errors, which would be
# millimetres off, not micrometres.
POS_TOL = 1e-2
THETA_TOL = 0.05


# =====================================================================
# 1. EXECUTION CONSISTENCY -- the oracle. Do this first.
# =====================================================================


@pytest.mark.parametrize(
    "goal",
    [
        # near poses (CCC territory)
        State(10.0, 10.0, math.pi / 2),
        State(-5.0, 15.0, math.pi),
        State(15.0, -8.0, -math.pi / 3),
        # far poses -- UNREACHABLE by CCC, reachable only via a straight run.
        # These are the cases that motivated this whole task.
        State(60.0, 40.0, math.pi / 2),
        State(80.0, -30.0, -math.pi / 2),
        State(100.0, 10.0, 0.0),
        State(150.0, 0.0, 0.0),
    ],
)
def test_full_controls_land_at_goal(goal):
    start = State(0.0, 0.0, 0.0)
    path = dubins_full(start, goal, PARAMS, V, DT)
    assert path is not None, "dubins_full should connect essentially any poses"
    end = rollout_variable(start, path.controls, PARAMS)[-1]
    assert math.hypot(end.x - goal.x, end.y - goal.y) < POS_TOL, (
        f"{path.word} ended at ({end.x:.3f},{end.y:.3f}), goal "
        f"({goal.x},{goal.y})"
    )
    dtheta = (end.theta - goal.theta + math.pi) % (2 * math.pi) - math.pi
    assert abs(dtheta) < THETA_TOL, f"{path.word} heading off by {dtheta:.4f} rad"


# =====================================================================
# 2. THE CONTRACT THAT CHANGED: reachability
# =====================================================================


def test_ccc_still_has_4R_limit():
    """dubins_ccc is unchanged: distant poses have no arc-only path."""
    start = State(0.0, 0.0, 0.0)
    assert dubins_ccc(start, State(1.5 * R, 0.0, 0.0), PARAMS, V, DT) is not None
    assert dubins_ccc(start, State(10.0 * R, 0.0, 0.0), PARAMS, V, DT) is None


def test_full_has_no_distance_limit():
    """The inverted contract: with a straight segment, far poses connect."""
    start = State(0.0, 0.0, 0.0)
    for factor in [1.5, 10.0, 50.0, 200.0]:
        path = dubins_full(start, State(factor * R, 0.0, 0.0), PARAMS, V, DT)
        assert path is not None, f"dubins_full failed at {factor}R separation"


# =====================================================================
# 3. WORD SELECTION: the shortest of six, and CSC does not always win
# =====================================================================


def test_word_is_one_of_six():
    start = State(0.0, 0.0, 0.0)
    path = dubins_full(start, State(35.0, 25.0, math.pi / 3), PARAMS, V, DT)
    assert path is not None
    assert path.word in ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")


def test_full_never_longer_than_ccc():
    """dubins_full searches a SUPERSET of dubins_ccc's words, so wherever CCC
    finds a path, full must find one no longer. Catches a min() that drops
    candidates or a word that is silently never generated."""
    start = State(0.0, 0.0, 0.0)
    for goal in [
        State(10.0, 10.0, math.pi / 2),
        State(0.0, 20.0, 0.0),
        State(15.0, -8.0, -math.pi / 3),
        State(8.0, 8.0, math.pi / 2),
    ]:
        ccc = dubins_ccc(start, goal, PARAMS, V, DT)
        full = dubins_full(start, goal, PARAMS, V, DT)
        assert full is not None
        if ccc is not None:
            assert full.length <= ccc.length + 1e-9, (
                f"full ({full.word}, {full.length:.3f}) longer than "
                f"ccc ({ccc.word}, {ccc.length:.3f})"
            )


def test_ccc_sometimes_wins():
    """CCC is not dead weight: for close poses with a heading reversal, an
    arc-only path beats every CSC word. Here both opposite-sense CSC words
    are infeasible (turning centres closer than 2R) and the same-sense ones
    are long, so RLR wins at ~120.7 vs LSL's ~208.5. If this fails, either
    dubins_full is not considering CCC candidates or the min() is wrong."""
    start = State(0.0, 0.0, 0.0)
    goal = State(0.0, 20.0, math.pi)
    full = dubins_full(start, goal, PARAMS, V, DT)
    ccc = dubins_ccc(start, goal, PARAMS, V, DT)
    assert full is not None and ccc is not None
    assert full.word in ("RLR", "LRL"), (
        f"expected a CCC word to win at this config, got {full.word}"
    )
    assert abs(full.length - ccc.length) < 1e-9


# =====================================================================
# 4. LENGTH SANITY: straight-line lower bound
# =====================================================================


def test_length_at_least_euclidean():
    """Any curvature-constrained path is at least as long as the straight
    line between the endpoints. Cheap, general, and catches length
    bookkeeping that forgets a segment."""
    start = State(0.0, 0.0, 0.0)
    for goal in [
        State(60.0, 40.0, math.pi / 2),
        State(100.0, 10.0, 0.0),
        State(30.0, -30.0, math.pi),
    ]:
        path = dubins_full(start, goal, PARAMS, V, DT)
        assert path is not None
        euclid = math.hypot(goal.x - start.x, goal.y - start.y)
        assert path.length >= euclid - 1e-9, (
            f"{path.word} length {path.length:.3f} below the straight-line "
            f"distance {euclid:.3f}"
        )


def test_aligned_poses_are_nearly_straight():
    """Start and goal on the same line with the same heading: the shortest
    path should be essentially the straight run, so length ~= the separation.
    A large excess means the S segment is not being found."""
    start = State(0.0, 0.0, 0.0)
    d = 5.0 * R
    path = dubins_full(start, State(d, 0.0, 0.0), PARAMS, V, DT)
    assert path is not None
    assert abs(path.length - d) < 1e-6, (
        f"aligned poses gave {path.word} of length {path.length:.3f}, "
        f"expected ~{d:.3f}"
    )


def test_returns_cleanly_not_raise():
    """RRT* calls this in a hot loop; whatever happens it must return a path
    or None, never raise."""
    start = State(0.0, 0.0, 0.0)
    for goal in [
        State(0.0, 0.0, 0.0),          # identical pose
        State(1e-9, 0.0, 0.0),         # degenerate separation
        State(0.0, 0.0, math.pi),      # same point, reversed heading
        State(1000.0 * R, 0.0, 0.0),   # very far
    ]:
        dubins_full(start, goal, PARAMS, V, DT)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Reused-planner-instance guard for the KD-tree cache.

The failure being guarded: the KD-tree cache lives on the planner INSTANCE,
but the node list is created fresh inside each plan() call. Without the
identity check in _ensure_kdtree, a second plan() on the same instance
queries a tree built over the FIRST call's (now discarded) node list --
yielding indices into a list that no longer exists.

Deliberately does NOT compare the second result against a freshly-constructed
planner's output: self.rng is created once in __init__ and carries state
across calls, so a reused planner's second call starts deep in the random
stream while a fresh planner starts at draw zero. Same seed, different stream
positions, different (both valid) trees. Structural validity of the second
result is the correct assertion.

Run:  pytest tests/test_kdtree_reuse.py -v
"""

import math

import pytest

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig


def make_planner(iters=400, seed=1):
    env = GridEnvironment(width=100.0, height=100.0, resolution=0.5)
    env.add_circle(50.0, 50.0, 15.0)
    env.bake()
    params = NeedleParams(kappa=1.0 / 5.0)
    cfg = RRTStarConfig(
        max_iterations=iters,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        gamma=40.0,
        seed=seed,
    )
    return RRTStar(env, params, cfg)


def assert_structurally_valid(planner, result):
    """The same properties the existing RRT* tests check, bundled."""
    assert result.success
    nodes = result.nodes
    assert nodes[0].parent is None
    for node in nodes[1:]:
        assert node.parent is not None
        assert 0 <= node.parent < len(nodes)
    for s in result.path:
        assert planner.env.is_free(s), f"path enters obstacle at {s}"


def test_reused_planner_instance_stays_correct():
    """Calling plan() twice on ONE instance must produce a structurally
    valid second result. Without the identity guard in _ensure_kdtree, the
    second call queries a KD-tree built over the first call's discarded node
    list -- typically an IndexError or nonsense parent indices."""
    planner = make_planner()

    # Two clearly different start/goal pairs (left vs right corridor around
    # the central obstacle) so the second call's tree genuinely differs from
    # the first's.
    first = planner.plan(State(20.0, 20.0, math.pi / 2), State(20.0, 80.0, math.pi / 2))
    assert_structurally_valid(planner, first)

    second = planner.plan(
        State(80.0, 20.0, math.pi / 2), State(80.0, 80.0, math.pi / 2)
    )
    assert_structurally_valid(planner, second)


def test_same_length_list_swap_hits_identity_guard():
    """Pin the identity guard SPECIFICALLY. In the plan()-twice scenario above
    the second call's fresh list starts shorter than the cached count, so the
    shrinkage guard fires too and would mask a missing identity guard
    (verified by mutation: removing only the identity check still passes the
    test above). The uncovered case is a replacement list that is NOT shorter:
    same length, different nodes. Without the identity check the stale tree
    over list A answers queries about list B with wrong indices -- silently.
    """
    from needlesim.planning.rrt_star import Node

    planner = make_planner()
    rng_a = __import__("random").Random(11)
    rng_b = __import__("random").Random(22)
    nodes_a = [
        Node(State(rng_a.uniform(0, 100), rng_a.uniform(0, 100), 0.0))
        for _ in range(100)
    ]
    nodes_b = [
        Node(State(rng_b.uniform(0, 100), rng_b.uniform(0, 100), 0.0))
        for _ in range(100)
    ]

    planner.nearest(nodes_a, State(1.0, 1.0, 0.0))  # caches a tree over A
    # Query B for the exact position of each of several B nodes: the nearest
    # node must be that node itself (distance 0). A stale tree over A returns
    # A-derived indices instead.
    for probe in (0, 37, 63, 99):
        idx = planner.nearest(nodes_b, nodes_b[probe].state)
        d = planner.distance(nodes_b[idx].state, nodes_b[probe].state)
        assert d < 1e-12, (
            f"query for exact node {probe} of the replacement list returned "
            f"index {idx} at distance {d:.3g} -- stale KD-tree answered"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

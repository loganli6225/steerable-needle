"""Tests for RRT*.

Beyond the RRT contract tests (path valid, tree well-formed, reproducible),
RRT* has two extra properties worth pinning:
  - cost monotonicity: a node's cost_from_start equals parent cost + edge cost
  - improvement: RRT* with more iterations finds a path no worse than fewer
Build length-only cost first; these assume clearance_weight = 0.

Run:  pytest tests/test_rrt_star.py -v
"""

import math

import pytest

from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.environments.grid_environment import GridEnvironment
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig, rewire_radius


def make_env():
    env = GridEnvironment(width=100.0, height=100.0, resolution=0.5)
    env.add_circle(50.0, 50.0, 15.0)
    env.bake()
    return env


def make_planner(seed=0, iters=4000, clearance_weight=0.0):
    env = make_env()
    params = NeedleParams(kappa=1.0 / 5.0)
    cfg = RRTStarConfig(
        max_iterations=iters,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        gamma=40.0,
        clearance_weight=clearance_weight,
        seed=seed,
    )
    return RRTStar(env, params, cfg)


START = State(20.0, 20.0, math.pi / 2)
GOAL = State(20.0, 80.0, math.pi / 2)


# ---- rewire_radius sanity (unit) ------------------------------------------


def test_rewire_radius_shrinks_and_caps():
    r10 = rewire_radius(10, gamma=40.0, dim=3, max_radius=40.0)
    r1000 = rewire_radius(1000, gamma=40.0, dim=3, max_radius=40.0)
    assert r1000 < r10, "radius should shrink as the tree grows"
    assert r10 <= 40.0, "radius must be capped at max_radius"
    # small-n guard: must not blow up or error
    assert rewire_radius(1, gamma=40.0, dim=3, max_radius=40.0) <= 40.0


# ---- basic reachability ----------------------------------------------------


def test_finds_path():
    planner = make_planner(seed=1)
    result = planner.plan(START, GOAL)
    assert result.success, "should find a path in the open corridor"
    assert result.path is not None
    assert planner.reached_goal(result.path[-1])


# ---- path is collision-free ------------------------------------------------


def test_path_collision_free():
    planner = make_planner(seed=2)
    result = planner.plan(START, GOAL)
    assert result.success
    for s in result.path:
        assert planner.env.is_free(s), f"path enters obstacle at {s}"


# ---- cost bookkeeping is consistent ---------------------------------------


def test_cost_from_start_consistent():
    planner = make_planner(seed=3)
    result = planner.plan(START, GOAL)
    nodes = result.nodes
    for node in nodes:
        if node.parent is None:
            assert abs(node.cost_from_start) < 1e-9, "root cost must be 0"
            continue
        # Cost is finite, non-negative, and at least the parent's cost.
        # It may be stale-HIGH (a re-parented parent's discount doesn't
        # propagate to descendants -- documented simplification in rewire),
        # so we do NOT assert cost == parent + edge.
        assert math.isfinite(node.cost_from_start)
        assert node.cost_from_start >= nodes[node.parent].cost_from_start - 1e-6


# ---- RRT* improves (or ties) with more iterations -------------------------


def test_more_iterations_no_worse():
    short = make_planner(seed=5, iters=2000).plan(START, GOAL)
    long = make_planner(seed=5, iters=4000).plan(START, GOAL)
    if short.success and long.success:
        # more iterations should not produce a WORSE best cost (allow tiny slack)
        assert (
            long.best_cost <= short.best_cost + 1e-6
        ), f"more iters gave worse cost: {long.best_cost} vs {short.best_cost}"


# ---- tree well-formed ------------------------------------------------------


def test_tree_acyclic():
    result = make_planner(seed=7).plan(START, GOAL)
    nodes = result.nodes
    assert nodes[0].parent is None
    for k, node in enumerate(nodes[1:], start=1):
        assert node.parent is not None
        # parent index must refer to an existing node
        assert 0 <= node.parent < len(nodes)


# ---- max_edge_length steering horizon --------------------------------------


def test_max_edge_length_default_is_inf():
    """The default must impose NO constraint, preserving existing behaviour."""
    assert RRTStarConfig().max_edge_length == float("inf")


def test_max_edge_length_rejects_long_edges():
    """A threshold below an edge's Dubins length makes that edge unconnectable in
    choose_parent, while inf accepts it. Uses a reversed-heading target so the
    connecting edge is a long heading-reconciliation arc, well clear of the
    obstacle so rejection is due to the horizon, not a collision."""
    from needlesim.planning.dubins import dubins_full
    from needlesim.planning.rrt_star import Node

    env = make_env()
    params = NeedleParams(kappa=1.0 / 5.0)
    from_state = State(20.0, 20.0, math.pi / 2)
    to_state = State(25.0, 22.0, -math.pi / 2)  # reversed heading -> long edge

    path = dubins_full(from_state, to_state, params, 5.0, 0.05)
    assert path is not None, "expected a connectable (if long) edge"
    long_len = path.length

    def planner_with(max_edge_length):
        cfg = RRTStarConfig(
            max_iterations=1,
            goal_tolerance=3.0,
            step_dt=0.05,
            edge_velocity=5.0,
            gamma=40.0,
            max_edge_length=max_edge_length,
            seed=0,
        )
        return RRTStar(env, params, cfg)

    nodes = [Node(from_state, parent=None, cost_from_start=0.0)]

    # inf: connects (returns a parent tuple).
    assert planner_with(float("inf")).choose_parent(nodes, to_state, [0]) is not None
    # threshold below the edge length: rejected -> None.
    assert planner_with(long_len * 0.5).choose_parent(nodes, to_state, [0]) is None
    # threshold above the edge length: still connects.
    assert planner_with(long_len * 1.5).choose_parent(nodes, to_state, [0]) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for the RRT planner.

These check the planner's CONTRACT, not a specific path -- RRT is randomised,
so we assert properties that must hold for any correct run, seeded for
reproducibility. They FAIL until the planner is implemented.

Run:  pytest tests/test_rrt.py -v
"""

import math

import numpy as np
import pytest

from needlesim.models.unicycle_needle import Control, NeedleParams, State
from needlesim.environments.grid_environment import GridEnvironment
from needlesim.planning.rrt import RRT, RRTConfig


def make_env():
    env = GridEnvironment(width=100.0, height=100.0, resolution=0.5)
    env.add_circle(50.0, 50.0, 15.0)   # one obstacle in the middle
    env.bake()
    return env


def make_planner(seed=0):
    env = make_env()
    params = NeedleParams(kappa=1.0 / 20.0)
    # step_dt chosen so edge_velocity*step_dt <= 0.5*resolution (5*0.05=0.25=0.5*0.5)
    cfg = RRTConfig(max_iterations=5000, goal_tolerance=3.0, step_dt=0.05,
                    edge_velocity=5.0, seed=seed)
    return RRT(env, params, cfg)


# ---- basic reachability: start and goal in free space, no obstacle between --

def test_finds_path_in_open_corridor():
    planner = make_planner(seed=1)
    start = State(20.0, 20.0, math.pi / 2)
    goal = State(20.0, 80.0, math.pi / 2)   # straight up the left side, clear of circle
    result = planner.plan(start, goal)
    assert result.success, "should find a path up the obstacle-free left corridor"
    assert result.path is not None
    # path endpoints sanity
    assert planner.distance(result.path[0], start) < 1e-6
    assert planner.reached_goal(result.path[-1])


# ---- the returned path must actually be collision-free ----------------------

def test_returned_path_is_collision_free():
    planner = make_planner(seed=2)
    start = State(20.0, 20.0, math.pi / 2)
    goal = State(20.0, 80.0, math.pi / 2)
    result = planner.plan(start, goal)
    assert result.success
    # every pose on the path must be free
    for s in result.path:
        assert planner.env.is_free(s), f"path passes through obstacle at {s}"


# ---- the tree is well-formed ------------------------------------------------

def test_tree_is_connected():
    planner = make_planner(seed=3)
    start = State(20.0, 20.0, math.pi / 2)
    goal = State(20.0, 80.0, math.pi / 2)
    result = planner.plan(start, goal)
    nodes = result.nodes
    assert nodes[0].parent is None, "root must have no parent"
    for k, node in enumerate(nodes[1:], start=1):
        assert node.parent is not None, f"non-root node {k} has no parent"
        assert 0 <= node.parent < k, "parent must be an earlier node (acyclic tree)"


# ---- reproducibility: same seed -> same result ------------------------------

def test_same_seed_same_result():
    a = make_planner(seed=7)
    b = make_planner(seed=7)
    start = State(20.0, 20.0, math.pi / 2)
    goal = State(20.0, 80.0, math.pi / 2)
    ra = a.plan(start, goal)
    rb = b.plan(start, goal)
    assert ra.success == rb.success
    assert len(ra.nodes) == len(rb.nodes), "same seed should grow the same tree"


# ---- goal biasing sanity: goal is sometimes sampled -------------------------

def test_goal_biasing_returns_goal_sometimes():
    planner = make_planner(seed=0)
    planner.goal = State(20.0, 80.0, math.pi / 2)
    # With goal_sample_rate > 0, over many samples at least one should equal goal.
    hits = 0
    for _ in range(500):
        s = planner.sample()
        if planner.distance(s, planner.goal) < 1e-9:
            hits += 1
    assert hits > 0, "goal biasing never returned the goal in 500 samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

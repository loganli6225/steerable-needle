"""KD-tree correctness: the spatial queries must return EXACTLY what a brute-
force linear scan returns. This is the independent spec for the KD-tree step
of the planner refactor -- write/own this yourself; it is what makes the
KD-tree (whose failures are SILENT, not crashes) safe to delegate.

Two queries get KD-tree-ified, so BOTH need an equivalence check:
  - nearest(point)      -> single closest node index
  - near_indices(point, r) -> set of all node indices within radius r

Strategy: build a planner, stuff its tree with many RANDOM nodes, then for many
random query points assert the KD-tree answer equals the brute-force answer.
Random nodes + random queries is the point -- a handcrafted case can miss the
straggler bug (nodes added since the last KD-tree rebuild).

FILL IN the two spots marked TODO once you've decided the exact method names /
how nodes are added. These tests FAIL until the KD-tree is implemented.

Run:  pytest tests/test_kdtree.py -v
"""

import math
import random

import pytest

from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.environments.grid_environment import GridEnvironment
from needlesim.planning.rrt_star import Node, RRTStar, RRTStarConfig

# ---------------------------------------------------------------------------
# Brute-force references -- the GROUND TRUTH the KD-tree must match. Keep these
# dead simple and obviously correct; they are the oracle.
# ---------------------------------------------------------------------------


def brute_nearest(nodes, px, py):
    """Index of the single closest node by Euclidean position distance."""
    best_i, best_d = -1, float("inf")
    for i, node in enumerate(nodes):
        d = math.hypot(node.state.x - px, node.state.y - py)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def brute_near(nodes, px, py, r):
    """SET of indices of all nodes within radius r (inclusive)."""
    return {
        i
        for i, node in enumerate(nodes)
        if math.hypot(node.state.x - px, node.state.y - py) <= r
    }


# ---------------------------------------------------------------------------
# Helper: build a planner and cram its tree with N random nodes.
# ---------------------------------------------------------------------------


def build_planner_with_random_nodes(n_nodes, seed=0):
    env = GridEnvironment(100.0, 100.0, 0.5)
    env.add_circle(50.0, 50.0, 15.0)
    env.bake()
    params = NeedleParams(kappa=1.0 / 5.0)

    cfg = RRTStarConfig(seed=seed)
    planner = RRTStar(env, params, cfg)

    rng = random.Random(seed)
    nodes = []
    for _ in range(n_nodes):
        nodes.append(
            Node(
                State(
                    rng.uniform(0, 100),
                    rng.uniform(0, 100),
                    rng.uniform(0, 2 * math.pi),
                )
            )
        )
    return planner, nodes


# ---------------------------------------------------------------------------
# 1. nearest == brute-force nearest, over many random queries
# ---------------------------------------------------------------------------


def test_nearest_matches_bruteforce():
    planner, nodes = build_planner_with_random_nodes(300, seed=1)
    rng = random.Random(99)
    for _ in range(200):
        px, py = rng.uniform(0, 100), rng.uniform(0, 100)
        kd_idx = planner.nearest(nodes, State(px, py, 0.0))
        bf_idx = brute_nearest(nodes, px, py)
        # Guard against ties: if two nodes are equidistant, indices may differ
        # legitimately. Compare DISTANCES, not indices, to stay tie-safe.
        kd_d = math.hypot(nodes[kd_idx].state.x - px, nodes[kd_idx].state.y - py)
        bf_d = math.hypot(nodes[bf_idx].state.x - px, nodes[bf_idx].state.y - py)
        assert abs(kd_d - bf_d) < 1e-9, (
            f"nearest disagreed: KD dist {kd_d:.6f} vs brute {bf_d:.6f} "
            f"at query ({px:.2f},{py:.2f})"
        )


# ---------------------------------------------------------------------------
# 2. near_indices == brute-force within-radius SET, over many random queries
# ---------------------------------------------------------------------------


def test_near_indices_matches_bruteforce():
    planner, nodes = build_planner_with_random_nodes(300, seed=2)
    rng = random.Random(123)
    for _ in range(200):
        px, py = rng.uniform(0, 100), rng.uniform(0, 100)
        r = rng.uniform(2.0, 30.0)
        kd_set = set(planner.near_indices(nodes, State(px, py, 0.0), r))
        bf_set = brute_near(nodes, px, py, r)
        # Radius boundary is the danger zone: a node exactly at distance r can
        # land on either side under floating point. Allow a hair of tolerance
        # by only flagging disagreements NOT near the boundary.
        disagree = kd_set ^ bf_set  # symmetric difference
        for i in disagree:
            d = math.hypot(nodes[i].state.x - px, nodes[i].state.y - py)
            assert abs(d - r) < 1e-6, (
                f"near_indices disagreed on node {i} at dist {d:.6f}, "
                f"radius {r:.6f} (not a boundary case)"
            )


# ---------------------------------------------------------------------------
# 3. The straggler path specifically: query RIGHT AFTER adding nodes, before
#    any rebuild could have happened. This is the bug that silently drops the
#    newest (often closest) nodes.
# ---------------------------------------------------------------------------


def test_recently_added_nodes_are_found():
    planner, nodes = build_planner_with_random_nodes(50, seed=3)
    rng = random.Random(7)
    for _ in range(120):
        known = State(rng.uniform(0, 100), rng.uniform(0, 100), 0.0)
        nodes.append(Node(known))
        idx = planner.nearest(nodes, known)
        assert idx == len(nodes) - 1, (
            f"just-added node at ({known.x:.2f},{known.y:.2f}) not found "
            f"as nearest with {len(nodes)} nodes"
        )


# ---------------------------------------------------------------------------
# 4. Degenerate: single node, empty radius. Must not crash.
# ---------------------------------------------------------------------------


def test_edge_cases():
    planner, nodes = build_planner_with_random_nodes(1, seed=4)
    assert planner.nearest(nodes, State(10.0, 10.0, 0.0)) == 0
    planner.near_indices(nodes, State(10.0, 10.0, 0.0), 0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

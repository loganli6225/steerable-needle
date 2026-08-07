"""Time RRT* across iteration counts, to measure the KD-tree swap.

Run this BEFORE the KD-tree change to record the linear-scan baseline, then
again AFTER, and compare. Same seed both times, so the random sequence is
identical and the comparison is apples-to-apples.

The node counts and success flags MUST match between runs. The KD-tree changes
HOW nearest/near_indices find neighbours, not WHAT gets sampled -- so a
differing node count means the KD-tree changed behaviour, which is a
correctness red flag, not just a performance one.

Watch the SCALING, not just the totals: the linear-scan version does an O(n)
scan per iteration on a growing tree, so time should grow super-linearly
(roughly quadratic). A working KD-tree should flatten that curve. The scaling
shape is what determines whether the full benchmark (3 planners x N seeds) is
feasible at all.

Run:  python scripts/time_rrt_star.py
"""

import math
import time

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig

# --- fixed scenario: matches tests/test_rrt_star.py so numbers are comparable
WIDTH, HEIGHT, RESOLUTION = 100.0, 100.0, 0.5
OBSTACLE = (50.0, 50.0, 15.0)  # cx, cy, r
KAPPA = 1.0 / 5.0
GAMMA = 40.0
SEED = 1
START = State(20.0, 20.0, math.pi / 2)
GOAL = State(20.0, 80.0, math.pi / 2)

ITERATION_COUNTS = [500, 1500, 3000]


def build_planner(iters: int) -> RRTStar:
    env = GridEnvironment(width=WIDTH, height=HEIGHT, resolution=RESOLUTION)
    env.add_circle(*OBSTACLE)
    env.bake()
    params = NeedleParams(kappa=KAPPA)
    cfg = RRTStarConfig(
        max_iterations=iters,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        gamma=GAMMA,
        seed=SEED,
    )
    return RRTStar(env, params, cfg)


def main() -> None:
    print(f"RRT* timing | seed={SEED} kappa={KAPPA} gamma={GAMMA}")
    print(f"{'iters':>7} {'time [s]':>10} {'nodes':>8} {'success':>8} {'s/iter':>10}")
    print("-" * 48)

    results = []
    for iters in ITERATION_COUNTS:
        planner = build_planner(iters)
        t0 = time.perf_counter()
        result = planner.plan(START, GOAL)
        elapsed = time.perf_counter() - t0
        results.append((iters, elapsed, len(result.nodes), result.success))
        print(
            f"{iters:>7} {elapsed:>10.2f} {len(result.nodes):>8} "
            f"{str(result.success):>8} {elapsed/iters:>10.5f}"
        )

    # Scaling: if time were linear in iterations, doubling iters doubles time
    # (ratio ~= the iteration ratio). Quadratic behaviour shows up as a ratio
    # noticeably LARGER than the iteration ratio.
    print("\nscaling between consecutive rows:")
    for (i0, t0_, _, _), (i1, t1_, _, _) in zip(results, results[1:]):
        iter_ratio = i1 / i0
        time_ratio = t1_ / t0_ if t0_ > 0 else float("inf")
        print(
            f"  {i0} -> {i1}: iters x{iter_ratio:.2f}, time x{time_ratio:.2f}"
            f"   ({'super-linear' if time_ratio > iter_ratio * 1.2 else 'near-linear'})"
        )

    print("\nRecord these numbers. After the KD-tree swap, node counts and")
    print("success flags must be IDENTICAL; only the times should change.")


if __name__ == "__main__":
    main()

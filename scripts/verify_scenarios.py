"""Measure whether each hand-designed scenario sits in a useful difficulty band.

For every scenario, run KinodynamicRRT and VanillaRRT across a handful of
seeds and print success rate + mean iterations-to-solution + mean wall-clock
over the SUCCESSFUL runs (means over failures are meaningless -- a failure
always spends the whole budget).

Why this exists: a benchmark scenario only discriminates planners if it is
neither trivial nor impossible. The target regime is KinodynamicRRT succeeding
MOST of the time but not always -- roughly 50-90%. Outside that band:

  - ~100% for both planners: too easy, measures only raw speed (vanilla wins
    by construction, so nothing is learned about feasibility).
  - ~0%: nothing is learned; the row is empty.

This script REPORTS the numbers. It does not tune geometry -- scenario
difficulty is a design decision the owner makes from this table.

Run:  python scripts/verify_scenarios.py
"""

from __future__ import annotations

import time

from needlesim.benchmark.scenarios import COMMON_CONDITIONS, HAND_DESIGNED, build_env
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig, VanillaRRT

N_SEEDS = 10
MAX_ITERATIONS = (
    20000  # the benchmark budget: "failure" means unsolved, not out-of-iters
)

# Shared planner config, IDENTICAL for both planners -- comparing at different
# margins/velocities/tolerances would be invalid. Pinned to COMMON_CONDITIONS.
COMMON_CFG = dict(
    max_iterations=MAX_ITERATIONS,
    goal_tolerance=COMMON_CONDITIONS["goal_tolerance"],
    step_dt=COMMON_CONDITIONS["step_dt"],
    edge_velocity=COMMON_CONDITIONS["edge_velocity"],
    margin=COMMON_CONDITIONS["margin"],
)

PLANNERS = [("KinodynamicRRT", KinodynamicRRT), ("VanillaRRT", VanillaRRT)]


def run_planner(planner_cls, scenario, seed):
    """One (planner, scenario, seed) run. Fresh env per run so a baked SDF is
    never shared. Returns (success, n_iterations, elapsed_seconds)."""
    env = build_env(scenario)
    params = NeedleParams(kappa=COMMON_CONDITIONS["kappa"])
    cfg = RRTConfig(seed=seed, **COMMON_CFG)
    planner = planner_cls(env, params, cfg)

    t0 = time.perf_counter()
    result = planner.plan(scenario.start, scenario.goal)
    elapsed = time.perf_counter() - t0
    return result.success, result.n_iterations, elapsed


def summarise(rows):
    """rows: list of (success, iters, time). Returns (n_success, mean_iters,
    mean_time) with means taken over successes only (None if none succeeded)."""
    successes = [(it, t) for ok, it, t in rows if ok]
    n = len(successes)
    if n == 0:
        return 0, None, None
    mean_iters = sum(it for it, _ in successes) / n
    mean_time = sum(t for _, t in successes) / n
    return n, mean_iters, mean_time


def main():
    print(
        f"Difficulty check: {N_SEEDS} seeds x {len(HAND_DESIGNED)} scenarios "
        f"x {len(PLANNERS)} planners, budget {MAX_ITERATIONS} iters"
    )
    print(
        f"Common: {COMMON_CONDITIONS['width']:.0f}x{COMMON_CONDITIONS['height']:.0f}mm "
        f"@ {COMMON_CONDITIONS['resolution']}mm, kappa={COMMON_CONDITIONS['kappa']:.4f}, "
        f"margin={COMMON_CONDITIONS['margin']}, goal_tol={COMMON_CONDITIONS['goal_tolerance']}"
    )
    print()
    header = (
        f"{'scenario':<20} {'planner':<16} {'success':>8} "
        f"{'mean_iters*':>12} {'mean_t[s]*':>11}"
    )
    print(header)
    print("-" * len(header))

    for scenario in HAND_DESIGNED:
        for planner_name, planner_cls in PLANNERS:
            rows = [run_planner(planner_cls, scenario, seed) for seed in range(N_SEEDS)]
            n_success, mean_iters, mean_time = summarise(rows)
            rate = f"{n_success}/{N_SEEDS}"
            iters_str = (
                f"{mean_iters:>12.0f}" if mean_iters is not None else f"{'--':>12}"
            )
            time_str = f"{mean_time:>11.2f}" if mean_time is not None else f"{'--':>11}"
            print(
                f"{scenario.name:<20} {planner_name:<16} {rate:>8} "
                f"{iters_str} {time_str}"
            )
        print()

    print("* means over successful runs only. Target for KinodynamicRRT: ~50-90%.")


if __name__ == "__main__":
    main()

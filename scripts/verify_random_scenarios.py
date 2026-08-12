"""Verify (do NOT tune) the random scenario set: report the outcome distribution.

Loads the persisted 30 random scenarios and runs KinodynamicRRT and VanillaRRT
at a few seeds each, then reports how many scenarios fall into each outcome
class:

  - both      : solved by kinodynamic AND vanilla (some seed each)
  - vanilla   : solved by vanilla only (curvature-limited for kinodynamic)
  - kino-only : solved by kinodynamic only (rare; flagged if it happens)
  - neither   : solved by no planner at any tried seed (accepted -- see below)

A scenario counts as "solved" by a planner if ANY of its tried seeds reached
the goal. This is a coarse feasibility read, not the full 10-seed benchmark.

This is VERIFICATION, not tuning. The distribution is the experimental design
and is reported as-is. In particular: with up to 7 obstacles at up to 25mm
radius / 40mm sides in a 150mm workspace, the dense tail may be largely
unsolvable for everyone. If a large fraction lands in "neither", that is a
PROPERTY OF THE STATED DISTRIBUTION and belongs in the write-up as context for
the aggregate numbers -- NOT a reason to quietly change the ranges.

RUNTIME: 30 scenarios x 2 planners x VERIFY_SEEDS seeds, budget 20k iters.
Unsolvable scenarios spend the full budget every seed, so this is the slow
part -- on the order of 15-30 min. Run it in the background.

Run:  python scripts/verify_random_scenarios.py
"""

from __future__ import annotations

import time
from pathlib import Path

from needlesim.benchmark.random_scenarios import load_scenarios
from needlesim.benchmark.scenarios import COMMON_CONDITIONS, build_env
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig, VanillaRRT

IN_PATH = Path("experiments/random_scenarios.json")
VERIFY_SEEDS = 3
MAX_ITERATIONS = 20000

COMMON_CFG = dict(
    max_iterations=MAX_ITERATIONS,
    goal_tolerance=COMMON_CONDITIONS["goal_tolerance"],
    step_dt=COMMON_CONDITIONS["step_dt"],
    edge_velocity=COMMON_CONDITIONS["edge_velocity"],
    margin=COMMON_CONDITIONS["margin"],
)


def _solved_by(planner_cls, scenario, env) -> tuple[bool, int]:
    """(solved by any tried seed, count of seeds that solved)."""
    params = NeedleParams(kappa=COMMON_CONDITIONS["kappa"])
    n_ok = 0
    for seed in range(VERIFY_SEEDS):
        planner = planner_cls(env, params, RRTConfig(seed=seed, **COMMON_CFG))
        if planner.plan(scenario.start, scenario.goal).success:
            n_ok += 1
    return n_ok > 0, n_ok


def main():
    seed, scenarios = load_scenarios(IN_PATH)
    print(f"Loaded {len(scenarios)} random scenarios (gen seed {seed}) from {IN_PATH}")
    print(f"Verify: {VERIFY_SEEDS} seeds/planner, budget {MAX_ITERATIONS} iters\n")

    header = f"{'scenario':<12} {'#obs':>4} {'kino':>7} {'vanilla':>8} {'class':>10}"
    print(header)
    print("-" * len(header))

    tally = {"both": 0, "vanilla": 0, "kino-only": 0, "neither": 0}
    t0 = time.perf_counter()
    for scen in scenarios:
        env = build_env(scen)
        k_ok, k_n = _solved_by(KinodynamicRRT, scen, env)
        v_ok, v_n = _solved_by(VanillaRRT, scen, env)
        if k_ok and v_ok:
            cls = "both"
        elif v_ok:
            cls = "vanilla"
        elif k_ok:
            cls = "kino-only"
        else:
            cls = "neither"
        tally[cls] += 1
        print(
            f"{scen.name:<12} {len(scen.obstacles):>4} "
            f"{k_n}/{VERIFY_SEEDS:<5} {v_n}/{VERIFY_SEEDS:<6} {cls:>10}"
        )

    print("-" * len(header))
    n = len(scenarios)
    print(f"\nOutcome distribution over {n} scenarios:")
    for cls in ("both", "vanilla", "kino-only", "neither"):
        print(f"  {cls:<10} {tally[cls]:>3}  ({100 * tally[cls] / n:.0f}%)")
    print(f"\nTotal wall-clock: {time.perf_counter() - t0:.1f}s")
    print(
        '"neither" reflects the stated distribution (dense/large-obstacle '
        "tail),\nnot a defect -- report it, do not retune the ranges."
    )


if __name__ == "__main__":
    main()

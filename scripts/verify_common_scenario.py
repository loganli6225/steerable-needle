"""Verify all three planners succeed on ONE common scenario.

The benchmark's validity rests on every planner facing identical conditions:
same world, same obstacle, same kappa, same MARGIN, same start/goal. This
script is the gate -- if a planner fails here, the scenario (or that planner's
budget) needs adjusting before any benchmark harness is written.

Two things are genuinely unknown and are what this script measures:

  1. Does RRT* succeed at margin=2.0? It has only ever run at margin=0.0.
     A 2mm margin shrinks the free corridor, and RRT*'s Dubins edges are long
     -- this is the most likely thing to break.
  2. Does KinodynamicRRT behave at kappa=1/5? It was tested at 1/20 (R=20mm).
     R=5mm is a MORE agile needle so it should be easier, but each extend arc
     now curves more sharply over the same n_steps_per_extend. Verify, don't
     assume.

NOTE ON CONFIGS: RRTConfig and RRTStarConfig carry different planner-specific
fields (n_steps_per_extend vs gamma/dim/max_radius), so they cannot share one
object. The SHARED fields below are set identically by construction -- that is
the whole point. If you change one, change it in COMMON only.

Run:  python scripts/verify_common_scenario.py
"""

import math
import time

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig, VanillaRRT
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig

# --- the candidate common scenario -----------------------------------------
WIDTH, HEIGHT, RESOLUTION = 150.0, 150.0, 0.5
OBSTACLE = (75.0, 75.0, 20.0)  # cx, cy, r
KAPPA = 1.0 / 50.0
START = State(30.0, 20.0, math.pi / 2)
GOAL = State(30.0, 130.0, math.pi / 2)

# Shared config fields -- IDENTICAL for all three planners. Comparing planners
# at different margins (or velocities, or goal tolerances) is invalid.
COMMON = dict(
    goal_tolerance=3.0,
    step_dt=0.05,
    edge_velocity=5.0,
    margin=2.0,  # needle OD + safety; RRT* has never run at this
    seed=1,
)

# Budgets are per-planner: RRT stops at first goal contact, RRT* always runs
# the full budget. Equal iteration counts would NOT be equal work.
RRT_ITERS = 20000
RRTSTAR_ITERS = 10000


def build_env():
    env = GridEnvironment(width=WIDTH, height=HEIGHT, resolution=RESOLUTION)
    env.add_circle(*OBSTACLE)
    env.bake()
    return env


def run(name, planner):
    t0 = time.perf_counter()
    result = planner.plan(START, GOAL)
    elapsed = time.perf_counter() - t0

    # RRTResult and RRTStarResult differ: only the latter carries best_cost.
    cost = getattr(result, "best_cost", None)
    cost_str = f"{cost:>8.1f}" if isinstance(cost, float) and math.isfinite(cost) else "     n/a"

    print(
        f"{name:<18} {str(result.success):<8} {len(result.nodes):>7} "
        f"{result.n_iterations:>7} {cost_str} {elapsed:>8.2f}"
    )
    return result.success


def main():
    params = NeedleParams(kappa=KAPPA)
    print(f"Common scenario: {WIDTH}x{HEIGHT}mm, circle{OBSTACLE}, kappa={KAPPA}")
    print(f"  margin={COMMON['margin']}  start=({START.x},{START.y})  "
          f"goal=({GOAL.x},{GOAL.y})  seed={COMMON['seed']}")
    print()
    print(f"{'planner':<18} {'success':<8} {'nodes':>7} {'iters':>7} {'cost':>8} {'time[s]':>8}")
    print("-" * 60)

    results = {}

    rrt_cfg = RRTConfig(max_iterations=RRT_ITERS, **COMMON)
    results["VanillaRRT"] = run("VanillaRRT", VanillaRRT(build_env(), params, rrt_cfg))

    rrt_cfg2 = RRTConfig(max_iterations=RRT_ITERS, **COMMON)
    results["KinodynamicRRT"] = run(
        "KinodynamicRRT", KinodynamicRRT(build_env(), params, rrt_cfg2)
    )

    star_cfg = RRTStarConfig(max_iterations=RRTSTAR_ITERS, gamma=40.0, **COMMON)
    results["RRTStar"] = run("RRTStar", RRTStar(build_env(), params, star_cfg))

    print()
    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("The scenario is not yet common ground. Options, in order of")
        print("preference: raise that planner's budget; loosen margin (but")
        print("then ALL planners use the new value); move start/goal; adjust")
        print("kappa (re-verify RRT* if you do -- it is the sensitive one).")
    else:
        print("All three succeed. This scenario is valid common ground for")
        print("the benchmark. Record these settings; the harness must use")
        print("them (or a documented variant) for every planner.")


if __name__ == "__main__":
    main()

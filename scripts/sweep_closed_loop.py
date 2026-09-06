"""Measure open-loop vs closed-loop across a kappa mismatch sweep.

Runs both scenarios (open, constrained_passage) over five seeds and four model
curvatures. Prints a per-seed table and then an aggregate summary across all
seeds. The summary is the point: single seeds vary substantially, so medians,
worst cases, and how runs ended are what characterise the behaviour.

The termination reason is what makes the error column interpretable: stopping
at 58mm because the controller gave up (replan_cap) is a different result
from wandering to 58mm because it would not.

Produced the Phase 3.5 sweep tables recorded in docs/roadmap.md. Rerun after
Phase 4's learned kappa to show what fixing the model buys over merely
re-aiming from a better pose.

Run:  python scripts/sweep_closed_loop.py
"""

import collections
import math
import statistics
import time

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.control.closed_loop import (
    ClosedLoopConfig,
    run_closed_loop,
    run_open_loop,
)
from needlesim.estimation.ekf import EKFConfig
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

TRUE_KAPPA = 1.0 / 50.0
SEEDS = range(1, 6)
MODEL_KAPPAS = [1 / 50, 1 / 40, 1 / 33, 1 / 25]

# Short labels so the reason fits a column.
SHORT = {
    "goal": "goal",
    "plan_exhausted": "plan_end",
    "replan_cap": "gave_up",
    "max_steps": "budget",
    "no_initial_plan": "no_plan",
    "left_bounds": "off_map",
    "unknown": "?",
}


def make_env(scenario):
    return build_env(scenario)


def make_planner(env, kappa, seed):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    # NOTE: the planner is built with the MODEL kappa -- it believes what the
    # filter believes, not what the world does.
    return KinodynamicRRT(env, NeedleParams(kappa=kappa), cfg)


def main():
    for scenario in (OPEN, CONSTRAINED_PASSAGE):
        t0 = time.perf_counter()
        start = scenario.start
        goal = scenario.goal

        # results[model_kappa] = list of (ol, cl) result pairs, one per seed
        results = {k: [] for k in MODEL_KAPPAS}
        for seed in SEEDS:
            print(f"true kappa = 1/{1/TRUE_KAPPA:.0f}, seed = {seed}")
            print(
                f"{'model':>8} {'ol err':>8} {'ol hit':>7} {'ol steps':>9} "
                f"{'ol end':>9} {'cl err':>8} {'cl hit':>7} {'replans':>8} "
                f"{'fails':>6} {'cl steps':>9} {'cl end':>9}"
            )
            print("-" * 106)

            for model_kappa in MODEL_KAPPAS:
                env = make_env(scenario)
                ol = run_open_loop(
                    planner=make_planner(env, model_kappa, seed),
                    env=env,
                    true_params=NeedleParams(kappa=TRUE_KAPPA),
                    model_params=NeedleParams(kappa=model_kappa),
                    start=start,
                    goal=goal,
                )

                env2 = make_env(scenario)
                cl = run_closed_loop(
                    planner=make_planner(env2, model_kappa, seed),
                    env=env2,
                    true_params=NeedleParams(kappa=TRUE_KAPPA),
                    model_params=NeedleParams(kappa=model_kappa),
                    start=start,
                    goal=goal,
                    ekf_config=EKFConfig(seed=seed),
                    loop_config=ClosedLoopConfig(seed=seed),
                )

                results[model_kappa].append((ol, cl))

                # reached_goal is now redundant with end == "goal", so the goal
                # columns are dropped in favour of the reason.
                print(
                    f"    1/{1/model_kappa:<4.0f} {ol.final_error_mm:8.1f} "
                    f"{str(ol.collided):>7} {ol.n_steps:9d} "
                    f"{SHORT[ol.termination_reason]:>9} "
                    f"{cl.final_error_mm:8.1f} {str(cl.collided):>7} "
                    f"{len(cl.replan_steps):8d} {cl.n_replan_failures:6d} "
                    f"{cl.n_steps:9d} {SHORT[cl.termination_reason]:>11}"
                )
            print()

        # ---------------- aggregate summary ----------------
        print("=" * 106)
        print(f"SUMMARY over {len(list(SEEDS))} seeds, scenario = {scenario.name}")
        print("Runs where the planner failed outright (error = inf) are excluded from")
        print("the error statistics and counted separately -- averaging in an inf, or")
        print("silently dropping it, would both misrepresent the result.")
        print()
        print(
            f"{'model':>8} {'n_ok':>5} {'ol med':>8} {'cl med':>8} {'cl worst':>9} "
            f"{'cl beats ol':>12} {'cl collide':>11} {'mean fails':>11}  how cl ended"
        )
        print("-" * 106)

        for model_kappa in MODEL_KAPPAS:
            pairs = results[model_kappa]
            # Keep only pairs where BOTH produced a finite error, so the medians
            # compare like with like.
            ok = [
                (ol, cl)
                for ol, cl in pairs
                if math.isfinite(ol.final_error_mm) and math.isfinite(cl.final_error_mm)
            ]
            if not ok:
                print(
                    f"    1/{1/model_kappa:<4.0f}  no runs where both planners succeeded"
                )
                continue

            ol_errs = [ol.final_error_mm for ol, _ in ok]
            cl_errs = [cl.final_error_mm for _, cl in ok]
            beats = sum(1 for ol, cl in ok if cl.final_error_mm < ol.final_error_mm)
            cl_hits = sum(1 for _, cl in ok if cl.collided)
            mean_fails = statistics.mean(cl.n_replan_failures for _, cl in ok)
            reasons = collections.Counter(SHORT[cl.termination_reason] for _, cl in ok)
            reason_str = " ".join(f"{r}={n}" for r, n in reasons.most_common())

            print(
                f"    1/{1/model_kappa:<4.0f} {len(ok):5d} "
                f"{statistics.median(ol_errs):8.1f} {statistics.median(cl_errs):8.1f} "
                f"{max(cl_errs):9.1f} {beats:>7d}/{len(ok):<4d} "
                f"{cl_hits:>7d}/{len(ok):<3d} {mean_fails:11.1f}  {reason_str}"
            )

        print()
        print("How to read it:")
        print("  cl med vs ol med  -- does replanning help TYPICALLY?")
        print("  cl worst          -- how bad is the tail? A method that usually")
        print("                       works and occasionally ends 60mm out is not")
        print("                       clinically deployable, so the tail matters as")
        print("                       much as the median.")
        print("  cl beats ol       -- how OFTEN does it help? A good median with a")
        print("                       3/5 win rate is a different claim from 5/5.")
        print("  how cl ended      -- THE column that makes the error interpretable.")
        print("                       'gave_up' means the controller hit the replan")
        print("                       cap and halted: it KNEW it was lost. That is a")
        print("                       different (and more deployable) failure than")
        print("                       silently ending far from target.")
        print()
        print(f"total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

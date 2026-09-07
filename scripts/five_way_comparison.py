"""Five-way comparison of open-loop, closed-loop and adaptive control.

    1 open        plan once, execute blind
    2 fixed       closed loop, drift trigger, planner keeps the wrong kappa
    3 learn/drift closed loop, drift trigger, planner rebuilt from the estimate
    4 learn/kappa closed loop, kappa-change trigger, ditto
    5 learn/union closed loop, EITHER trigger, ditto

1 vs 2 isolates what replanning buys; 2 vs 3 what fixing the MODEL buys on top
of it; 3 vs 4 whether drift is the right thing to trigger on; and 5 asks
whether catching both signals beats catching either.

The union was originally motivated by the observation that, run ALONE, the two
triggers fire at almost disjoint times -- kappa-change around steps 21-81,
drift around 101-281 -- suggesting a union would catch two distinct signals.
That inference is wrong, and this script is what showed it: each trigger
changes the trajectory the other sees, so under a union kappa-replanning
refreshes the path before drift can accumulate, and drift is pre-empted.

WATCH condition 5's replan_steps AGAINST condition 4's, not its count against
3+4. Measured, they are IDENTICAL in 9 of 10 runs; the lone exception is open
seed 1, where drift contributes a single late replan (step 381) that helps.
The near-identity is the point: it demonstrates that drift is largely inert
once model-triggered replanning is active.

Run:  python scripts/five_way_comparison.py
"""

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.control.adaptive_loop import (
    AdaptiveLoopConfig,
    ReplanTrigger,
    run_closed_loop_adaptive,
)
from needlesim.control.closed_loop import (
    ClosedLoopConfig,
    run_closed_loop,
    run_open_loop,
)
from needlesim.estimation.ekf import EKFConfig
from needlesim.estimation.ekf_augmented import AugmentedEKFConfig
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

TRUE, GUESS = 1 / 50, 1 / 25


def make_planner(env, kappa, seed):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, NeedleParams(kappa=kappa), cfg)


def adaptive(scenario, seed, trigger, args):
    """Conditions 3, 4 and 5 differ only in the trigger."""
    e = build_env(scenario)
    return run_closed_loop_adaptive(
        make_planner(e, GUESS, seed),
        e,
        NeedleParams(kappa=TRUE),
        GUESS,
        ekf_config=AugmentedEKFConfig(seed=seed),
        loop_config=AdaptiveLoopConfig(trigger=trigger, seed=seed),
        **args,
    )


for scenario in (CONSTRAINED_PASSAGE, OPEN):
    print(f"\n=== {scenario.name} | true 1/50, guess 1/25 ===")
    print(
        f"{'seed':>5} {'1 open':>8} {'2 fixed':>8} {'3 drift':>9} "
        f"{'4 kappa':>9} {'5 union':>9} {'k3':>8} {'k5':>8}"
    )
    print("-" * 76)
    for seed in range(1, 6):
        args = dict(start=scenario.start, goal=scenario.goal)

        e = build_env(scenario)
        r1 = run_open_loop(
            make_planner(e, GUESS, seed),
            e,
            NeedleParams(kappa=TRUE),
            NeedleParams(kappa=GUESS),
            **args,
        )

        e = build_env(scenario)
        r2 = run_closed_loop(
            make_planner(e, GUESS, seed),
            e,
            NeedleParams(kappa=TRUE),
            NeedleParams(kappa=GUESS),
            ekf_config=EKFConfig(seed=seed),
            loop_config=ClosedLoopConfig(seed=seed),
            **args,
        )

        r3 = adaptive(scenario, seed, ReplanTrigger.DRIFT, args)
        r4 = adaptive(scenario, seed, ReplanTrigger.KAPPA_CHANGE, args)
        r5 = adaptive(scenario, seed, ReplanTrigger.UNION, args)

        k3 = r3.kappa_estimates[-1] if r3.kappa_estimates else float("nan")
        k5 = r5.kappa_estimates[-1] if r5.kappa_estimates else float("nan")

        print(
            f"{seed:5d} {r1.final_error_mm:8.1f} {r2.final_error_mm:8.1f} "
            f"{r3.final_error_mm:9.1f} {r4.final_error_mm:9.1f} "
            f"{r5.final_error_mm:9.1f} 1/{1/k3:6.1f} 1/{1/k5:6.1f}"
        )
        print(
            f"      {'':8} {len(r2.replan_steps):8d} {len(r3.replan_steps):9d} "
            f"{len(r4.replan_steps):9d} {len(r5.replan_steps):9d}   (replans)"
        )
        print(
            f"        drift {r3.replan_steps}  kappa {r4.replan_steps}  "
            f"union {r5.replan_steps}"
        )

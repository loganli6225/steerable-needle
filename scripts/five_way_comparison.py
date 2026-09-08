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

TWO WORLDS (Phase 4c baseline)
------------------------------
The simulator can run in either of two worlds, selected by a `World`:

  CONSTANT_WORLD  the historical setup -- true kappa 1/50, belief 1/25, a 2x
                  constant mismatch. This is the DEFAULT when run as a script,
                  so the recorded five-way numbers stay reproducible.
  FIELD_WORLD     Phase 4b's depth-dependent tissue field drives the simulator
                  (NeedleParams.kappa_field=PROSTATE_PATH) while the planner
                  and filters keep a single scalar belief of 1/29 -- a
                  thickness-weighted mean of the four layer radii over the
                  150mm workspace ((45*34+20*22+5*15+80*28)/150 = 28.6mm), the
                  kind of uninformed average prior a clinician might hold, NOT
                  a fitted value. This mismatch (varying world, scalar belief)
                  is what Phase 4c exists to close; this run produces the
                  number 4c must beat.

The field is opt-in and reaches ONLY the simulator, through the `true_params`
argument every run function already takes -- no planner, filter or model logic
changes. The learned field of 4c will slot into the same `World.field` switch.

Run:  python scripts/five_way_comparison.py            # CONSTANT_WORLD
      (FIELD_WORLD is driven by scripts/four_c_baseline.py)
"""

from __future__ import annotations

from dataclasses import dataclass

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
from needlesim.models.tissue_field import TissueField
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig


@dataclass(frozen=True)
class World:
    """One simulator world for the comparison.

    field       None -> the simulator runs on the scalar `true_scalar`
                (constant kappa). A TissueField -> the simulator runs on the
                field and `true_scalar` becomes INERT (`_time_deriv` branches
                to the field), kept only as the reference the augmented loop's
                `final_kappa_error` is measured against, so it is set to the
                belief's average rather than left stale.
    true_scalar the simulator's scalar kappa (constant world), or the inert
                reference above (field world).
    guess       the single scalar belief the planner and both filters hold.
    """

    name: str
    field: TissueField | None
    true_scalar: float
    guess: float

    def true_params(self) -> NeedleParams:
        return NeedleParams(kappa=self.true_scalar, kappa_field=self.field)


# The historical constant-mismatch setup -- the DEFAULT, so recorded numbers
# reproduce byte-for-byte.
CONSTANT_WORLD = World(name="constant", field=None, true_scalar=1 / 50, guess=1 / 25)

# Phase 4b field in the world, scalar belief everywhere else. `true_scalar` is
# inert here (the field drives the sim); it is set to 1/29 so the augmented
# loop's final_kappa_error is reported against the same thickness-weighted mean
# the belief uses, not a stale number.
FIELD_WORLD = World(name="field", field=TissueField(), true_scalar=1 / 29, guess=1 / 29)


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


def adaptive(scenario, seed, trigger, world, ekf_config=None):
    """Conditions 3, 4 and 5 differ only in the trigger.

    `ekf_config` defaults to AugmentedEKFConfig(seed=seed); the process-noise
    sweep passes an explicit one.
    """
    e = build_env(scenario)
    return run_closed_loop_adaptive(
        make_planner(e, world.guess, seed),
        e,
        world.true_params(),
        world.guess,
        start=scenario.start,
        goal=scenario.goal,
        ekf_config=ekf_config or AugmentedEKFConfig(seed=seed),
        loop_config=AdaptiveLoopConfig(trigger=trigger, seed=seed),
    )


def run_conditions(scenario, seed, world):
    """All five conditions for one (scenario, seed, world). Returns the five
    result objects in order (open, fixed, drift, kappa, union)."""
    args = dict(start=scenario.start, goal=scenario.goal)

    e = build_env(scenario)
    r1 = run_open_loop(
        make_planner(e, world.guess, seed),
        e,
        world.true_params(),
        NeedleParams(kappa=world.guess),
        **args,
    )

    e = build_env(scenario)
    r2 = run_closed_loop(
        make_planner(e, world.guess, seed),
        e,
        world.true_params(),
        NeedleParams(kappa=world.guess),
        ekf_config=EKFConfig(seed=seed),
        loop_config=ClosedLoopConfig(seed=seed),
        **args,
    )

    r3 = adaptive(scenario, seed, ReplanTrigger.DRIFT, world)
    r4 = adaptive(scenario, seed, ReplanTrigger.KAPPA_CHANGE, world)
    r5 = adaptive(scenario, seed, ReplanTrigger.UNION, world)
    return r1, r2, r3, r4, r5


def emit_table(world, scenarios=(CONSTRAINED_PASSAGE, OPEN), seeds=range(1, 6)):
    """Print the five-way table for a world, in the historical format.

    Returns {(scenario_name, seed): (r1, r2, r3, r4, r5)} so callers (the 4c
    baseline) can instrument the same runs without re-running them. Printed
    output is unchanged by this.
    """
    results = {}
    for scenario in scenarios:
        print(
            f"\n=== {scenario.name} | world={world.name} | "
            f"true 1/{1 / world.true_scalar:.0f}, guess 1/{1 / world.guess:.0f} ==="
        )
        print(
            f"{'seed':>5} {'1 open':>8} {'2 fixed':>8} {'3 drift':>9} "
            f"{'4 kappa':>9} {'5 union':>9} {'k3':>8} {'k5':>8}"
        )
        print("-" * 76)
        for seed in seeds:
            r1, r2, r3, r4, r5 = run_conditions(scenario, seed, world)
            results[(scenario.name, seed)] = (r1, r2, r3, r4, r5)

            k3 = r3.kappa_estimates[-1] if r3.kappa_estimates else float("nan")
            k5 = r5.kappa_estimates[-1] if r5.kappa_estimates else float("nan")

            print(
                f"{seed:5d} {r1.final_error_mm:8.1f} {r2.final_error_mm:8.1f} "
                f"{r3.final_error_mm:9.1f} {r4.final_error_mm:9.1f} "
                f"{r5.final_error_mm:9.1f} 1/{1 / k3:6.1f} 1/{1 / k5:6.1f}"
            )
            print(
                f"      {'':8} {len(r2.replan_steps):8d} "
                f"{len(r3.replan_steps):9d} "
                f"{len(r4.replan_steps):9d} {len(r5.replan_steps):9d}   (replans)"
            )
            print(
                f"        drift {r3.replan_steps}  kappa {r4.replan_steps}  "
                f"union {r5.replan_steps}"
            )
    return results


if __name__ == "__main__":
    emit_table(CONSTANT_WORLD)

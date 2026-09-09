"""Phase 4c step 0: oracle-field headroom check.

The question this answers is whether 4c is worth building AT ALL. Put the KNOWN
field into the planner -- no learning, no fitting -- and see whether even a
perfect model beats the scalar baseline. If the oracle cannot win on any
non-saturated metric, the scenarios or the metrics are the problem, and no
amount of learning fixes that. Better to know for the price of this script than
after a full collect-and-fit cycle.

WHY THE OBVIOUS METRIC IS A TRAP. goal_tolerance = 3.0, and both closed loops
terminate the instant hypot(tip, goal) < 3.0, so every successful run is
censored into [0, 3) and reports ~3.0. The field-world `fixed` numbers are
measuring that the loop STOPPED, not how well the model tracks. A perfect field
cannot beat the tolerance floor there, so measuring the field's value on
closed-loop endpoint error risks a FALSE NEGATIVE. Hence the headline is
OPEN-LOOP endpoint error (no drift crutch, so the model's contribution is
visible and attributable), with closed-loop reported alongside AND labelled
censored.

CONDITIONS (world is always the real field; filters always scalar):
    B  baseline  -- planner believes a scalar 1/29 (kappa_field=None)
    A  oracle    -- planner believes the field, served through a LOOKUP TABLE
                    (kappa=1/29, kappa_field=LookupField(TissueField))

The planner change is a NeedleParams construction, nothing else: KinodynamicRRT
is field-native (extend rolls `step` forward through self.params, and
is_arc_free does the same), and nothing in its path reads params.kappa as a
scalar -- verified by grep before writing this (the only 1/params.kappa reads
are in dubins.py / RRT*, and kappa=0.0 is VanillaRRT's straight-line
integrator). The NeedleEKF guard stays satisfied because the filter is handed
scalar model_params, not the field.

THE CONFOUND (flagged in the plan review): a field-believing planner does not
merely track better, it PLANS differently -- it believes R=15 in the capsule
(more manoeuvrable) and R=34 in fat (less), so it can return a different path
before any execution. So a difference between A and B could be a different plan
rather than better tracking. This separates the two: plan geometry (length,
node count, whether the paths differ) vs plan-to-execution deviation (the
tracking question proper). Open-loop is where that separation is clean, because
execution is blind -- no replanning muddies which plan is being followed.

Run:  python scripts/four_c_oracle_headroom.py
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.benchmark.vanilla_tracker import crosstrack_distance
from needlesim.control.closed_loop import (
    ClosedLoopConfig,
    run_closed_loop,
    run_open_loop,
)
from needlesim.estimation.ekf import EKFConfig
from needlesim.models.tissue_field import TissueField
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

BELIEF = 1 / 29  # the scalar belief (thickness-weighted mean R over 150mm)
SEEDS = range(1, 6)
SCENARIOS = (CONSTRAINED_PASSAGE, OPEN)


# --- the lookup-table field (the plumbing the learned field will reuse) ------


class LookupField:
    """Precomputed kappa(y) over a fine y-grid, linearly interpolated, O(1) per
    call. Duck-types TissueField.kappa_at so it drops into
    NeedleParams.kappa_field and is consumed by `step` with no other change.

    Built here from the true TissueField; in 4c the same class wraps the
    learned field. `step` calls kappa_at thousands of times per plan and the
    real kappa_at does several math.exp per call, so tabulating once and
    interpolating keeps the model off the RK4 hot path. Values off the grid
    ends are clamped to the nearest tabulated value (the grid spans well beyond
    the workspace, so this only matters far outside it).

    A plain class, not a dataclass: it must stay hashable-by-identity so a
    frozen NeedleParams carrying it can still hash (a dataclass over the value
    array would not)."""

    def __init__(self, source, y_min: float, y_max: float, spacing: float) -> None:
        n = int(round((y_max - y_min) / spacing)) + 1
        ys = np.linspace(y_min, y_max, n)
        self._ks = [float(source.kappa_at(0.0, float(y))) for y in ys]
        self._y0 = float(ys[0])
        self._dy = float(ys[1] - ys[0])
        self._nm1 = n - 1
        self._k0 = self._ks[0]
        self._klast = self._ks[-1]

    def kappa_at(self, x: float, y: float) -> float:
        t = (y - self._y0) / self._dy
        if t <= 0.0:
            return self._k0
        if t >= self._nm1:
            return self._klast
        i = int(t)
        f = t - i
        return self._ks[i] * (1.0 - f) + self._ks[i + 1] * f


def verify_lookup(lut: LookupField, source: TissueField) -> float:
    """Max abs error of the lookup vs the real kappa_at, sampled DENSELY and
    OFF-GRID (0.013mm step, coprime-ish with the 0.02mm grid so samples do not
    land on nodes) across the whole insertion range. Returns the max error."""
    ys = np.arange(0.0, 150.0, 0.013)
    return max(
        abs(lut.kappa_at(0.0, float(y)) - source.kappa_at(0.0, float(y))) for y in ys
    )


# --- planners / metrics -----------------------------------------------------


def make_planner(env, params: NeedleParams, seed: int) -> KinodynamicRRT:
    """Same config as the baseline five-way; only `params` differs between the
    scalar (B) and field (A) conditions."""
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, params, cfg)


def path_length(path: list[State]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:]))


def plan_vs_exec(executed: list[State], plan: list[State]) -> tuple[float, float]:
    """Mean and max cross-track deviation of the executed trajectory from the
    planned polyline -- the tracking question proper."""
    if not executed or len(plan) < 2:
        return (float("nan"), float("nan"))
    devs = [crosstrack_distance(s, plan) for s in executed]
    return (statistics.mean(devs), max(devs))


# --- the run ----------------------------------------------------------------


def main():
    world_field = TissueField()  # the SIMULATOR's field -- real kappa_at
    lut = LookupField(world_field, y_min=-10.0, y_max=160.0, spacing=0.02)
    lut_err = verify_lookup(lut, world_field)
    print(
        f"Lookup table: {len(lut._ks)} nodes over y=[-10,160] @ 0.02mm; "
        f"max abs error vs kappa_at = {lut_err:.2e} 1/mm "
        f"(field kappa spans {min(lut._ks):.4f}-{max(lut._ks):.4f})."
    )
    if lut_err > 1e-4:
        print("  WARNING: lookup error exceeds 1e-4; refine the grid.")

    scalar_params = NeedleParams(kappa=BELIEF)
    field_params = NeedleParams(kappa=BELIEF, kappa_field=lut)
    true_params = NeedleParams(kappa=BELIEF, kappa_field=world_field)

    # collected[(scenario, seed)] = dict of the four runs
    collected = {}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            sg = dict(start=scenario.start, goal=scenario.goal)

            env = build_env(scenario)
            b_open = run_open_loop(
                make_planner(env, scalar_params, seed),
                env,
                true_params,
                scalar_params,
                **sg,
            )
            env = build_env(scenario)
            a_open = run_open_loop(
                make_planner(env, field_params, seed),
                env,
                true_params,
                scalar_params,
                **sg,
            )
            env = build_env(scenario)
            b_closed = run_closed_loop(
                make_planner(env, scalar_params, seed),
                env,
                true_params,
                scalar_params,
                ekf_config=EKFConfig(seed=seed),
                loop_config=ClosedLoopConfig(seed=seed),
                **sg,
            )
            env = build_env(scenario)
            a_closed = run_closed_loop(
                make_planner(env, field_params, seed),
                env,
                true_params,
                scalar_params,
                ekf_config=EKFConfig(seed=seed),
                loop_config=ClosedLoopConfig(seed=seed),
                **sg,
            )
            collected[(scenario.name, seed)] = dict(
                b_open=b_open, a_open=a_open, b_closed=b_closed, a_closed=a_closed
            )

    _report(collected)


def _fmt_flags(r) -> str:
    return f"{'COLLIDE' if r.collided else 'clear':>7} {'OOB' if r.left_bounds else 'in':>3}"


def _report(collected):
    for scenario in SCENARIOS:
        name = scenario.name
        rows = [collected[(name, s)] for s in SEEDS]

        print(f"\n\n########## {name} ##########")

        # 1. OPEN-LOOP ENDPOINT -- the headline, NOT censored for B.
        print("\n--- 1. OPEN-LOOP endpoint error [mm] (HEADLINE; uncensored for B) ---")
        print(f"{'seed':>5} {'B scalar':>9} {'A oracle':>9} {'improvement':>12}")
        bo, ao = [], []
        for s, r in zip(SEEDS, rows):
            b = r["b_open"].final_error_mm
            a = r["a_open"].final_error_mm
            bo.append(b)
            ao.append(a)
            print(f"{s:>5} {b:>9.1f} {a:>9.1f} {b - a:>11.1f}")
        print(
            f"{'median':>5} {statistics.median(bo):>9.1f} "
            f"{statistics.median(ao):>9.1f} "
            f"{statistics.median(bo) - statistics.median(ao):>11.1f}"
        )

        # 2. OPEN-LOOP feasibility -- non-saturated, capsule-relevant.
        print("\n--- 2. OPEN-LOOP feasibility (collision / out-of-bounds) ---")
        print(f"{'seed':>5} {'B scalar':>15} {'A oracle':>15}")
        for s, r in zip(SEEDS, rows):
            print(f"{s:>5} {_fmt_flags(r['b_open']):>15} {_fmt_flags(r['a_open']):>15}")

        # 3. CLOSED-LOOP endpoint -- CENSORED at goal_tolerance; read replans.
        print("\n--- 3. CLOSED-LOOP endpoint [mm] (CENSORED at 3.0; read replans) ---")
        print(
            f"{'seed':>5} {'B err':>7} {'A err':>7} {'B repl':>7} {'A repl':>7} "
            f"{'B reason':>16} {'A reason':>16}"
        )
        b_repl, a_repl = [], []
        for s, r in zip(SEEDS, rows):
            b, a = r["b_closed"], r["a_closed"]
            b_repl.append(len(b.replan_steps))
            a_repl.append(len(a.replan_steps))
            print(
                f"{s:>5} {b.final_error_mm:>7.1f} {a.final_error_mm:>7.1f} "
                f"{len(b.replan_steps):>7} {len(a.replan_steps):>7} "
                f"{b.termination_reason:>16} {a.termination_reason:>16}"
            )
        print(
            f"{'total':>5} {'':>7} {'':>7} {sum(b_repl):>7} {sum(a_repl):>7} "
            f"{'replans':>16}"
        )

        # 4. CONFOUND -- planning vs tracking (open-loop).
        print("\n--- 4. CONFOUND: planning vs tracking (open-loop) ---")
        print(
            f"{'seed':>5} {'B len':>7} {'A len':>7} {'B nds':>6} {'A nds':>6} "
            f"{'differ?':>8} {'B dev mean/max':>16} {'A dev mean/max':>16}"
        )
        for s, r in zip(SEEDS, rows):
            bp = r["b_open"].plans[0]
            ap = r["a_open"].plans[0]
            blen, alen = path_length(bp), path_length(ap)
            differ = abs(blen - alen) > 1.0 or len(bp) != len(ap)
            bdm, bdx = plan_vs_exec(r["b_open"].executed_states, bp)
            adm, adx = plan_vs_exec(r["a_open"].executed_states, ap)
            print(
                f"{s:>5} {blen:>7.1f} {alen:>7.1f} {len(bp):>6} {len(ap):>6} "
                f"{('yes' if differ else 'no'):>8} "
                f"{bdm:>7.1f}/{bdx:<8.1f} {adm:>7.1f}/{adx:<8.1f}"
            )


if __name__ == "__main__":
    main()

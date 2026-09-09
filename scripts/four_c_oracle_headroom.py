"""Phase 4c step 0: oracle-field headroom check + capsule ablation.

The question step 0 answers is whether 4c is worth building AT ALL. Put the
KNOWN field into the planner -- no learning, no fitting -- and see whether even
a perfect model beats the scalar baseline. If the oracle cannot win on any
non-saturated metric, the scenarios or the metrics are the problem, and no
amount of learning fixes that.

WHY THE OBVIOUS METRIC IS A TRAP. goal_tolerance = 3.0, and both closed loops
terminate the instant hypot(tip, goal) < 3.0, so every successful run is
censored into [0, 3) and reports ~3.0. A perfect field cannot beat the
tolerance floor there, so measuring the field's value on closed-loop endpoint
risks a FALSE NEGATIVE. Hence the headline is OPEN-LOOP endpoint error (no
drift crutch, so the model's contribution is visible and attributable), plus
the non-saturated feasibility metric (collision / out-of-bounds), with
closed-loop reported alongside AND labelled censored.

CONDITIONS (world is always the real field; filters always scalar). Each is a
planner belief, served through a lookup table (the plumbing 4c reuses):
    B  scalar          -- kappa 1/29 everywhere (kappa_field=None); the baseline
    A  oracle          -- the true field everywhere; the ceiling
    C  capsule-blanked -- true field EXCEPT the capsule held at 1/29. Isolates
                          the COST of not knowing the capsule given you know
                          everything else -- i.e. the realistic 4c failure mode,
                          because the 5mm capsule is the one under-observed layer
                          (~1 measurement per insertion). A vs C is the headline
                          ablation; C vs B is how much of the oracle win survives
                          WITHOUT the capsule.
    D  capsule-only    -- 1/29 everywhere EXCEPT the true capsule. The mirror:
                          the GAIN from knowing only the capsule. (A-C) and
                          (D-B) do not sum to the oracle gap -- the difference is
                          layer interaction, which is why both bounds are run.

INTERPRETATION GUARDRAILS (from the plan review):
  - Judge the capsule's share on FEASIBILITY (collisions / OOB), not just
    endpoint mm: a small mm-share can still be the whole difference between a
    collision and a clear pass, and the collision is the clinically relevant
    outcome. So the verdict is "does blanking the capsule REINTRODUCE the
    collisions the full oracle avoided", not "how many mm does it restore".
  - Report per-scenario, never pooled. The capsule's share is scenario-
    dependent -- in constrained_passage the capsule (y65-70) sits directly below
    the critical wall (y72-78), so expect a large share there.
  - Treat the measured share as a LOWER bound on the capsule's importance: these
    scenarios were designed pre-field, and a capsule-decision-relevant held-out
    test (step 1) would stress it more. A large share here is decisive (must
    solve capsule observation); a small share here is weak evidence to relax.

The planner change is a NeedleParams construction, nothing else: KinodynamicRRT
is field-native (extend and is_arc_free roll `step` forward through
self.params), and nothing in its path reads params.kappa as a scalar. The
NeedleEKF guard stays satisfied because the filter is handed scalar
model_params, not the field.

Run:  python scripts/four_c_oracle_headroom.py
"""

from __future__ import annotations

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
from needlesim.models.tissue_field import TissueField, TissueLayer
from needlesim.models.unicycle_needle import NeedleParams, State
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

BELIEF = 1 / 29  # the scalar belief (thickness-weighted mean R over 150mm)
SEEDS = range(1, 6)
SCENARIOS = (CONSTRAINED_PASSAGE, OPEN)
CAPSULE = "capsule"


# --- the lookup-table field (the plumbing the learned field will reuse) ------


class LookupField:
    """Precomputed kappa(y) over a fine y-grid, linearly interpolated, O(1) per
    call. Duck-types TissueField.kappa_at so it drops into
    NeedleParams.kappa_field and is consumed by `step` with no other change.

    `step` calls kappa_at thousands of times per plan and the real kappa_at does
    several math.exp per call, so tabulating once and interpolating keeps the
    model off the RK4 hot path. Off-grid ends clamp to the nearest node (the
    grid spans well beyond the workspace). A plain class, not a dataclass, so it
    stays hashable-by-identity -- a frozen NeedleParams carrying it must hash."""

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


def verify_lookup(lut: LookupField, source) -> float:
    """Max abs error of the lookup vs the real kappa_at, sampled DENSELY and
    OFF-GRID (0.013mm step, coprime-ish with the 0.02mm grid) across the whole
    insertion range."""
    ys = np.arange(0.0, 150.0, 0.013)
    return max(
        abs(lut.kappa_at(0.0, float(y)) - source.kappa_at(0.0, float(y))) for y in ys
    )


# --- the ablated fields -----------------------------------------------------


def with_layer_kappa(field: TissueField, layer_name: str, kappa: float) -> TissueField:
    """Copy `field` with one named layer's kappa replaced. Everything else
    (edges, transition width, the other layers) is left true."""
    layers = tuple(
        TissueLayer(
            kappa if lyr.name == layer_name else lyr.kappa, lyr.upper_edge_mm, lyr.name
        )
        for lyr in field.layers
    )
    return TissueField(layers=layers, transition_mm=field.transition_mm)


def all_but_layer_kappa(
    field: TissueField, layer_name: str, kappa: float
) -> TissueField:
    """Copy `field` with EVERY layer's kappa replaced by `kappa` EXCEPT the
    named one, which keeps its true value. The mirror of with_layer_kappa."""
    layers = tuple(
        TissueLayer(
            lyr.kappa if lyr.name == layer_name else kappa, lyr.upper_edge_mm, lyr.name
        )
        for lyr in field.layers
    )
    return TissueField(layers=layers, transition_mm=field.transition_mm)


# --- planners / metrics -----------------------------------------------------


def make_planner(env, params: NeedleParams, seed: int) -> KinodynamicRRT:
    """Same config as the baseline five-way; only `params` differs by condition."""
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, params, cfg)


def plan_vs_exec_max(executed: list[State], plan: list[State]) -> float:
    """Max cross-track deviation of the executed trajectory from the planned
    polyline -- the tracking question proper."""
    if not executed or len(plan) < 2:
        return float("nan")
    return max(crosstrack_distance(s, plan) for s in executed)


# --- the run ----------------------------------------------------------------


def main():
    world_field = TissueField()  # the SIMULATOR's field -- real kappa_at
    scalar_params = NeedleParams(kappa=BELIEF)

    def field_params(source) -> NeedleParams:
        lut = LookupField(source, y_min=-10.0, y_max=160.0, spacing=0.02)
        err = verify_lookup(lut, source)
        if err > 1e-4:
            raise SystemExit(f"lookup error {err:.2e} exceeds 1e-4; refine grid")
        return NeedleParams(kappa=BELIEF, kappa_field=lut), err

    oracle_p, oracle_err = field_params(world_field)
    blanked_p, blanked_err = field_params(
        with_layer_kappa(world_field, CAPSULE, BELIEF)
    )
    only_p, only_err = field_params(all_but_layer_kappa(world_field, CAPSULE, BELIEF))
    print(
        "Lookup tables @ 0.02mm, max abs error vs kappa_at: "
        f"oracle {oracle_err:.1e}, capsule-blanked {blanked_err:.1e}, "
        f"capsule-only {only_err:.1e} (all << 1e-4)."
    )

    # Order matters only for display; each run is deterministic from its own
    # seed+params+env, so B and A reproduce the step-0 numbers regardless.
    conditions = [
        ("B scalar", scalar_params),
        ("A oracle", oracle_p),
        ("C caps-blank", blanked_p),
        ("D caps-only", only_p),
    ]

    true_params = NeedleParams(kappa=BELIEF, kappa_field=world_field)

    # collected[(scenario, seed, label)] = {"open": result, "closed": result}
    collected = {}
    for scenario in SCENARIOS:
        for seed in SEEDS:
            sg = dict(start=scenario.start, goal=scenario.goal)
            for label, params in conditions:
                env = build_env(scenario)
                r_open = run_open_loop(
                    make_planner(env, params, seed),
                    env,
                    true_params,
                    scalar_params,
                    **sg,
                )
                env = build_env(scenario)
                r_closed = run_closed_loop(
                    make_planner(env, params, seed),
                    env,
                    true_params,
                    scalar_params,
                    ekf_config=EKFConfig(seed=seed),
                    loop_config=ClosedLoopConfig(seed=seed),
                    **sg,
                )
                collected[(scenario.name, seed, label)] = {
                    "open": r_open,
                    "closed": r_closed,
                }

    _report(collected, [lbl for lbl, _ in conditions])


def _report(collected, labels):
    for scenario in SCENARIOS:
        name = scenario.name
        print(f"\n\n{'#' * 30} {name} {'#' * 30}")

        def col(seed, label, loop, attr):
            return getattr(collected[(name, seed, label)][loop], attr)

        # 1. OPEN-LOOP endpoint per condition, per seed.
        print("\n--- 1. OPEN-LOOP endpoint error [mm] per condition ---")
        print("seed " + "".join(f"{lbl:>13}" for lbl in labels))
        med = {}
        for seed in SEEDS:
            vals = [col(seed, lbl, "open", "final_error_mm") for lbl in labels]
            print(f"{seed:>4} " + "".join(f"{v:>13.1f}" for v in vals))
        print(
            "med  "
            + "".join(
                f"{statistics.median([col(s, lbl, 'open', 'final_error_mm') for s in SEEDS]):>13.1f}"
                for lbl in labels
            )
        )
        for lbl in labels:
            med[lbl] = statistics.median(
                [col(s, lbl, "open", "final_error_mm") for s in SEEDS]
            )

        # 2. FEASIBILITY per condition (the decision metric) -- open AND closed.
        print("\n--- 2. FEASIBILITY over 5 seeds (collisions / out-of-bounds) ---")
        print(
            f"{'condition':>13} {'open coll':>10} {'open OOB':>9} "
            f"{'closed coll':>12} {'closed OOB':>11}"
        )
        for lbl in labels:
            oc = sum(col(s, lbl, "open", "collided") for s in SEEDS)
            oo = sum(col(s, lbl, "open", "left_bounds") for s in SEEDS)
            cc = sum(col(s, lbl, "closed", "collided") for s in SEEDS)
            co = sum(col(s, lbl, "closed", "left_bounds") for s in SEEDS)
            print(f"{lbl:>13} {oc:>8}/5 {oo:>7}/5 {cc:>10}/5 {co:>9}/5")

        # 3. TRACKING (open-loop plan-vs-exec max dev, median over seeds) and
        #    CLOSED-LOOP replans (censored endpoint, so read the replans).
        print("\n--- 3. TRACKING (open dev max, median) & CLOSED-LOOP replans ---")
        print(f"{'condition':>13} {'open devmax':>12} {'closed replans (total)':>24}")
        for lbl in labels:
            devs = [
                plan_vs_exec_max(
                    collected[(name, s, lbl)]["open"].executed_states,
                    collected[(name, s, lbl)]["open"].plans[0],
                )
                for s in SEEDS
            ]
            repl = sum(len(col(s, lbl, "closed", "replan_steps")) for s in SEEDS)
            print(f"{lbl:>13} {statistics.median(devs):>11.1f} {repl:>24}")

        # 4. THE ABLATION VERDICT.
        b, a, c, d = (
            med["B scalar"],
            med["A oracle"],
            med["C caps-blank"],
            med["D caps-only"],
        )
        oracle_win = b - a
        retained = b - c
        capsule_cost = c - a
        a_coll = sum(col(s, "A oracle", "open", "collided") for s in SEEDS)
        a_oob = sum(col(s, "A oracle", "open", "left_bounds") for s in SEEDS)
        c_coll = sum(col(s, "C caps-blank", "open", "collided") for s in SEEDS)
        c_oob = sum(col(s, "C caps-blank", "open", "left_bounds") for s in SEEDS)
        frac = retained / oracle_win if oracle_win > 1e-9 else float("nan")
        print("\n--- 4. CAPSULE ABLATION VERDICT (open-loop medians, mm) ---")
        print(f"  oracle win over baseline (B-A)      : {oracle_win:6.1f}")
        print(
            f"  win retained WITHOUT capsule (B-C)  : {retained:6.1f}  ({frac * 100:.0f}%)"
        )
        print(f"  extra error from blanking capsule (C-A): {capsule_cost:6.1f}")
        print(f"  capsule-only gain (B-D)             : {b - d:6.1f}")
        print(
            f"  feasibility: full oracle {a_coll}/5 coll {a_oob}/5 OOB; "
            f"capsule-blanked {c_coll}/5 coll {c_oob}/5 OOB "
            f"-> blanking {'REINTRODUCES' if (c_coll > a_coll or c_oob > a_oob) else 'does NOT reintroduce'} failures"
        )


if __name__ == "__main__":
    main()

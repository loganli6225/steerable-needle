"""Phase 4c Gate 3 task 2: the held-out verdict.

Runs the frozen conservative-serving mechanism against the HELD-OUT
misspecifications (a) sharp capsule and (b) lateral x-tilt -- with (c) Gaussian
bump as the develop-against reference -- and decides go/no-go on the
PRE-REGISTERED criterion below. Parameters were frozen before this ran
(docs/four_c_conservative_serving_spec.md); this script does not tune them.

PRE-REGISTERED JOB-2 PASS CRITERION (fixed before any (a)/(b) run):
  Per held-out world, on constrained_passage, 5 seeds, with baseline (scalar
  1/29) and oracle (true world field) measured in the SAME runs as floor/ceiling:
    PRIMARY (feasibility, deciding): PASS iff served field has 0 collisions AND
      0 out-of-bounds across all 5 seeds (matches the oracle). Any collision/OOB
      the oracle avoids is a FAIL -- conservative serving is a safety fail-safe,
      colliding breaks its one promise.
    SECONDARY (endpoint, for a CLEAN pass): median open-loop endpoint closes
      >=50% of the baseline->oracle gap. Feasibility-pass but endpoint <50% =
      "safe but imprecise" (qualified). Feasibility-fail = fail regardless.
    CONTRAST (open, specificity): served ~ baseline (median endpoint within 2mm,
      same feasibility). If open SEPARATES, the primary result is flagged
      suspect, not accepted.

TWO CARRIED CAVEATS (from task-1 review):
  - whiteness is computed on the augmented pool at N>=130 (a small-N pool can
    clear both gates by chance; large N keeps the null |r| ~0.167 < the 0.30
    floor).
  - the served-field veto is validated for BROAD-in-y structure only, so a
    "quiet" result on (b) means "no broad lateral structure detected", not "no
    lateral structure".

METRIC IS OPEN-LOOP (endpoint + collision): closed loops censor at
goal_tolerance=3.0, which would hide the field's contribution (the step-0
saturation lesson). Baseline/served/oracle differ ONLY in the planner's field;
the simulator always steps the TRUE (misspecified) world field, preserving the
true/model split.

Resumable: per-world served-field FITS (the expensive, kill-risk step) cache to
SCRATCH; the planner runs are cheap and run in one stage. Re-invoke until
'report' prints.

Scripts only; reuses task-1 serving (four_c_serving), the Gate-2 fitter
(four_c_gates), run_open_loop, and the benchmark scenarios. Nothing under src/.
Run (repeatedly until it reports):  python scripts/four_c_gate3_verdict.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from four_c_gates import TRUE, _capsule_dwell_arcs, _capsule_knots, fit_y, path_sets
from four_c_serving import (
    YGRID,
    GaussianBumpField,
    XTiltField,
    _make_dataset_under,
    build_served_field,
    multi_depth_arcs,
    sharp_capsule_field,
)

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, OPEN, build_env
from needlesim.control.closed_loop import run_open_loop
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

SCRATCH = (
    "/private/tmp/claude-501/-Users-LoganLi-steerable-needle/"
    "71979fe2-8faf-48c0-9647-a7605668fa55/scratchpad"
)
BELIEF = 1.0 / 29.0
SEEDS = range(1, 6)
POOL_SEED = 2
CAP_MID = 67.5

# Held-out (a)/(b) and develop-against (c). Each is a TRUE world field that
# drives BOTH the simulator and the oracle planner; the served field is fitted
# from characterization data generated under it.
WORLDS = {
    "a_sharp": sharp_capsule_field(),
    "b_xtilt": XTiltField(TRUE),
    "c_gauss": GaussianBumpField(),
}
SCENARIOS = {"constrained_passage": CONSTRAINED_PASSAGE, "open": OPEN}


def aug_pool():
    """Augmented characterization pool, N=130 (>=130 per the carried caveat)."""
    return (
        path_sets(60, POOL_SEED)["revisit"]
        + _capsule_dwell_arcs(30, POOL_SEED)
        + multi_depth_arcs(40, POOL_SEED)
    )


# --- resumable fit stage (per world; the expensive, kill-risk step) ----------


def _fit_path(world):
    return f"{SCRATCH}/g3_fit_{world}.npz"


def fit_world(world):
    """Fit kappa(y) from characterization data under `world`, build the served
    field, cache k_eff(YGRID) + the job-1 capsule recovery."""
    field = WORLDS[world]
    coords = _capsule_knots()
    data = _make_dataset_under(aug_pool(), field, seed=POOL_SEED)
    k_hat, _ = fit_y(coords, data)
    served = build_served_field(coords, k_hat, data)
    cap = (YGRID >= 65.0) & (YGRID <= 70.0)
    np.savez(
        _fit_path(world),
        k_eff=served._k,
        cap_served=float(np.mean(served._k[cap])),
        cap_true=float(field.kappa_at(0.0, CAP_MID)),
    )


class _ServedFromGrid:
    """Reconstruct a served field's kappa_at from the cached k_eff grid."""

    def __init__(self, k_eff):
        self._k = k_eff

    def kappa_at(self, x, y):
        return float(np.interp(y, YGRID, self._k))


# --- planner-field builders per condition ------------------------------------


def _planner_params(condition, world):
    if condition == "baseline":  # scalar belief, no field
        return NeedleParams(kappa=BELIEF)
    if condition == "oracle":  # the TRUE (misspecified) world field
        return NeedleParams(kappa=BELIEF, kappa_field=WORLDS[world])
    if condition == "served":  # the learned kappa(y), from cache
        k_eff = np.load(_fit_path(world))["k_eff"]
        return NeedleParams(kappa=BELIEF, kappa_field=_ServedFromGrid(k_eff))
    raise ValueError(condition)


def _make_planner(env, params, seed):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, params, cfg)


# --- cheap run stage (all planner runs) --------------------------------------


def _runs_path():
    return f"{SCRATCH}/g3_runs.json"


def run_all():
    """Open-loop over every (scenario, world, condition, seed). Simulator steps
    the TRUE world field; planner uses the condition's field."""
    rows = []
    for sc_name, scenario in SCENARIOS.items():
        for world in WORLDS:
            true_params = NeedleParams(kappa=BELIEF, kappa_field=WORLDS[world])
            for condition in ("baseline", "served", "oracle"):
                pp = _planner_params(condition, world)
                for seed in SEEDS:
                    env = build_env(scenario)
                    r = run_open_loop(
                        _make_planner(env, pp, seed),
                        env,
                        true_params,
                        pp,
                        start=scenario.start,
                        goal=scenario.goal,
                    )
                    rows.append(
                        dict(
                            scenario=sc_name,
                            world=world,
                            condition=condition,
                            seed=seed,
                            collided=bool(r.collided),
                            left_bounds=bool(r.left_bounds),
                            final_error_mm=float(r.final_error_mm),
                            reached_goal=bool(r.reached_goal),
                        )
                    )
    json.dump(rows, open(_runs_path(), "w"))


# --- report (apply the pre-registered criterion) -----------------------------


def _med(xs):
    return float(np.median(xs)) if xs else float("nan")


def report():
    rows = json.load(open(_runs_path()))

    def sel(sc, world, cond, key):
        return [
            r[key]
            for r in rows
            if r["scenario"] == sc and r["world"] == world and r["condition"] == cond
        ]

    def fails(sc, world, cond):
        c = sum(sel(sc, world, cond, "collided"))
        o = sum(sel(sc, world, cond, "left_bounds"))
        return c, o

    print("\n" + "=" * 78)
    print("PHASE 4c GATE 3 -- HELD-OUT VERDICT  [pre-registered criterion]")
    print("=" * 78)

    # job-1 diagnostic
    print("\n--- JOB 1 (diagnostic): capsule recovery per world ---")
    for world in WORLDS:
        if not os.path.exists(_fit_path(world)):
            continue
        d = np.load(_fit_path(world))
        cs, ct = float(d["cap_served"]), float(d["cap_true"])
        print(
            f"  {world:10s} served capsule R {1/cs:5.1f}mm  true R {1/ct:5.1f}mm  "
            f"({100*abs(cs-ct)/ct:+.0f}% kappa)"
        )

    # job-2 primary + secondary on constrained_passage; contrast on open
    for world in ("a_sharp", "b_xtilt"):  # held-out verdict worlds
        print(f"\n--- JOB 2 world={world} ---")
        for sc in ("constrained_passage", "open"):
            b_coll, b_oob = fails(sc, world, "baseline")
            s_coll, s_oob = fails(sc, world, "served")
            o_coll, o_oob = fails(sc, world, "oracle")
            b_e = _med(sel(sc, world, "baseline", "final_error_mm"))
            s_e = _med(sel(sc, world, "served", "final_error_mm"))
            o_e = _med(sel(sc, world, "oracle", "final_error_mm"))
            gap = b_e - o_e
            closed = (b_e - s_e) / gap * 100 if abs(gap) > 1e-9 else float("nan")
            print(f"  [{sc}]")
            print(
                f"    feasibility (coll/OOB of 5): baseline {b_coll}/{b_oob}  "
                f"served {s_coll}/{s_oob}  oracle {o_coll}/{o_oob}"
            )
            print(
                f"    endpoint med [mm]: baseline {b_e:.1f}  served {s_e:.1f}  "
                f"oracle {o_e:.1f}  (served closes {closed:.0f}% of gap)"
            )
            if sc == "constrained_passage":
                feas_pass = s_coll == 0 and s_oob == 0
                clean = feas_pass and (closed >= 50.0)
                verdict = (
                    "CLEAN PASS"
                    if clean
                    else "PASS (safe but imprecise)" if feas_pass else "FAIL"
                )
                print(f"    VERDICT (primary): {verdict}")
            else:  # open contrast
                sep = abs(s_e - b_e) > 2.0 or (s_coll, s_oob) != (b_coll, b_oob)
                print(
                    f"    CONTRAST: {'SEPARATES -> primary SUSPECT' if sep else 'served ~ baseline (specific, OK)'}"
                )


def main():
    for world in WORLDS:
        if not os.path.exists(_fit_path(world)):
            print(f"fitting {world} ...")
            fit_world(world)
            print(f"  cached {world}")
            return
    if not os.path.exists(_runs_path()):
        print("running all planner runs ...")
        run_all()
        print("  cached runs")
        return
    report()


if __name__ == "__main__":
    main()

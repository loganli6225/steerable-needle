"""Phase 4c Gate 3, addendum to the (a) sharp-capsule verdict -- DIAGNOSTIC ONLY.

The pre-registered verdict is FAIL for (a) (served 1/5 collisions vs the oracle's
0/5 on constrained_passage) and it STANDS regardless of what this shows. This
addendum does NOT re-open it and does NOT tune any frozen parameter (m, k_max,
whiteness gate are untouched). It answers one diagnostic question the writeup
needs: is that single collision a PROPERTY of the mechanism at a ~1mm capsule
sharpness, or an ARTIFACT of one frozen characterization draw / a thin margin?

The clean test: re-fit the (a) served field at DIFFERENT POOL SEEDS (different
random characterization arcs), everything else identical, and run the same
constrained_passage planner seeds (1-5). Changing the pool seed changes zero
parameters -- only which arcs were collected. If served collisions stay ~1/5
across draws, the boundary is a mechanism property (~1mm sharpness); if they
vary to 0/5, the single collision was draw-specific / a thin margin.

Report whatever it shows. A clean re-run does NOT convert the verdict to a pass.

Resumable: one pool-seed fit per invocation (the kill-risk step) caches to
SCRATCH. Re-invoke until it reports.
Run:  python scripts/four_c_gate3_addendum_a.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from four_c_gates import _capsule_dwell_arcs, _capsule_knots, fit_y, path_sets
from four_c_serving import (
    YGRID,
    _make_dataset_under,
    build_served_field,
    multi_depth_arcs,
    sharp_capsule_field,
)

from needlesim.benchmark.scenarios import CONSTRAINED_PASSAGE, build_env
from needlesim.control.closed_loop import run_open_loop
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig

SCRATCH = (
    "/private/tmp/claude-501/-Users-LoganLi-steerable-needle/"
    "71979fe2-8faf-48c0-9647-a7605668fa55/scratchpad"
)
BELIEF = 1.0 / 29.0
POOL_SEEDS = [2, 3, 4]  # 2 reproduces the verdict; 3,4 are the robustness draws
PLANNER_SEEDS = range(1, 6)
WORLD = sharp_capsule_field()  # (a), fixed; only the characterization draw varies


class _Served:
    def __init__(self, k_eff):
        self._k = k_eff

    def kappa_at(self, x, y):
        return float(np.interp(y, YGRID, self._k))


def _pool(ps):
    return (
        path_sets(60, ps)["revisit"]
        + _capsule_dwell_arcs(30, ps)
        + multi_depth_arcs(40, ps)
    )


def _path(ps):
    return f"{SCRATCH}/g3add_a_seed{ps}.json"


def _planner(env, params, seed):
    cfg = RRTConfig(
        max_iterations=20000,
        goal_tolerance=3.0,
        step_dt=0.05,
        edge_velocity=5.0,
        margin=2.0,
        seed=seed,
    )
    return KinodynamicRRT(env, params, cfg)


def do_pool_seed(ps):
    coords = _capsule_knots()
    data = _make_dataset_under(_pool(ps), WORLD, seed=ps)
    k_hat, _ = fit_y(coords, data)
    served = build_served_field(coords, k_hat, data)
    pp = NeedleParams(kappa=BELIEF, kappa_field=_Served(served._k))
    true_params = NeedleParams(kappa=BELIEF, kappa_field=WORLD)
    coll = oob = 0
    errs = []
    for seed in PLANNER_SEEDS:
        env = build_env(CONSTRAINED_PASSAGE)
        r = run_open_loop(
            _planner(env, pp, seed),
            env,
            true_params,
            pp,
            start=CONSTRAINED_PASSAGE.start,
            goal=CONSTRAINED_PASSAGE.goal,
        )
        coll += int(r.collided)
        oob += int(r.left_bounds)
        errs.append(float(r.final_error_mm))
    json.dump(
        dict(
            pool_seed=ps, collisions=coll, oob=oob, endpoint_med=float(np.median(errs))
        ),
        open(_path(ps), "w"),
    )


def main():
    for ps in POOL_SEEDS:
        if not os.path.exists(_path(ps)):
            print(f"fitting+running (a) at pool_seed={ps} ...")
            do_pool_seed(ps)
            print(f"  cached pool_seed={ps}")
            return
    print(
        "\n=== (a) sharp-capsule collision robustness across characterization draws ==="
    )
    print("DIAGNOSTIC ONLY -- the pre-registered verdict FAIL stands regardless.\n")
    print(
        f"  {'pool_seed':>10} {'collisions/5':>13} {'oob/5':>7} {'endpoint_med_mm':>16}"
    )
    vals = []
    for ps in POOL_SEEDS:
        d = json.load(open(_path(ps)))
        vals.append(d["collisions"])
        print(
            f"  {ps:>10} {d['collisions']:>13} {d['oob']:>7} {d['endpoint_med']:>16.1f}"
        )
    lo, hi = min(vals), max(vals)
    if hi == 0:
        verdict = "draw-specific: seed-2's 1 collision did not recur -> thin margin, not a stable property"
    elif lo >= 1:
        verdict = "stable: served collides on every draw -> mechanism property at ~1mm sharpness"
    else:
        verdict = f"mixed ({lo}-{hi}/5): borderline; the collision is draw-sensitive, near the boundary"
    print(f"\n  READING: {verdict}")
    print(
        "  (Verdict unchanged: (a) FAILS the 0-collision bar; this only characterizes WHY.)"
    )


if __name__ == "__main__":
    main()

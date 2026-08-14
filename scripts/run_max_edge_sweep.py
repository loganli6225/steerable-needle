"""Experiment: does a max-edge-length (steering-horizon) constraint help RRT*,
or starve it?

The scaled secondary experiment showed RRT* paths 4.4-5.7x longer than
KinodynamicRRT, and the figures showed why: the path is a chain of near-full
turning circles (~314mm at R=1/kappa=50mm) that reconcile arbitrary sampled
headings. This sweep tests whether forbidding edges longer than a threshold X
(below the loop signature) forces the tree onto short edges and lowers cost
(HELPS), or makes arbitrary-heading node pairs simply unconnectable so the tree
starves (STARVES).

The distinguishing diagnostic is the choose_parent rejection rate -- the
fraction of choose_parent calls returning None. It is measured WITHOUT touching
rrt_star.py, via a thin counting subclass here. If cost falls while rejection
stays moderate, the constraint works; if rejection climbs toward the ~98.5%
seen at primary scale, it is starvation by another route.

Configuration is the settled secondary-experiment one: scaled `open`
(500mm, gamma=133, budget 5000). Sweep max_edge_length in {inf,400,250,150,100},
5 seeds each; settings below any all-zero-success setting are skipped. The best
surviving setting is then rerun on scaled `constrained_passage` to check the
finding is not specific to `open`.

Separate CSV: experiments/results/max_edge_sweep_raw.csv (gitignored). NOT pooled
with either benchmark.

Run:  python scripts/run_max_edge_sweep.py
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

from needlesim.benchmark.harness_scaled import (
    RRTSTAR_BUDGET,
    RRTSTAR_GAMMA,
    RRTSTAR_MAX_RADIUS,
    rrtstar_endpoint_error_mm,
    rrtstar_heading_discontinuity,
)
from needlesim.benchmark.scaled_scenarios import (
    SCALED_COMMON_CONDITIONS,
    SCALED_SCENARIOS,
)
from needlesim.benchmark.scenarios import build_env
from needlesim.models.unicycle_needle import NeedleParams
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig

KAPPA = SCALED_COMMON_CONDITIONS["kappa"]
SEEDS = 5
# inf plus four finite thresholds spanning above and below the ~314mm loop.
THRESHOLDS = [float("inf"), 400.0, 250.0, 150.0, 100.0]
DEFAULT_CSV_PATH = Path("experiments/results/max_edge_sweep_raw.csv")

CSV_COLUMNS = [
    "scenario_name",
    "max_edge_length",
    "seed",
    "success",
    "n_nodes",
    "path_cost_mm",
    "endpoint_error_mm",
    "heading_disc_max_rad",
    "wall_time_s",
    "choose_parent_calls",
    "choose_parent_none",
    "choose_parent_none_rate",
]


class CountingRRTStar(RRTStar):
    """RRT* that tallies choose_parent calls and None returns, to measure the
    steering-horizon rejection rate. Overrides ONLY the counting -- delegates the
    real logic to the parent -- so it changes no planner behaviour."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_calls = 0
        self.cp_none = 0

    def choose_parent(self, nodes, new_state, neighbourhood):
        self.cp_calls += 1
        result = super().choose_parent(nodes, new_state, neighbourhood)
        if result is None:
            self.cp_none += 1
        return result


def make_config(seed: int, max_edge_length: float) -> RRTStarConfig:
    return RRTStarConfig(
        max_iterations=RRTSTAR_BUDGET,
        gamma=RRTSTAR_GAMMA,
        max_radius=RRTSTAR_MAX_RADIUS,
        goal_tolerance=SCALED_COMMON_CONDITIONS["goal_tolerance"],
        step_dt=SCALED_COMMON_CONDITIONS["step_dt"],
        edge_velocity=SCALED_COMMON_CONDITIONS["edge_velocity"],
        margin=SCALED_COMMON_CONDITIONS["margin"],
        max_edge_length=max_edge_length,
        seed=seed,
    )


def run_one(scenario, max_edge_length: float, seed: int) -> dict:
    params = NeedleParams(kappa=KAPPA)
    cfg = make_config(seed, max_edge_length)
    planner = CountingRRTStar(build_env(scenario), params, cfg)

    t0 = time.perf_counter()
    result = planner.plan(scenario.start, scenario.goal)
    wall = time.perf_counter() - t0

    none_rate = planner.cp_none / planner.cp_calls if planner.cp_calls else 0.0
    row = dict(
        scenario_name=scenario.name,
        max_edge_length=max_edge_length,
        seed=seed,
        success=result.success,
        n_nodes=len(result.nodes),
        path_cost_mm=None,
        endpoint_error_mm=None,
        heading_disc_max_rad=None,
        wall_time_s=wall,
        choose_parent_calls=planner.cp_calls,
        choose_parent_none=planner.cp_none,
        choose_parent_none_rate=none_rate,
    )
    if result.success:
        hmax, _, _ = rrtstar_heading_discontinuity(result, params)
        row["path_cost_mm"] = result.best_cost
        row["endpoint_error_mm"] = rrtstar_endpoint_error_mm(
            result, scenario.start, scenario.goal, params
        )
        row["heading_disc_max_rad"] = hmax
    return row


@dataclass
class _Agg:
    n: int = 0
    ok: int = 0
    nodes: float = 0.0
    cost_sum: float = 0.0
    cost_n: int = 0
    time: float = 0.0
    none_rate: float = 0.0


def _summarise(rows_for_setting) -> _Agg:
    a = _Agg()
    for r in rows_for_setting:
        a.n += 1
        a.nodes += r["n_nodes"]
        a.time += r["wall_time_s"]
        a.none_rate += r["choose_parent_none_rate"]
        if r["success"]:
            a.ok += 1
            a.cost_sum += r["path_cost_mm"]
            a.cost_n += 1
    return a


def _print_summary(scenario_name, rows):
    print(f"\n=== summary: {scenario_name} ===")
    print(
        f"{'max_edge':>9} {'succ':>6} {'nodes':>8} {'cost_mm':>10} "
        f"{'time_s':>8} {'cp_none_rate':>13}"
    )
    by_thr = {}
    for r in rows:
        by_thr.setdefault(r["max_edge_length"], []).append(r)
    for thr in sorted(by_thr, key=lambda x: (math.isinf(x), -x)):
        a = _summarise(by_thr[thr])
        label = "inf" if math.isinf(thr) else f"{thr:.0f}"
        cost = f"{a.cost_sum / a.cost_n:.1f}" if a.cost_n else "--"
        print(
            f"{label:>9} {a.ok:>3}/{a.n:<2} {a.nodes / a.n:>8.0f} {cost:>10} "
            f"{a.time / a.n:>8.1f} {a.none_rate / a.n:>12.1%}"
        )


def _append_rows(writer, f, rows_out, new_rows):
    for row in new_rows:
        writer.writerow(row)
        rows_out.append(row)
    f.flush()


def main():
    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    open_scn = SCALED_SCENARIOS["open"]
    all_rows = []

    with DEFAULT_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        # --- the sweep on scaled open ---
        best_finite = None  # (thr, mean_cost) among finite settings with successes
        for thr in THRESHOLDS:
            label = "inf" if math.isinf(thr) else f"{thr:.0f}"
            setting_rows = []
            for seed in range(SEEDS):
                row = run_one(open_scn, thr, seed)
                setting_rows.append(row)
                flag = "ok " if row["success"] else "FAIL"
                print(
                    f"[open max_edge={label:>4} seed={seed}] {flag} "
                    f"nodes={row['n_nodes']:>5} "
                    f"cost={row['path_cost_mm'] if row['path_cost_mm'] else 0:>7.0f} "
                    f"cp_none={row['choose_parent_none_rate']:.1%} "
                    f"{row['wall_time_s']:.0f}s"
                )
            _append_rows(writer, f, all_rows, setting_rows)
            a = _summarise(setting_rows)
            if math.isfinite(thr) and a.cost_n:
                mean_cost = a.cost_sum / a.cost_n
                if best_finite is None or mean_cost < best_finite[1]:
                    best_finite = (thr, mean_cost)
            # Stop descending once a setting yields zero successes: everything
            # below it is more constrained and will only do worse.
            if a.ok == 0 and math.isfinite(thr):
                print(
                    f"  -> {label} yielded 0/{SEEDS} successes; "
                    "skipping all lower thresholds."
                )
                break

        _print_summary("open", [r for r in all_rows if r["scenario_name"] == "open"])

        # --- cross-check the best finite setting on constrained_passage ---
        if best_finite is not None:
            thr = best_finite[0]
            print(
                f"\nBest finite setting on open: max_edge={thr:.0f} "
                f"(mean cost {best_finite[1]:.0f}mm). "
                "Cross-checking on constrained_passage..."
            )
            cp_scn = SCALED_SCENARIOS["constrained_passage"]
            cp_rows = []
            for seed in range(SEEDS):
                row = run_one(cp_scn, thr, seed)
                cp_rows.append(row)
                flag = "ok " if row["success"] else "FAIL"
                print(
                    f"[constrained max_edge={thr:.0f} seed={seed}] {flag} "
                    f"nodes={row['n_nodes']:>5} "
                    f"cost={row['path_cost_mm'] if row['path_cost_mm'] else 0:>7.0f} "
                    f"cp_none={row['choose_parent_none_rate']:.1%} "
                    f"{row['wall_time_s']:.0f}s"
                )
            _append_rows(writer, f, all_rows, cp_rows)
            _print_summary("constrained_passage", cp_rows)
        else:
            print("\nNo finite setting produced successes on open; no cross-check.")

    print(f"\nDone. Raw rows -> {DEFAULT_CSV_PATH}")


if __name__ == "__main__":
    main()

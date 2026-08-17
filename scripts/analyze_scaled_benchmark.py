"""Read the SCALED benchmark CSV and print summary tables. NEVER re-runs planners.

All three planners are reported per scenario (unlike the primary analysis, which
splits illustrative hand-designed from statistical random -- the secondary grid
has only the four scaled scenarios). The table carries success rate and
mean +/- std OVER SUCCESSFUL RUNS for cost, time, iterations, endpoint error, and
heading discontinuity.

THE COMPARISON THAT MATTERS is printed separately: path cost, RRTStar vs
KinodynamicRRT, on scenarios where BOTH succeed -- as a percentage and with the
wall-time it cost. That is the number the whole experiment turns on.

Run:  python scripts/analyze_scaled_benchmark.py
"""

from __future__ import annotations

from collections import defaultdict

# Reuse the primary analysis helpers wholesale -- identical CSV schema.
from analyze_benchmark import _fnum, _mean_std, _success, _to_float, load_rows

from needlesim.benchmark.harness_scaled import DEFAULT_CSV_PATH
from needlesim.benchmark.scaled_scenarios import SCALED_HAND_DESIGNED

PLANNER_ORDER = ["VanillaRRT", "KinodynamicRRT", "RRTStar"]


def report_per_scenario(rows):
    print("=" * 92)
    print("SCALED SCENARIOS (500x500mm, R=1/kappa=50mm, all three planners)")
    print("=" * 92)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario_name"], r["planner"])].append(r)

    header = (
        f"{'scenario':<20} {'planner':<15} {'succ':>6} "
        f"{'cost_mm':>15} {'time_s':>14} {'iters':>14} "
        f"{'endpt_mm':>13} {'hdisc_max':>13}"
    )
    for scen in SCALED_HAND_DESIGNED:
        print()
        for planner in PLANNER_ORDER:
            runs = grouped.get((scen.name, planner), [])
            n = len(runs)
            ok = [r for r in runs if _success(r)]
            rate = f"{len(ok)}/{n}"
            cost = _mean_std([_to_float(r, "path_cost_mm") for r in ok])
            tm = _mean_std([_to_float(r, "wall_time_s") for r in ok])
            it = _mean_std([_to_float(r, "n_iterations") for r in ok])
            ep = _mean_std([_to_float(r, "endpoint_error_mm") for r in ok])
            hd = _mean_std([_to_float(r, "heading_disc_max_rad") for r in ok])
            if planner == PLANNER_ORDER[0]:
                print(header)
                print("-" * len(header))
            print(
                f"{scen.name:<20} {planner:<15} {rate:>6} "
                f"{_fnum(*cost, width=11)} {_fnum(*tm, width=10, prec=2)} "
                f"{_fnum(*it, width=10, prec=0)} "
                f"{_fnum(*ep, prec=2)} {_fnum(*hd, prec=3)}"
            )


def report_vanilla_tracked(rows):
    """VanillaRRT's tracked-execution metrics at scale (endpoint_error_mm is
    empty for vanilla by design -- see vanilla_tracker.py). Same three tracker
    metrics as the primary analysis: tracked endpoint, max cross-track, collide
    rate. At 500mm the point-robot fiction is starker still."""
    print()
    print("=" * 92)
    print("VANILLA TRACKED EXECUTION (open-loop segment following; vanilla only)")
    print("=" * 92)
    grouped = defaultdict(list)
    for r in rows:
        if r["planner"] == "VanillaRRT":
            grouped[r["scenario_name"]].append(r)

    header = (
        f"{'scenario':<22} {'succ':>6} {'tracked_endpt_mm':>18} "
        f"{'max_xtrack_mm':>16} {'collides':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for scen in SCALED_HAND_DESIGNED:
        runs = grouped.get(scen.name, [])
        ok = [r for r in runs if _success(r)]
        rate = f"{len(ok)}/{len(runs)}"
        ep = _mean_std([_to_float(r, "tracked_endpoint_error_mm") for r in ok])
        ct = _mean_std([_to_float(r, "tracked_max_crosstrack_mm") for r in ok])
        n_col = sum(1 for r in ok if r["tracked_collides"] == "True")
        col = f"{n_col}/{len(ok)}" if ok else "--"
        print(
            f"{scen.name:<22} {rate:>6} {_fnum(*ep, width=14, prec=1)} "
            f"{_fnum(*ct, width=12, prec=1)} {col:>10}"
        )


def report_cost_comparison(rows):
    """RRT* vs Kinodynamic path cost on scenarios where BOTH succeed. Paired by
    scenario over the seed means -- the headline number of the experiment."""
    print()
    print("=" * 92)
    print("HEADLINE: RRTStar vs KinodynamicRRT path cost, where BOTH succeed")
    print("=" * 92)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario_name"], r["planner"])].append(r)

    def mean_of(scen_name, planner, key):
        ok = [r for r in grouped.get((scen_name, planner), []) if _success(r)]
        vals = [_to_float(r, key) for r in ok]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    print(
        f"\n{'scenario':<20} {'kino_cost':>11} {'star_cost':>11} "
        f"{'cost_diff':>12} {'kino_t':>9} {'star_t':>9} {'time_x':>8}"
    )
    print("-" * 84)
    for scen in SCALED_HAND_DESIGNED:
        kc = mean_of(scen.name, "KinodynamicRRT", "path_cost_mm")
        sc = mean_of(scen.name, "RRTStar", "path_cost_mm")
        kt = mean_of(scen.name, "KinodynamicRRT", "wall_time_s")
        st = mean_of(scen.name, "RRTStar", "wall_time_s")
        if kc is None or sc is None:
            # Note which side failed -- especially target_behind.
            who = []
            if kc is None:
                who.append("kino 0-success")
            if sc is None:
                who.append("star 0-success")
            print(f"{scen.name:<20} {'--':>11} {'--':>11}  ({', '.join(who)})")
            continue
        diff_pct = 100.0 * (sc - kc) / kc
        time_x = st / kt if kt else float("nan")
        sign = "+" if diff_pct >= 0 else ""
        print(
            f"{scen.name:<20} {kc:>11.1f} {sc:>11.1f} "
            f"{sign}{diff_pct:>10.1f}% {kt:>9.2f} {st:>9.2f} {time_x:>7.1f}x"
        )
    print(
        "\ncost_diff > 0 means RRT* paths are LONGER (worse) than kinodynamic's; "
        "time_x is RRT* wall time / kino wall time."
    )


def report_target_behind(rows):
    """target_behind is kinodynamic 0/30 at the primary scale (symmetric
    split-effort). Whether RRTStar solves it at this scale is called out."""
    print()
    print("-" * 92)
    for planner in PLANNER_ORDER:
        runs = [
            r
            for r in rows
            if r["scenario_name"] == "target_behind" and r["planner"] == planner
        ]
        ok = sum(1 for r in runs if _success(r))
        print(f"target_behind: {planner:<15} {ok}/{len(runs)} succeeded")


def main():
    csv_path = DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise SystemExit(
            f"No CSV at {csv_path}. Run scripts/run_scaled_benchmark.py first."
        )
    rows = load_rows(csv_path)
    print(f"Read {len(rows)} raw records from {csv_path}\n")
    report_per_scenario(rows)
    report_vanilla_tracked(rows)
    report_cost_comparison(rows)
    report_target_behind(rows)
    print("\n(cost/time/iters/endpt/hdisc are mean +- std over SUCCESSFUL runs)")


if __name__ == "__main__":
    main()

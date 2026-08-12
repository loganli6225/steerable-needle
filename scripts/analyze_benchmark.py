"""Read the raw benchmark CSV and print summary tables. NEVER re-runs planners.

Hand-designed and random scenarios are reported SEPARATELY: the four are
illustrative (each gets a figure later), the 30 are statistical (an aggregate
only). Mixing them buries the illustrative cases and biases the aggregate,
since the four were chosen precisely because they are interesting.

  - Hand-designed: one row per (scenario, planner) -- success rate, and
    mean +/- std OVER SUCCESSFUL RUNS for cost, time, iterations, endpoint
    error, and heading discontinuity (per-run max, and the count of
    discontinuous nodes).
  - Random: aggregate across the 30 per planner, plus the
    both / vanilla-only / kino-only / neither breakdown (a scenario is
    "solved" by a planner if ANY of its seeds reached the goal).

Run:  python scripts/analyze_benchmark.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from needlesim.benchmark.harness import DEFAULT_CSV_PATH
from needlesim.benchmark.scenarios import HAND_DESIGNED

PLANNER_ORDER = ["VanillaRRT", "KinodynamicRRT"]


def _mean_std(values):
    """(mean, std) over a list; (nan, nan) if empty, std=0 for a single value."""
    if not values:
        return math.nan, math.nan
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def _fnum(mean, std, width=9, prec=1):
    if math.isnan(mean):
        return f"{'--':>{width}}"
    return f"{mean:>{width}.{prec}f}+-{std:.{prec}f}"


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row, key):
    v = row[key]
    return float(v) if v not in ("", None) else None


def _success(row):
    return row["success"] == "True"


def report_hand_designed(rows):
    print("=" * 78)
    print("HAND-DESIGNED SCENARIOS (illustrative, reported per scenario)")
    print("=" * 78)
    hand = [r for r in rows if r["scenario_kind"] == "hand"]
    grouped = defaultdict(list)
    for r in hand:
        grouped[(r["scenario_name"], r["planner"])].append(r)

    header = (
        f"{'scenario':<20} {'planner':<15} {'succ':>6} "
        f"{'cost_mm':>13} {'time_s':>12} {'iters':>13} "
        f"{'endpt_mm':>13} {'hdisc_max':>13} {'hdisc_n':>9}"
    )
    for scen in HAND_DESIGNED:
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
            hn = _mean_std([_to_float(r, "heading_disc_count") for r in ok])
            if planner == PLANNER_ORDER[0]:
                print(header)
                print("-" * len(header))
            print(
                f"{scen.name:<20} {planner:<15} {rate:>6} "
                f"{_fnum(*cost)} {_fnum(*tm, prec=2)} {_fnum(*it, prec=0)} "
                f"{_fnum(*ep, prec=2)} {_fnum(*hd, prec=3)} {_fnum(*hn, width=6, prec=1)}"
            )


def report_random(rows):
    print()
    print("=" * 78)
    print("RANDOM SCENARIOS (statistical, aggregate over 30)")
    print("=" * 78)
    rand = [r for r in rows if r["scenario_kind"] == "random"]

    # Aggregate metrics per planner over ALL runs / successful runs.
    print(
        f"\n{'planner':<15} {'succ':>10} {'cost_mm':>13} {'time_s':>12} "
        f"{'iters':>13} {'endpt_mm':>13} {'hdisc_max':>13}"
    )
    print("-" * 92)
    for planner in PLANNER_ORDER:
        runs = [r for r in rand if r["planner"] == planner]
        ok = [r for r in runs if _success(r)]
        rate = f"{len(ok)}/{len(runs)}"
        cost = _mean_std([_to_float(r, "path_cost_mm") for r in ok])
        tm = _mean_std([_to_float(r, "wall_time_s") for r in ok])
        it = _mean_std([_to_float(r, "n_iterations") for r in ok])
        ep = _mean_std([_to_float(r, "endpoint_error_mm") for r in ok])
        hd = _mean_std([_to_float(r, "heading_disc_max_rad") for r in ok])
        print(
            f"{planner:<15} {rate:>10} {_fnum(*cost)} {_fnum(*tm, prec=2)} "
            f"{_fnum(*it, prec=0)} {_fnum(*ep, prec=2)} {_fnum(*hd, prec=3)}"
        )

    # Per-scenario solved-by-any-seed classification.
    solved = defaultdict(lambda: {"VanillaRRT": False, "KinodynamicRRT": False})
    scen_names = set()
    for r in rand:
        scen_names.add(r["scenario_name"])
        if _success(r):
            solved[r["scenario_name"]][r["planner"]] = True

    tally = {"both": 0, "vanilla-only": 0, "kino-only": 0, "neither": 0}
    for name in scen_names:
        v = solved[name]["VanillaRRT"]
        k = solved[name]["KinodynamicRRT"]
        if v and k:
            tally["both"] += 1
        elif v:
            tally["vanilla-only"] += 1
        elif k:
            tally["kino-only"] += 1
        else:
            tally["neither"] += 1

    total = len(scen_names)
    print(f"\nSolved-by-any-seed breakdown over {total} random scenarios:")
    for cls in ("both", "vanilla-only", "kino-only", "neither"):
        pct = 100 * tally[cls] / total if total else 0
        print(f"  {cls:<14} {tally[cls]:>3}  ({pct:.0f}%)")


def main():
    csv_path = DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise SystemExit(f"No CSV at {csv_path}. Run scripts/run_benchmark.py first.")
    rows = load_rows(csv_path)
    print(f"Read {len(rows)} raw records from {csv_path}\n")
    report_hand_designed(rows)
    report_random(rows)
    print("\n(cost/time/iters/endpt/hdisc are mean +- std over SUCCESSFUL runs)")


if __name__ == "__main__":
    main()

"""Figures for the max-edge-length (steering-horizon) experiment.

Two kinds:

  1. A sweep SUMMARY plot (docs/figures/max_edge_sweep_summary.png): max edge
     length on the x-axis (finite thresholds ascending, inf at the right edge,
     labelled), with three stacked panels -- success rate, mean path cost over
     successes, and the choose_parent rejection rate. This is the figure that
     shows the trade: whether cost falls before success dies (helps) or after
     (starves). Where the success and cost curves cross is the result.

  2. PATH figures, one per threshold that yielded successes on scaled open
     (docs/figures/max_edge_open_<label>.png), in the SAME grammar as
     docs/figures/benchmark_scaled_*.png. Each shows RRT*'s near-median-cost path
     at that threshold (purple) against KinodynamicRRT's path as a fixed
     reference (blue), so the comparison is visible in every frame. The question
     these answer that the numbers cannot: do the loops DISAPPEAR, or just get
     truncated into a different shape?

Reads experiments/results/max_edge_sweep_raw.csv (produced by
scripts/run_max_edge_sweep.py). Path figures re-run the plotted RRT* seed to
recover its trajectory (the CSV stores metrics, not paths).

Run:  python scripts/plot_max_edge_figures.py
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the scaled figures' grammar so these read against the existing set.
from plot_scaled_benchmark_figures import (  # noqa: E402
    C_KINO,
    C_STAR,
    C_START,
    C_GOAL,
    _draw_geometry,
    _draw_pose,
    _finish,
    star_executed_trace,
)

from needlesim.benchmark.harness_scaled import KAPPA, make_rrt_config
from needlesim.benchmark.scaled_scenarios import SCALED_SCENARIOS  # noqa: E402
from needlesim.benchmark.scenarios import build_env  # noqa: E402
from needlesim.models.unicycle_needle import NeedleParams  # noqa: E402
from needlesim.planning.rrt import KinodynamicRRT  # noqa: E402
from needlesim.planning.rrt_star import RRTStar  # noqa: E402
from run_max_edge_sweep import DEFAULT_CSV_PATH, make_config  # noqa: E402

OUT_DIR = Path("docs/figures")


def _load_open_rows():
    with DEFAULT_CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["scenario_name"] == "open"]


def _thr_label(thr: float) -> str:
    return "inf" if math.isinf(thr) else f"{thr:.0f}"


def _parse_thr(s: str) -> float:
    return float(s)  # "inf" -> inf, "400.0" -> 400.0


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------


def _summary_by_threshold(rows):
    """threshold -> (success_rate, mean_cost_or_nan, mean_none_rate)."""
    by_thr = defaultdict(list)
    for r in rows:
        by_thr[_parse_thr(r["max_edge_length"])].append(r)
    out = {}
    for thr, rs in by_thr.items():
        n = len(rs)
        ok = [r for r in rs if r["success"] == "True"]
        rate = len(ok) / n
        cost = statistics.mean(float(r["path_cost_mm"]) for r in ok) if ok else math.nan
        none_rate = statistics.mean(float(r["choose_parent_none_rate"]) for r in rs)
        out[thr] = (rate, cost, none_rate)
    return out


def make_summary(rows):
    summ = _summary_by_threshold(rows)
    # x order: finite thresholds ascending, then inf at the right edge.
    thrs = sorted(summ, key=lambda x: (math.isinf(x), x))
    xs = list(range(len(thrs)))
    labels = [_thr_label(t) for t in thrs]
    rates = [summ[t][0] for t in thrs]
    costs = [summ[t][1] for t in thrs]
    nones = [summ[t][2] for t in thrs]

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 7.2), sharex=True)

    axes[0].plot(xs, [100 * r for r in rates], "o-", color="#2ca02c", linewidth=1.8)
    axes[0].set_ylabel("success rate [%]")
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, alpha=0.3)

    # cost: skip nan (no successes) so the line breaks rather than dips to 0.
    cx = [x for x, c in zip(xs, costs) if not math.isnan(c)]
    cc = [c for c in costs if not math.isnan(c)]
    axes[1].plot(cx, cc, "o-", color=C_STAR, linewidth=1.8)
    axes[1].set_ylabel("mean path cost [mm]\n(successes)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, [100 * n for n in nones], "o-", color="#d62728", linewidth=1.8)
    axes[2].set_ylabel("choose_parent\nrejection rate [%]")
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("max edge length [mm]  (inf = unconstrained, right edge)")

    axes[0].set_title(
        "Steering-horizon sweep on scaled `open`\n"
        "does a max-edge-length constraint help RRT*, or starve it?",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    out = OUT_DIR / "max_edge_sweep_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Path figures (per successful threshold)
# ---------------------------------------------------------------------------


def _median_seed(rows_for_thr):
    """Seed of the successful run nearest median cost (ties -> lower seed)."""
    succ = [
        (int(r["seed"]), float(r["path_cost_mm"]))
        for r in rows_for_thr
        if r["success"] == "True"
    ]
    if not succ:
        return None
    med = statistics.median(c for _, c in succ)
    seed, _ = min(succ, key=lambda sc: (abs(sc[1] - med), sc[0]))
    return seed


def _run_star(scenario, seed, thr):
    params = NeedleParams(kappa=KAPPA)
    cfg = make_config(seed, thr)
    result = RRTStar(build_env(scenario), params, cfg).plan(
        scenario.start, scenario.goal
    )
    return result, params


def _run_kino_reference(scenario, seed):
    params = NeedleParams(kappa=KAPPA)
    cfg = make_rrt_config(seed)
    result = KinodynamicRRT(build_env(scenario), params, cfg).plan(
        scenario.start, scenario.goal
    )
    from needlesim.models.unicycle_needle import rollout

    steps = [c for c in result.controls for _ in range(cfg.n_steps_per_extend)]
    trace = rollout(scenario.start, steps, cfg.step_dt, params)
    cost = (
        len(result.controls) * cfg.n_steps_per_extend * cfg.edge_velocity * cfg.step_dt
    )
    return [(s.x, s.y) for s in trace], cost


def make_path_figure(scenario, thr, rows_for_thr, kino_ref):
    seed = _median_seed(rows_for_thr)
    n = len(rows_for_thr)
    ok = sum(1 for r in rows_for_thr if r["success"] == "True")

    result, params = _run_star(scenario, seed, thr)
    sx, sy = zip(*star_executed_trace(result, scenario, params))

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    _draw_geometry(ax, build_env(scenario))

    kx, ky = zip(*kino_ref[0])
    ax.plot(kx, ky, color=C_KINO, linewidth=2.0, label="Kino (reference)", zorder=4)
    ax.plot(
        sx,
        sy,
        color=C_STAR,
        linewidth=1.8,
        label=f"RRT* (max_edge={_thr_label(thr)})",
        zorder=5,
    )
    _draw_pose(ax, scenario.start, C_START, "start", "o")
    _draw_pose(ax, scenario.goal, C_GOAL, "goal", "*")

    title = (
        f"open (scaled 500mm) — max_edge {_thr_label(thr)} mm\n"
        f"RRT* {ok}/{n}, cost {result.best_cost:.0f} mm "
        f"({result.best_cost / kino_ref[1]:.1f}x kino {kino_ref[1]:.0f} mm)"
    )
    subtitle = (
        f"RRT* seed {seed}; kino reference cost {kino_ref[1]:.0f} mm. "
        "Are the loops gone, or just truncated?"
    )
    _finish(ax, scenario, title, subtitle)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUT_DIR / f"max_edge_open_{_thr_label(thr)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, seed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_open_rows()
    if not rows:
        raise SystemExit(
            f"No open rows in {DEFAULT_CSV_PATH}. Run run_max_edge_sweep.py first."
        )

    summary_out = make_summary(rows)
    print(f"Summary figure -> {summary_out}\n")

    scenario = SCALED_SCENARIOS["open"]
    # Kino reference: one representative run (seed 0), drawn identically in every
    # frame so the comparison is fixed.
    kino_ref = _run_kino_reference(scenario, 0)

    by_thr = defaultdict(list)
    for r in rows:
        by_thr[_parse_thr(r["max_edge_length"])].append(r)

    print("Path figures (near-median-cost RRT* seed per threshold):")
    for thr in sorted(by_thr, key=lambda x: (math.isinf(x), x)):
        rs = by_thr[thr]
        if not any(r["success"] == "True" for r in rs):
            print(f"  max_edge={_thr_label(thr):>4}: 0 successes -- no path figure")
            continue
        out, seed = make_path_figure(scenario, thr, rs, kino_ref)
        print(f"  {out}   [seed={seed}]")


if __name__ == "__main__":
    main()

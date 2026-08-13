"""Deliverable figures: the four SCALED (500mm) secondary-experiment scenarios.

Companion to scripts/plot_benchmark_figures.py, and deliberately the same
picture grammar so a reader who has seen the primary figures reads these without
relearning: identical colours per planner, identical markers/legend, mm axes,
equal aspect, 150 dpi, tracked docs/figures/ location. The one addition is
RRTStar (purple) -- the whole point of the secondary experiment.

What each figure argues, BEFORE a reader reaches a number:

  - VanillaRRT (orange planned / red executed): its executed trajectory peels
    off immediately and leaves the 500mm arena -- the point-robot fiction, now
    even starker at scale (lands 368-498mm off).
  - KinodynamicRRT (blue): one smooth curve, planned == executed, landing on
    target. 10/10 on all four scaled scenarios.
  - RRTStar (purple): ALSO continuous (heading disc ~1e-13), but visibly a
    CHAIN OF LARGE LOOPS rather than a direct route. That loop structure IS the
    finding -- 4.4-5.7x the cost between the same endpoints, because the minimum
    turning circle (314mm) forces most of a loop for every pose correction. It
    is drawn to be legible, not tidied away.

cluttered is the exception: RRTStar scored 0/10, so there is no RRT* path. Kino
and vanilla are drawn as usual and one FAILED RRT* run's tree is overlaid faintly
-- as the primary figures did for target_behind's failed kino tree -- because
"RRT* collapses under clutter" is one of the three grounds for its exclusion and
the tree shows what that collapse looks like (loops sweeping into obstacles).

Seeds: the CSV stores metrics, not trajectories, so the plotted seeds are re-run
here to recover their paths -- the near-MEDIAN-cost successful seed per (scenario,
planner), representative not cherry-picked, and printed for reproducibility. RRT*
runs take ~130s each, so this is a few minutes, not a grid.

Output: docs/figures/benchmark_scaled_<scenario>.png (TRACKED; the raw CSV under
experiments/results/ is gitignored).

Run:  python scripts/plot_scaled_benchmark_figures.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reuse the primary figures' colours (same planner -> same colour) and the
# generic CSV/seed helpers, so the two figure sets are guaranteed consistent.
from plot_benchmark_figures import (  # noqa: E402
    C_GOAL,
    C_KINO,
    C_MARGIN,
    C_OBSTACLE,
    C_START,
    C_SURFACE,
    C_VAN_EXEC,
    C_VAN_PLAN,
    _by_scen_planner,
    executed_trace,
    median_seed,
    planned_polyline,
    success_rate,
)

from needlesim.benchmark.harness import (  # noqa: E402
    endpoint_error_mm,
    heading_discontinuity,
    path_cost_mm,
)
from needlesim.benchmark.harness_scaled import (  # noqa: E402
    DEFAULT_CSV_PATH,
    KAPPA,
    make_rrt_config,
    make_rrtstar_config,
    rrtstar_endpoint_error_mm,
    rrtstar_heading_discontinuity,
)
from needlesim.benchmark.scaled_scenarios import (  # noqa: E402
    SCALE,
    SCALED_COMMON_CONDITIONS,
    SCALED_SCENARIOS,
)
from needlesim.benchmark.scenarios import build_env  # noqa: E402
from needlesim.models.unicycle_needle import (  # noqa: E402
    NeedleParams,
    State,
    rollout_variable,
)
from needlesim.planning.rrt import KinodynamicRRT, VanillaRRT  # noqa: E402
from needlesim.planning.rrt_star import RRTStar  # noqa: E402

OUT_DIR = Path("docs/figures")
MARGIN = SCALED_COMMON_CONDITIONS["margin"]  # 2.0 -- NOT scaled (needle width)
GOAL_TOL = SCALED_COMMON_CONDITIONS["goal_tolerance"]  # 10.0 -- scaled

# RRTStar's colours -- new to this set (purple, distinct from kino's blue).
C_STAR = "#9467bd"  # RRT* planned == executed
C_STAR_TREE = "#c5b0d5"  # faint failed-RRT* tree (cluttered only)

# Data-unit drawing geometry scales with the workspace so the figures read at the
# SAME visual proportion as the primary set (point-unit sizes -- markersize,
# linewidths, fonts -- are screen-relative and stay put).
POSE_ARROW_LEN = 11.0 * SCALE
POSE_HEAD_W = 3.0 * SCALE
POSE_HEAD_L = 3.5 * SCALE


def _scaled_rows():
    with DEFAULT_CSV_PATH.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["scenario_kind"] == "scaled"]


# ---------------------------------------------------------------------------
# Re-run a seed to recover its path.
# ---------------------------------------------------------------------------


def run_rrt(planner_cls, scenario, seed):
    env = build_env(scenario)
    params = NeedleParams(kappa=KAPPA)
    cfg = make_rrt_config(seed)
    result = planner_cls(env, params, cfg).plan(scenario.start, scenario.goal)
    return result, cfg, params


def run_star(scenario, seed):
    env = build_env(scenario)
    params = NeedleParams(kappa=KAPPA)
    cfg = make_rrtstar_config(seed)
    result = RRTStar(env, params, cfg).plan(scenario.start, scenario.goal)
    return result, cfg, params


def star_executed_trace(result, scenario, params):
    """RRT*'s planned path == its executed trajectory: roll the (control, dt)
    pairs through the real model. Continuous by construction."""
    trace = rollout_variable(scenario.start, result.control_dt_pairs, params)
    return [(s.x, s.y) for s in trace]


# ---------------------------------------------------------------------------
# Drawing (data-unit geometry scaled; colours/point-sizes identical to primary)
# ---------------------------------------------------------------------------


def _draw_geometry(ax, env):
    xs = (np.arange(env.n_cols) + 0.5) * env.resolution
    ys = (np.arange(env.n_rows) + 0.5) * env.resolution
    ax.contourf(xs, ys, env.sdf, levels=[env.sdf.min(), 0.0], colors=[C_OBSTACLE])
    ax.contour(xs, ys, env.sdf, levels=[0.0], colors=[C_SURFACE], linewidths=1.0)
    # margin contour: thin at 500mm because margin (2mm) does NOT scale -- it is
    # the physically correct needle width regardless of workspace size.
    ax.contour(
        xs,
        ys,
        env.sdf,
        levels=[MARGIN],
        colors=[C_MARGIN],
        linewidths=0.9,
        linestyles="dashed",
    )


def _draw_pose(ax, pose: State, color, label, marker):
    ax.plot(
        pose.x,
        pose.y,
        marker=marker,
        color=color,
        markersize=11,
        linestyle="none",
        label=label,
        zorder=6,
    )
    ax.arrow(
        pose.x,
        pose.y,
        POSE_ARROW_LEN * math.cos(pose.theta),
        POSE_ARROW_LEN * math.sin(pose.theta),
        head_width=POSE_HEAD_W,
        head_length=POSE_HEAD_L,
        fc=color,
        ec=color,
        length_includes_head=True,
        zorder=6,
    )


def _draw_tree(ax, result, color):
    """Faint exploration tree from a (failed) run: one segment per node->parent."""
    for node in result.nodes:
        if node.parent is None:
            continue
        p = result.nodes[node.parent].state
        ax.plot(
            [node.state.x, p.x],
            [node.state.y, p.y],
            color=color,
            linewidth=0.5,
            alpha=0.6,
            zorder=2,
        )


def _finish(ax, scenario, title, subtitle, legend_loc="upper right"):
    ax.add_patch(
        plt.Circle(
            (scenario.goal.x, scenario.goal.y),
            GOAL_TOL,
            fill=False,
            color=C_GOAL,
            linestyle=":",
            linewidth=1.0,
            zorder=6,
        )
    )
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc=legend_loc, fontsize=7, framealpha=0.92)
    ax.text(
        0.5,
        -0.14,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )


def make_figure(scen_name, grouped):
    scenario = SCALED_SCENARIOS[scen_name]
    env = build_env(scenario)
    v_ok, v_n = success_rate(grouped, scen_name, "VanillaRRT")
    k_ok, k_n = success_rate(grouped, scen_name, "KinodynamicRRT")
    s_ok, s_n = success_rate(grouped, scen_name, "RRTStar")

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    _draw_geometry(ax, env)

    subtitle_parts = []

    # --- Vanilla: planned polyline + executed trace (both always present) ----
    v_seed = median_seed(grouped, scen_name, "VanillaRRT")
    v_result, v_cfg, v_params = run_rrt(VanillaRRT, scenario, v_seed)
    px, py = zip(*planned_polyline(v_result))
    ax.plot(px, py, color=C_VAN_PLAN, linewidth=1.6, label="Vanilla planned", zorder=4)
    ex, ey = zip(*executed_trace(v_result, scenario, v_cfg, v_params))
    ax.plot(
        ex,
        ey,
        color=C_VAN_EXEC,
        linewidth=1.8,
        linestyle=(0, (5, 2)),
        label="Vanilla executed",
        zorder=5,
    )
    v_ep = endpoint_error_mm(v_result, scenario.start, scenario.goal, v_cfg, v_params)
    v_cost = path_cost_mm(v_result, v_cfg)
    v_hd, _, v_hn = heading_discontinuity(v_result, v_cfg, v_params)
    seeds_note = f"seeds: vanilla={v_seed}"
    subtitle_parts.append(
        f"Vanilla (seed {v_seed}): {v_cost:.0f} mm, executed {v_ep:.0f} mm off, "
        f"hdisc {v_hd:.2f} rad @ {v_hn} nodes"
    )

    # --- Kinodynamic: one smooth curve (planned == executed). 10/10 at scale --
    k_seed = median_seed(grouped, scen_name, "KinodynamicRRT")
    k_result, k_cfg, k_params = run_rrt(KinodynamicRRT, scenario, k_seed)
    kx, ky = zip(*executed_trace(k_result, scenario, k_cfg, k_params))
    ax.plot(
        kx, ky, color=C_KINO, linewidth=2.0, label="Kino planned = executed", zorder=5
    )
    k_ep = endpoint_error_mm(k_result, scenario.start, scenario.goal, k_cfg, k_params)
    k_cost = path_cost_mm(k_result, k_cfg)
    k_hd, _, k_hn = heading_discontinuity(k_result, k_cfg, k_params)
    seeds_note += f", kino={k_seed}"
    subtitle_parts.append(
        f"Kino (seed {k_seed}): {k_cost:.0f} mm, lands {k_ep:.1f} mm off, "
        f"hdisc {k_hd:.3f} rad ({k_hn} nodes)"
    )

    # --- RRTStar: continuous loop-chain where it succeeds; failed tree if 0/n -
    if s_ok > 0:
        s_seed = median_seed(grouped, scen_name, "RRTStar")
        s_result, _, s_params = run_star(scenario, s_seed)
        sx, sy = zip(*star_executed_trace(s_result, scenario, s_params))
        ax.plot(
            sx,
            sy,
            color=C_STAR,
            linewidth=1.8,
            label="RRT* planned = executed",
            zorder=4,
        )
        s_ep = rrtstar_endpoint_error_mm(
            s_result, scenario.start, scenario.goal, s_params
        )
        s_hd, _, s_hn = rrtstar_heading_discontinuity(s_result, s_params)
        seeds_note += f", star={s_seed}"
        subtitle_parts.append(
            f"RRT* (seed {s_seed}): {s_result.best_cost:.0f} mm = "
            f"{s_result.best_cost / k_cost:.1f}x kino, lands {s_ep:.1f} mm off, "
            f"hdisc {s_hd:.0e} rad"
        )
    else:
        fail_seed = 0
        s_result, _, _ = run_star(scenario, fail_seed)
        _draw_tree(ax, s_result, C_STAR_TREE)
        ax.plot(
            [],
            [],
            color=C_STAR_TREE,
            linewidth=1.5,
            label=f"RRT* tree (seed {fail_seed}, failed)",
        )
        seeds_note += f", star-fail-tree={fail_seed}"
        subtitle_parts.append(
            f"RRT* {s_ok}/{s_n}: its long loops sweep into the clutter and "
            "collide; none reach goal in budget"
        )

    _draw_pose(ax, scenario.start, C_START, "start", "o")
    _draw_pose(ax, scenario.goal, C_GOAL, "goal", "*")

    title = (
        f"{scen_name} — scaled 500mm\n"
        f"Vanilla {v_ok}/{v_n} · Kino {k_ok}/{k_n} · RRT* {s_ok}/{s_n}"
    )
    legend_loc = "upper left" if scen_name == "cluttered" else "upper right"
    _finish(ax, scenario, title, "\n".join(subtitle_parts), legend_loc=legend_loc)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT_DIR / f"benchmark_scaled_{scen_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, seeds_note


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = _by_scen_planner(_scaled_rows())
    print("Scaled deliverable figures (near-median-cost representative seeds):\n")
    for scen_name in ("open", "constrained_passage", "target_behind", "cluttered"):
        out, seeds = make_figure(scen_name, grouped)
        print(f"  {out}   [{seeds}]")


if __name__ == "__main__":
    main()

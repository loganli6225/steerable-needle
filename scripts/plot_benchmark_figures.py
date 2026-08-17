"""Deliverable figures: the four hand-designed benchmark scenarios.

These are the write-up figures, not diagnostics. Each makes the benchmark's
argument visually BEFORE a reader reaches a number: VanillaRRT is fast but its
planned path kinks at every node and its EXECUTED trajectory (its controls
rolled through the real needle model) peels away and leaves the arena;
KinodynamicRRT is slower and sometimes fails, but its planned and executed
paths are the same smooth curve that lands on target.

Seeds: the CSV stores metrics, not trajectories, so the specific plotted seeds
are re-run here to recover their paths. The plotted run is the near-MEDIAN-cost
successful run per (scenario, planner) -- representative, not cherry-picked.
The chosen seeds are printed so the figures are reproducible.

target_behind is the exception: KinodynamicRRT scored 0/30, so there is no kino
path. Vanilla's planned + executed paths are drawn, and one failed kino run's
TREE is overlaid faintly -- it shows HOW it failed (effort split between the
two equally-costly detours around the obstacle, neither completing), which is
the scenario's actual finding.

Output: docs/figures/benchmark_<scenario>.png at 150 dpi (a TRACKED location;
experiments/results/ is gitignored).

Run:  python scripts/plot_benchmark_figures.py
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
import numpy as np  # noqa: E402

from needlesim.benchmark.harness import (  # noqa: E402
    DEFAULT_CSV_PATH,
    KAPPA,
    _expand_controls,
    endpoint_error_mm,
    heading_discontinuity,
    make_config,
)
from needlesim.benchmark.scenarios import SCENARIOS, build_env  # noqa: E402
from needlesim.benchmark.vanilla_tracker import track_path  # noqa: E402
from needlesim.models.unicycle_needle import (  # noqa: E402
    NeedleParams,
    State,
    rollout,
)
from needlesim.planning.rrt import KinodynamicRRT, VanillaRRT  # noqa: E402

OUT_DIR = Path("docs/figures")
MARGIN = 2.0
GOAL_TOL = 3.0

# Consistent legend across all four figures -- learn it once.
C_OBSTACLE = "#b8b8b8"
C_SURFACE = "#5a5a5a"
C_MARGIN = "#d1495b"
C_START = "#2ca02c"
C_GOAL = "#111111"
C_VAN_PLAN = "#ff7f0e"  # vanilla planned polyline
C_VAN_EXEC = "#d62728"  # vanilla executed (the money curve)
C_KINO = "#1f77b4"  # kinodynamic planned == executed
C_KINO_TREE = "#9ecae1"  # faint failed-kino tree (target_behind only)


# ---------------------------------------------------------------------------
# CSV: pick the near-median-cost successful seed, and success rates.
# ---------------------------------------------------------------------------


def _hand_rows():
    with DEFAULT_CSV_PATH.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["scenario_kind"] == "hand"]


def _by_scen_planner(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r["scenario_name"], r["planner"])].append(r)
    return d


def success_rate(grouped, scen_name, planner):
    runs = grouped[(scen_name, planner)]
    n_ok = sum(1 for r in runs if r["success"] == "True")
    return n_ok, len(runs)


def median_seed(grouped, scen_name, planner):
    """Seed of the successful run whose cost is nearest the median (ties -> lower
    seed). None if no successful run."""
    runs = [
        (int(r["seed"]), float(r["path_cost_mm"]))
        for r in grouped[(scen_name, planner)]
        if r["success"] == "True"
    ]
    if not runs:
        return None
    med = statistics.median(c for _, c in runs)
    seed, _ = min(runs, key=lambda sc: (abs(sc[1] - med), sc[0]))
    return seed


# ---------------------------------------------------------------------------
# Re-run a seed to recover its path.
# ---------------------------------------------------------------------------


def run(planner_cls, scenario, seed):
    env = build_env(scenario)
    params = NeedleParams(kappa=KAPPA)
    cfg = make_config(seed)
    planner = planner_cls(env, params, cfg)
    result = planner.plan(scenario.start, scenario.goal)
    return result, cfg, params


def planned_polyline(result):
    return [(s.x, s.y) for s in result.path]


def executed_trace(result, scenario, cfg, params):
    """Roll the returned controls through the real model from the start -- where
    the needle actually goes if you execute this plan. Used for Kino/RRTStar,
    whose stored controls ARE model-generated (planned == executed)."""
    steps = _expand_controls(result.controls, cfg.n_steps_per_extend)
    trace = rollout(scenario.start, steps, cfg.step_dt, params)
    return [(s.x, s.y) for s in trace]


def vanilla_tracked(result, scenario, cfg, params, env):
    """VanillaRRT's executed curve is the OPEN-LOOP TRACKED trajectory -- its
    path read as a reference polyline and followed with feasible curved
    controls (see benchmark/vanilla_tracker.py) -- NOT its raw stored controls,
    which just drive a radius-1/kappa circle. Returns the TrackedRun so the
    figure can also mark where it strays into an obstacle."""
    return track_path(
        result.path,
        scenario.start,
        scenario.goal,
        cfg.edge_velocity,
        cfg.step_dt,
        env,
        params,
        cfg.margin,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_geometry(ax, env):
    xs = (np.arange(env.n_cols) + 0.5) * env.resolution
    ys = (np.arange(env.n_rows) + 0.5) * env.resolution
    ax.contourf(xs, ys, env.sdf, levels=[env.sdf.min(), 0.0], colors=[C_OBSTACLE])
    ax.contour(xs, ys, env.sdf, levels=[0.0], colors=[C_SURFACE], linewidths=1.0)
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
    L = 11.0
    ax.arrow(
        pose.x,
        pose.y,
        L * math.cos(pose.theta),
        L * math.sin(pose.theta),
        head_width=3.0,
        head_length=3.5,
        fc=color,
        ec=color,
        length_includes_head=True,
        zorder=6,
    )


def _draw_vanilla_executed(ax, tracked):
    """Draw the tracked (executed) trajectory and, if it collides, mark the
    first colliding state with a bold X and redraw the post-collision tail as a
    solid line -- so `tracked_collides` (the headline) is readable at a glance."""
    ex = [s.x for s in tracked.trace]
    ey = [s.y for s in tracked.trace]
    ax.plot(
        ex,
        ey,
        color=C_VAN_EXEC,
        linewidth=1.8,
        linestyle=(0, (5, 2)),
        label="Vanilla executed",
        zorder=5,
    )
    if tracked.collides:
        ci = tracked.first_collision_index
        # Post-collision tail solid: the needle is already inside tissue here.
        ax.plot(ex[ci:], ey[ci:], color=C_VAN_EXEC, linewidth=1.8, zorder=5)
        ax.plot(
            ex[ci],
            ey[ci],
            marker="X",
            color=C_VAN_EXEC,
            markersize=13,
            markeredgecolor="black",
            markeredgewidth=0.9,
            linestyle="none",
            label="Vanilla collision",
            zorder=7,
        )


def _draw_tree(ax, result):
    """Faint exploration tree from a (failed) run: one segment per node->parent."""
    for node in result.nodes:
        if node.parent is None:
            continue
        p = result.nodes[node.parent].state
        ax.plot(
            [node.state.x, p.x],
            [node.state.y, p.y],
            color=C_KINO_TREE,
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
    scenario = SCENARIOS[scen_name]
    env = build_env(scenario)
    v_ok, v_n = success_rate(grouped, scen_name, "VanillaRRT")
    k_ok, k_n = success_rate(grouped, scen_name, "KinodynamicRRT")

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    _draw_geometry(ax, env)

    # --- Vanilla: planned polyline + executed trace (both always present) ----
    v_seed = median_seed(grouped, scen_name, "VanillaRRT")
    v_result, v_cfg, v_params = run(VanillaRRT, scenario, v_seed)
    px, py = zip(*planned_polyline(v_result))
    ax.plot(px, py, color=C_VAN_PLAN, linewidth=1.6, label="Vanilla planned", zorder=4)
    v_tracked = vanilla_tracked(v_result, scenario, v_cfg, v_params, env)
    _draw_vanilla_executed(ax, v_tracked)
    v_hd, _, v_hn = heading_discontinuity(v_result, v_cfg, v_params)

    seeds_note = f"seeds: vanilla={v_seed}"
    collide_note = "COLLIDES" if v_tracked.collides else "no collision"
    subtitle_parts = [
        f"Vanilla (seed {v_seed}): tracked lands "
        f"{v_tracked.endpoint_error_mm:.0f} mm off, max cross-track "
        f"{v_tracked.max_crosstrack_mm:.0f} mm — {collide_note}",
        f"    heading disc max {v_hd:.2f} rad at {v_hn} nodes (planned path)",
    ]

    # --- Kinodynamic: one smooth curve (planned == executed), or a failed tree
    if k_ok > 0:
        k_seed = median_seed(grouped, scen_name, "KinodynamicRRT")
        k_result, k_cfg, k_params = run(KinodynamicRRT, scenario, k_seed)
        kx, ky = zip(*executed_trace(k_result, scenario, k_cfg, k_params))
        ax.plot(
            kx,
            ky,
            color=C_KINO,
            linewidth=2.0,
            label="Kino planned = executed",
            zorder=5,
        )
        k_ep = endpoint_error_mm(
            k_result, scenario.start, scenario.goal, k_cfg, k_params
        )
        k_hd, _, k_hn = heading_discontinuity(k_result, k_cfg, k_params)
        seeds_note += f", kino={k_seed}"
        subtitle_parts.append(
            f"Kino (seed {k_seed}): lands {k_ep:.1f} mm off; "
            f"heading disc {k_hd:.3f} rad ({k_hn} nodes)"
        )
    else:
        # No kino path -- overlay one failed run's tree to show HOW it failed.
        fail_seed = 0
        k_result, _, _ = run(KinodynamicRRT, scenario, fail_seed)
        _draw_tree(ax, k_result)
        ax.plot(
            [],
            [],
            color=C_KINO_TREE,
            linewidth=1.5,
            label=f"Kino tree (seed {fail_seed}, failed)",
        )
        seeds_note += f", kino-fail-tree={fail_seed}"
        subtitle_parts.append(
            f"Kino {k_ok}/{k_n}: effort splits between the two equal detours; "
            "neither completes in budget"
        )

    _draw_pose(ax, scenario.start, C_START, "start", "o")
    _draw_pose(ax, scenario.goal, C_GOAL, "goal", "*")

    title = f"{scen_name}   (Vanilla {v_ok}/{v_n}, Kino {k_ok}/{k_n})"
    # Legend content is identical everywhere (learn it once); only its corner
    # moves, to avoid occluding the goal where it sits top-right (cluttered).
    legend_loc = "upper left" if scen_name == "cluttered" else "upper right"
    _finish(ax, scenario, title, "\n".join(subtitle_parts), legend_loc=legend_loc)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = OUT_DIR / f"benchmark_{scen_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, seeds_note


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped = _by_scen_planner(_hand_rows())
    print("Deliverable benchmark figures (near-median-cost representative seeds):\n")
    for scen_name in ("open", "constrained_passage", "target_behind", "cluttered"):
        out, seeds = make_figure(scen_name, grouped)
        print(f"  {out}   [{seeds}]")


if __name__ == "__main__":
    main()

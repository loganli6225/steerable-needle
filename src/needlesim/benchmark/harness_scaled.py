"""SECONDARY benchmark harness: all THREE planners on the enlarged (500mm)
scenarios. Companion to harness.py; see scaled_scenarios.py for why this scale
exists and docs/benchmark_scaled_results.md for the finding.

REUSE, and where it stops. The RRT-family metrics (path_cost_mm,
endpoint_error_mm, heading_discontinuity), the RunRecord row, the CSV columns,
and the resumable-append machinery are imported wholesale from harness.py --
VanillaRRT and KinodynamicRRT have the identical RRTResult shape here as in the
primary benchmark, so their rows are computed by the exact same code. RRTStar is
the one thing that does NOT share a shape: its result carries control_dt_pairs
(variable-dt (Control, dt) segments) and a best_cost field instead of a flat
uniform-step controls list, so it needs three small metric ADAPTERS below. They
produce the same RunRecord columns, so the two planner families land in one CSV
the analysis reads uniformly.

BUDGETS ARE PER-PLANNER AND CALIBRATED, NOT INHERITED (see the calibration in
docs/benchmark_scaled_results.md):
  - RRT family: 20000 iterations. KinodynamicRRT's worst calibrated seed on
    scaled `open` needed 10482; 20000 leaves headroom. Both RRT planners stop
    at first goal contact, so this is a ceiling, not a fixed cost.
  - RRTStar: 5000 iterations (it NEVER stops early -- runs the whole budget by
    design). Calibration showed cost plateaus by 5000 (identical at 10000).
  - RRTStar gamma / max_radius: SCALED x10/3 to ~133. The rewire radius is in
    mm; at the primary default (40) RRTStar's neighbourhoods shrink below what
    the enlarged scene needs and it fails EVERY run. 133 is the scale-consistent
    value that makes it function -- not tuning-for-numbers, the same x10/3 the
    geometry gets.

MARGIN IS SET EXPLICITLY AND IDENTICALLY (2.0) for all three. RRTConfig defaults
it to 2.0 but RRTStarConfig defaults it to 0.0 (a recorded roadmap hazard);
comparing planners at different margins would be invalid, so neither default is
trusted here.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

from needlesim.benchmark.harness import (
    CSV_COLUMNS,
    HEADING_DISC_THRESHOLD,
    RunRecord,
    _existing_keys,
    _wrap,
    endpoint_error_mm,
    heading_discontinuity,
    path_cost_mm,
)
from needlesim.benchmark.scaled_scenarios import (
    SCALED_COMMON_CONDITIONS,
    SCALED_HAND_DESIGNED,
    SCALE,
)
from needlesim.benchmark.scenarios import build_env
from needlesim.models.unicycle_needle import NeedleParams, State, rollout_variable
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig, VanillaRRT
from needlesim.planning.rrt_star import RRTStar, RRTStarConfig

KAPPA = SCALED_COMMON_CONDITIONS["kappa"]
RRT_BUDGET = 20000
RRTSTAR_BUDGET = 5000
RRTSTAR_GAMMA = 40.0 * SCALE  # ~133.3 -- scaled from the primary default of 40
RRTSTAR_MAX_RADIUS = 40.0 * SCALE
SEEDS = 10

DEFAULT_CSV_PATH = Path("experiments/results/benchmark_scaled_raw.csv")

# VanillaRRT/KinodynamicRRT are RRT-family (RRTResult shape); RRTStar is not.
RRT_PLANNERS = {"VanillaRRT": VanillaRRT, "KinodynamicRRT": KinodynamicRRT}
PLANNER_ORDER = ["VanillaRRT", "KinodynamicRRT", "RRTStar"]


def make_rrt_config(seed: int) -> RRTConfig:
    return RRTConfig(
        max_iterations=RRT_BUDGET,
        goal_tolerance=SCALED_COMMON_CONDITIONS["goal_tolerance"],
        step_dt=SCALED_COMMON_CONDITIONS["step_dt"],
        edge_velocity=SCALED_COMMON_CONDITIONS["edge_velocity"],
        margin=SCALED_COMMON_CONDITIONS["margin"],
        seed=seed,
    )


def make_rrtstar_config(seed: int) -> RRTStarConfig:
    return RRTStarConfig(
        max_iterations=RRTSTAR_BUDGET,
        gamma=RRTSTAR_GAMMA,
        max_radius=RRTSTAR_MAX_RADIUS,
        goal_tolerance=SCALED_COMMON_CONDITIONS["goal_tolerance"],
        step_dt=SCALED_COMMON_CONDITIONS["step_dt"],
        edge_velocity=SCALED_COMMON_CONDITIONS["edge_velocity"],
        margin=SCALED_COMMON_CONDITIONS["margin"],  # explicit: default is 0.0
        seed=seed,
    )


# ---------------------------------------------------------------------------
# RRT* metric adapters. Same three quantities as the primary metrics, computed
# from RRTStarResult's differently-shaped fields. None mutate anything.
# ---------------------------------------------------------------------------


def rrtstar_endpoint_error_mm(result, start: State, goal: State, params) -> float:
    """Distance from goal after EXECUTING the returned (control, dt) pairs on the
    real needle. RRT*'s edges are Dubins paths integrated with the same model, so
    this lands within goal_tolerance (calibration: ~0.002mm on scaled open)."""
    final = rollout_variable(start, result.control_dt_pairs, params)[-1]
    return math.hypot(final.x - goal.x, final.y - goal.y)


def rrtstar_heading_discontinuity(result, params) -> tuple[float, float, int]:
    """(max, mean, count-over-threshold) of heading discontinuity at interior
    path nodes -- the SAME geometric-infeasibility measure as the RRT-family
    metric, adapted to RRT*'s variable-dt edges.

    reconstruct() flattens the per-edge controls, so the per-edge boundaries are
    recovered by mapping each path State back to its Node by object identity
    (reconstruct appends the very same State objects it read from the nodes). For
    each interior node the arrival heading is the theta after rolling its
    edge_from_parent from the previous path node; the departure heading is the
    node's stored theta. Because both come from the same continuous Dubins
    integration, disc ~0 (calibration: max 1.2e-13) -- if it is not, something is
    wrong and the run should be inspected, not the threshold adjusted."""
    path = result.path
    node_by_state_id = {id(n.state): n for n in result.nodes}

    discs = []
    for k in range(1, len(path) - 1):
        node = node_by_state_id[id(path[k])]
        arrival = rollout_variable(path[k - 1], node.edge_from_parent.controls, params)[
            -1
        ].theta
        departure = path[k].theta
        discs.append(abs(_wrap(arrival - departure)))

    if not discs:
        return 0.0, 0.0, 0
    max_d = max(discs)
    mean_d = sum(discs) / len(discs)
    count = sum(1 for d in discs if d > HEADING_DISC_THRESHOLD)
    return max_d, mean_d, count


# ---------------------------------------------------------------------------
# One run + the grid
# ---------------------------------------------------------------------------


def run_one(scenario, planner_name: str, seed: int) -> RunRecord:
    """Execute one (scenario, planner, seed) run and compute its metrics. A fresh
    env is baked per run so no SDF is shared. Dispatches on planner family: the
    RRT pair uses the primary metrics, RRTStar the adapters above."""
    env = build_env(scenario)
    params = NeedleParams(kappa=KAPPA)

    t0 = time.perf_counter()
    if planner_name == "RRTStar":
        cfg = make_rrtstar_config(seed)
        result = RRTStar(env, params, cfg).plan(scenario.start, scenario.goal)
    else:
        cfg = make_rrt_config(seed)
        result = RRT_PLANNERS[planner_name](env, params, cfg).plan(
            scenario.start, scenario.goal
        )
    wall = time.perf_counter() - t0

    base = dict(
        scenario_name=scenario.name,
        scenario_kind="scaled",
        planner=planner_name,
        seed=seed,
        success=result.success,
        n_iterations=result.n_iterations,
        n_nodes=len(result.nodes),
        wall_time_s=wall,
    )
    if not result.success:
        return RunRecord(
            **base,
            path_cost_mm=None,
            endpoint_error_mm=None,
            heading_disc_max_rad=None,
            heading_disc_mean_rad=None,
            heading_disc_count=None,
        )

    if planner_name == "RRTStar":
        hmax, hmean, hcount = rrtstar_heading_discontinuity(result, params)
        return RunRecord(
            **base,
            path_cost_mm=result.best_cost,  # true summed edge length
            endpoint_error_mm=rrtstar_endpoint_error_mm(
                result, scenario.start, scenario.goal, params
            ),
            heading_disc_max_rad=hmax,
            heading_disc_mean_rad=hmean,
            heading_disc_count=hcount,
        )

    hmax, hmean, hcount = heading_discontinuity(result, cfg, params)
    return RunRecord(
        **base,
        path_cost_mm=path_cost_mm(result, cfg),
        endpoint_error_mm=endpoint_error_mm(
            result, scenario.start, scenario.goal, cfg, params
        ),
        heading_disc_max_rad=hmax,
        heading_disc_mean_rad=hmean,
        heading_disc_count=hcount,
    )


def iter_grid():
    """Yield (scenario, planner_name, seed) for the 4x3x10 = 120-run grid."""
    for scenario in SCALED_HAND_DESIGNED:
        for planner_name in PLANNER_ORDER:
            for seed in range(SEEDS):
                yield scenario, planner_name, seed


def run_harness(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    """Run the grid, appending one row per run. Resumable: any (scenario,
    planner, seed) already in csv_path is skipped on startup."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(csv_path)
    grid = list(iter_grid())
    total = len(grid)
    print(
        f"Scaled grid: {len(SCALED_HAND_DESIGNED)} scenarios x "
        f"{len(PLANNER_ORDER)} planners x {SEEDS} seeds = {total} runs\n"
        f"  RRT budget={RRT_BUDGET}  RRT* budget={RRTSTAR_BUDGET}  "
        f"RRT* gamma={RRTSTAR_GAMMA:.1f}  margin={SCALED_COMMON_CONDITIONS['margin']}"
    )
    if done:
        print(f"Resuming: {len(done)} of {total} runs already recorded.")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for i, (scenario, planner_name, seed) in enumerate(grid, 1):
            key = (scenario.name, planner_name, str(seed))
            if key in done:
                continue
            record = run_one(scenario, planner_name, seed)
            writer.writerow(asdict(record))
            f.flush()
            flag = "ok " if record.success else "FAIL"
            print(
                f"[{i:>3}/{total}] {flag} {scenario.name:<20} "
                f"{planner_name:<15} seed={seed:<2} {record.wall_time_s:7.2f}s"
            )
    print(f"Done. Raw records -> {csv_path}")

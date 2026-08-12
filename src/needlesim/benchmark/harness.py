"""Benchmark harness: run the planner x scenario x seed grid, one CSV row/run.

Records RAW per-run records, not just aggregates, so the results can be
re-sliced (per-scenario distributions, cost histograms, filtering by success)
WITHOUT re-running 840 planner invocations. Rows are appended as each run
completes, which makes the harness resumable after a crash and gives live
progress on a ~30-60 min run. Analysis lives in a SEPARATE script so tables
regenerate without re-running.

The grid (see docs/roadmap.md for why RRTStar is excluded -- it cannot connect
poses at R=50mm in this workspace, a recorded finding, not an oversight):
  - 4 hand-designed scenarios x 2 planners x 30 seeds = 240 runs
  - 30 random scenarios   x 2 planners x 10 seeds = 600 runs

EQUAL ITERATION COUNTS ARE NOT EQUAL WORK: one VanillaRRT iteration is a single
straight-line rollout plus a collision check; one KinodynamicRRT iteration is
two arc rollouts plus a comparison. That is exactly why wall_time_s is recorded
as its own metric rather than inferred from n_iterations.

ONE MODEL FOR ALL METRICS. Every metric rollout below uses NeedleParams(KAPPA)
-- the real needle. "Execute this plan on the needle and see where it lands" is
the semantics of endpoint_error and of the heading-arrival rollout. For
KinodynamicRRT this IS the model that built the tree, so arrival headings
reproduce the stored node headings and heading discontinuity is ~0. For
VanillaRRT the tree was built with a kappa=0 point-robot cheat plus a per-node
heading teleport, so under the real model its path neither lands on the goal
nor has continuous headings -- the benchmark's central claim, measured.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from needlesim.benchmark.random_scenarios import load_scenarios
from needlesim.benchmark.scenarios import COMMON_CONDITIONS, HAND_DESIGNED, build_env
from needlesim.models.unicycle_needle import NeedleParams, State, rollout
from needlesim.planning.rrt import KinodynamicRRT, RRTConfig, VanillaRRT

KAPPA = COMMON_CONDITIONS["kappa"]
MAX_ITERATIONS = 20000
HAND_SEEDS = 30
RANDOM_SEEDS = 10
# A node counts as a heading discontinuity if the wrapped arrival-vs-departure
# gap exceeds this. Kinodynamic should sit at float noise (~1e-12); vanilla at
# order-1 radians. 1e-6 is comfortably between the two regimes.
HEADING_DISC_THRESHOLD = 1e-6

RANDOM_SCENARIOS_PATH = Path("experiments/random_scenarios.json")
DEFAULT_CSV_PATH = Path("experiments/results/benchmark_raw.csv")

PLANNERS = {"VanillaRRT": VanillaRRT, "KinodynamicRRT": KinodynamicRRT}

CSV_COLUMNS = [
    "scenario_name",
    "scenario_kind",  # hand | random
    "planner",
    "seed",
    "success",
    "n_iterations",
    "n_nodes",
    "wall_time_s",
    "path_cost_mm",
    "endpoint_error_mm",
    "heading_disc_max_rad",
    "heading_disc_mean_rad",
    "heading_disc_count",
]


@dataclass
class RunRecord:
    """One row of the raw CSV. Metric fields are None on failure (a failure is
    still a row -- an empty metric cell is data)."""

    scenario_name: str
    scenario_kind: str
    planner: str
    seed: int
    success: bool
    n_iterations: int
    n_nodes: int
    wall_time_s: float
    path_cost_mm: float | None
    endpoint_error_mm: float | None
    heading_disc_max_rad: float | None
    heading_disc_mean_rad: float | None
    heading_disc_count: int | None


def make_config(seed: int) -> RRTConfig:
    """Common conditions, IDENTICAL for every run. Comparing planners under
    different margins/velocities/tolerances would be invalid."""
    return RRTConfig(
        max_iterations=MAX_ITERATIONS,
        goal_tolerance=COMMON_CONDITIONS["goal_tolerance"],
        step_dt=COMMON_CONDITIONS["step_dt"],
        edge_velocity=COMMON_CONDITIONS["edge_velocity"],
        margin=COMMON_CONDITIONS["margin"],
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Metrics. All take the RRTResult, the config (for the fixed edge geometry),
# and the real model params. None of them mutate anything.
# ---------------------------------------------------------------------------


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _expand_controls(controls, n_steps: int):
    """RRTResult.controls holds ONE Control per EDGE (reconstruct appends
    node.control once per node). Each edge is that control repeated
    n_steps_per_extend times at step_dt -- expand before rolling."""
    return [c for c in controls for _ in range(n_steps)]


def path_cost_mm(result, cfg: RRTConfig) -> float:
    """Total executed path length. Speed is constant (edge_velocity), and each
    edge is n_steps_per_extend steps of edge_velocity*step_dt, so each edge's
    arc length is n_steps_per_extend*edge_velocity*step_dt and the total is
    that times the number of edges. RRTResult carries no cost field, hence this
    closed form rather than a stored value."""
    edge_len = cfg.n_steps_per_extend * cfg.edge_velocity * cfg.step_dt
    return len(result.controls) * edge_len


def endpoint_error_mm(
    result, start: State, goal: State, cfg: RRTConfig, params: NeedleParams
) -> float:
    """Distance from goal if you EXECUTED the returned controls on the needle.
    Expand the per-edge controls to per-step, roll from the start through the
    real model, measure |final - goal|. Kinodynamic's controls were generated
    by this same model so it lands within tolerance; vanilla's were generated
    with kappa=0, so under the real curvature it ends up elsewhere."""
    steps = _expand_controls(result.controls, cfg.n_steps_per_extend)
    final = rollout(start, steps, cfg.step_dt, params)[-1]
    return math.hypot(final.x - goal.x, final.y - goal.y)


def heading_discontinuity(
    result, cfg: RRTConfig, params: NeedleParams
) -> tuple[float, float, int]:
    """(max, mean, count-over-threshold) of heading discontinuity at interior
    path nodes. Geometric-infeasibility measure carrying the benchmark's claim.

    For each interior node k (has both an incoming edge k-1 and an outgoing
    edge k):
      arrival  = theta after rolling edge k-1's controls from path[k-1]
                 ([controls[k-1]] * n_steps_per_extend at step_dt)
      departure = path[k].theta as stored
      disc = |wrap(arrival - departure)|

    Kinodynamic: the stored node state IS that rollout's endpoint (same model,
    same parent), so arrival == departure and disc ~0. Vanilla: extend
    synthesises a heading pointing at the sample before rolling, so the stored
    departure heading jumps away from the arrival heading at every node."""
    path = result.path
    controls = result.controls
    n_steps = cfg.n_steps_per_extend

    discs = []
    # interior nodes: k from 1 to len(path)-2 (the goal node, k=len-1, has no
    # outgoing edge and so no departure heading to compare against).
    for k in range(1, len(path) - 1):
        arrival = rollout(
            path[k - 1], [controls[k - 1]] * n_steps, cfg.step_dt, params
        )[-1].theta
        departure = path[k].theta
        discs.append(abs(_wrap(arrival - departure)))

    if not discs:  # a single-edge path has no interior nodes
        return 0.0, 0.0, 0
    max_d = max(discs)
    mean_d = sum(discs) / len(discs)
    count = sum(1 for d in discs if d > HEADING_DISC_THRESHOLD)
    return max_d, mean_d, count


# ---------------------------------------------------------------------------
# Running one cell and the whole grid
# ---------------------------------------------------------------------------


def run_one(scenario, kind: str, planner_name: str, seed: int) -> RunRecord:
    """Execute one (scenario, planner, seed) run and compute its metrics. A
    fresh env is built per run so a baked SDF is never shared."""
    env = build_env(scenario)
    params = NeedleParams(kappa=KAPPA)
    cfg = make_config(seed)
    planner = PLANNERS[planner_name](env, params, cfg)

    t0 = time.perf_counter()
    result = planner.plan(scenario.start, scenario.goal)
    wall = time.perf_counter() - t0

    base = dict(
        scenario_name=scenario.name,
        scenario_kind=kind,
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
    """Yield (scenario, kind, planner_name, seed) for the whole 840-run grid."""
    for scenario in HAND_DESIGNED:
        for planner_name in PLANNERS:
            for seed in range(HAND_SEEDS):
                yield scenario, "hand", planner_name, seed
    _, random_scenarios = load_scenarios(RANDOM_SCENARIOS_PATH)
    for scenario in random_scenarios:
        for planner_name in PLANNERS:
            for seed in range(RANDOM_SEEDS):
                yield scenario, "random", planner_name, seed


def _existing_keys(csv_path: Path) -> set[tuple[str, str, str]]:
    """(scenario_name, planner, seed) already recorded, so a resumed run skips
    them. seed kept as its CSV string to compare against str(seed)."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as f:
        return {
            (row["scenario_name"], row["planner"], row["seed"])
            for row in csv.DictReader(f)
        }


def run_harness(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    """Run the grid, appending one row per run. Resumable: on startup, any
    (scenario, planner, seed) already in csv_path is skipped."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(csv_path)
    grid = list(iter_grid())
    total = len(grid)
    if done:
        print(f"Resuming: {len(done)} of {total} runs already recorded.")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for i, (scenario, kind, planner_name, seed) in enumerate(grid, 1):
            key = (scenario.name, planner_name, str(seed))
            if key in done:
                continue
            record = run_one(scenario, kind, planner_name, seed)
            writer.writerow(asdict(record))
            f.flush()  # append-as-you-go: survive a crash, show live progress
            flag = "ok " if record.success else "FAIL"
            print(
                f"[{i:>3}/{total}] {flag} {scenario.name:<20} "
                f"{planner_name:<15} seed={seed:<2} {record.wall_time_s:6.2f}s"
            )
    print(f"Done. Raw records -> {csv_path}")

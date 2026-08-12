"""Random benchmark scenario generator (PART 2).

Produces `Scenario` objects of the SAME type as the hand-designed four, so the
harness treats both uniformly. The 30 random scenarios are reported SEPARATELY
from the hand-designed ones: those are illustrative (each gets a figure), these
are statistical (an aggregate table only). Mixing them would bury the
illustrative cases in an average and bias the aggregate, since the four were
chosen precisely because they are interesting.

THE DISTRIBUTION (implemented exactly as stated; this docstring IS the spec):

  - Workspace, resolution, start, goal: identical to the hand-designed
    scenarios (150x150mm @ 0.5mm). Start and goal are FIXED across all 30
    (bottom-centre insertion heading up -> top-centre), so ONLY the obstacles
    vary. Fixed poses are FIXED_START / FIXED_GOAL below.
  - Obstacle count ~ Uniform{2..7} (inclusive).
  - Each obstacle is a circle OR an axis-aligned rectangle with equal
    probability.
      * Circle: radius ~ Uniform(5, 25) mm; centre ~ Uniform over the
        workspace.
      * Rect: width, height ~ Uniform(5, 40) mm; lower-left corner ~ Uniform
        over the workspace (so a rect may extend past an edge -- add_rect
        clips it, which yields legitimate edge-hugging strips).
  - OVERLAPPING OBSTACLES ARE PERMITTED. Two overlapping discs form a peanut
    shape; this is intentional variety, not a defect. No disjointness is
    assumed or enforced.
  - Any obstacle that would place the fixed start or goal in collision
    INCLUDING the 2mm margin is resampled. This is NOT solvability screening;
    it only excludes degenerate cases where the start pose itself is in
    collision, which is not a planning problem at all.
  - NO solvability screening otherwise. Some scenarios will be unsolvable;
    that is accepted and failures are reported honestly. Screening would need
    a reference planner to certify solvability, which introduces its own bias.

Rotation is NOT supported (add_rect is axis-aligned only) and is not added --
elongated rects at random positions already give vertical strips, horizontal
strips and near-squares, which is adequate variety.

REPRODUCIBILITY: generation is seeded -- the same seed yields the same 30
scenarios. Generate once and PERSIST (save_scenarios -> JSON), then run the
benchmark against the loaded set, so it stays re-runnable against an identical
set even if this generator code later changes. The seed is stored in the file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from needlesim.benchmark.scenarios import (
    COMMON_CONDITIONS,
    HEIGHT,
    RESOLUTION,
    WIDTH,
    Circle,
    Rect,
    Scenario,
)
from needlesim.models.unicycle_needle import State

MARGIN = COMMON_CONDITIONS["margin"]

# Fixed poses for every random scenario: bottom-centre insertion heading up,
# goal top-centre. Centred (vs an off-axis corridor) so uniformly-placed
# obstacles are equally likely to obstruct the nominal traversal.
_UP = math.pi / 2
FIXED_START = State(75.0, 20.0, _UP)
FIXED_GOAL = State(75.0, 130.0, _UP)

N_SCENARIOS = 30
DEFAULT_SEED = 20260811  # recorded; same seed -> same 30 scenarios

# Distribution parameters (see module docstring -- do not tune to make numbers
# look better; the distribution is the experimental design).
MIN_OBSTACLES, MAX_OBSTACLES = 2, 7
CIRCLE_R_MIN, CIRCLE_R_MAX = 5.0, 25.0
RECT_SIDE_MIN, RECT_SIDE_MAX = 5.0, 40.0

# The reject test below is analytic (point-to-obstacle distance), but is_free
# reads a discretised SDF on a RESOLUTION grid. A one-cell buffer above the
# margin guarantees the analytic decision agrees with the baked SDF's
# is_free(margin) at start/goal, so no generated scenario trips the smoke test
# on a quantisation edge. It only makes the (already narrow) reject band a hair
# wider around the two fixed points; it does not alter the obstacle
# distribution elsewhere.
_REJECT_CLEARANCE = MARGIN + RESOLUTION


def _obstacle_clearance(obstacle, px: float, py: float) -> float:
    """Distance from point (px, py) to the obstacle's surface, >= 0 outside and
    0 when inside (the inside case does not need a magnitude here -- any inside
    point fails the >= margin test regardless)."""
    if isinstance(obstacle, Circle):
        return math.hypot(px - obstacle.cx, py - obstacle.cy) - obstacle.r
    # Rect: axis-aligned; distance to the box (0 if the point is inside).
    x0, x1 = sorted((obstacle.x0, obstacle.x1))
    y0, y1 = sorted((obstacle.y0, obstacle.y1))
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _clears_fixed_poses(obstacle) -> bool:
    """True if the obstacle keeps BOTH fixed poses clear by the reject
    clearance. Rejecting per-obstacle is sufficient for the union: if every
    obstacle is >= margin from a point, the nearest is too, so the baked SDF's
    clearance (a min over obstacles) is >= margin there."""
    return all(
        _obstacle_clearance(obstacle, p.x, p.y) >= _REJECT_CLEARANCE
        for p in (FIXED_START, FIXED_GOAL)
    )


def _sample_obstacle(rng: np.random.Generator):
    """One obstacle from the stated distribution (may collide with the fixed
    poses; the caller resamples until it does not)."""
    if rng.random() < 0.5:
        r = float(rng.uniform(CIRCLE_R_MIN, CIRCLE_R_MAX))
        cx = float(rng.uniform(0.0, WIDTH))
        cy = float(rng.uniform(0.0, HEIGHT))
        return Circle(cx, cy, r)
    w = float(rng.uniform(RECT_SIDE_MIN, RECT_SIDE_MAX))
    h = float(rng.uniform(RECT_SIDE_MIN, RECT_SIDE_MAX))
    x0 = float(rng.uniform(0.0, WIDTH))
    y0 = float(rng.uniform(0.0, HEIGHT))
    return Rect(x0, y0, x0 + w, y0 + h)


def _generate_one(rng: np.random.Generator, index: int, seed: int) -> Scenario:
    n = int(rng.integers(MIN_OBSTACLES, MAX_OBSTACLES + 1))  # inclusive upper
    obstacles = []
    for _ in range(n):
        obs = _sample_obstacle(rng)
        while not _clears_fixed_poses(obs):  # resample degenerate placements
            obs = _sample_obstacle(rng)
        obstacles.append(obs)
    return Scenario(
        name=f"random_{index:02d}",
        width=WIDTH,
        height=HEIGHT,
        resolution=RESOLUTION,
        obstacles=tuple(obstacles),
        start=FIXED_START,
        goal=FIXED_GOAL,
        description=f"random ({n} obstacles), gen seed {seed}",
    )


def generate_scenarios(
    seed: int = DEFAULT_SEED, n: int = N_SCENARIOS
) -> list[Scenario]:
    """Generate `n` random scenarios reproducibly from `seed`. Same seed and n
    -> identical list (the single rng stream is consumed in order)."""
    rng = np.random.default_rng(seed)
    return [_generate_one(rng, i, seed) for i in range(n)]


# ---------------------------------------------------------------------------
# Serialisation -- persist the set so the benchmark is re-runnable against an
# identical scenario set even if this generator changes.
# ---------------------------------------------------------------------------


def _obstacle_to_dict(obstacle) -> dict:
    if isinstance(obstacle, Circle):
        return {"type": "circle", "cx": obstacle.cx, "cy": obstacle.cy, "r": obstacle.r}
    return {
        "type": "rect",
        "x0": obstacle.x0,
        "y0": obstacle.y0,
        "x1": obstacle.x1,
        "y1": obstacle.y1,
    }


def _obstacle_from_dict(d: dict):
    if d["type"] == "circle":
        return Circle(d["cx"], d["cy"], d["r"])
    if d["type"] == "rect":
        return Rect(d["x0"], d["y0"], d["x1"], d["y1"])
    raise ValueError(f"unknown obstacle type: {d['type']!r}")


def _state_to_dict(s: State) -> dict:
    return {"x": s.x, "y": s.y, "theta": s.theta}


def _state_from_dict(d: dict) -> State:
    return State(d["x"], d["y"], d["theta"])


def scenario_to_dict(scenario: Scenario) -> dict:
    return {
        "name": scenario.name,
        "width": scenario.width,
        "height": scenario.height,
        "resolution": scenario.resolution,
        "obstacles": [_obstacle_to_dict(o) for o in scenario.obstacles],
        "start": _state_to_dict(scenario.start),
        "goal": _state_to_dict(scenario.goal),
        "description": scenario.description,
    }


def scenario_from_dict(d: dict) -> Scenario:
    return Scenario(
        name=d["name"],
        width=d["width"],
        height=d["height"],
        resolution=d["resolution"],
        obstacles=tuple(_obstacle_from_dict(o) for o in d["obstacles"]),
        start=_state_from_dict(d["start"]),
        goal=_state_from_dict(d["goal"]),
        description=d["description"],
    )


def save_scenarios(path, scenarios: list[Scenario], seed: int) -> None:
    """Persist scenarios + the seed that produced them to JSON."""
    payload = {
        "seed": seed,
        "n_scenarios": len(scenarios),
        "common_conditions": COMMON_CONDITIONS,
        "scenarios": [scenario_to_dict(s) for s in scenarios],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_scenarios(path) -> tuple[int, list[Scenario]]:
    """Load a persisted set. Returns (seed, scenarios)."""
    payload = json.loads(Path(path).read_text())
    scenarios = [scenario_from_dict(d) for d in payload["scenarios"]]
    return payload["seed"], scenarios

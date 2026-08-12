"""Benchmark scenarios as DATA.

A `Scenario` is a plain, frozen, serialisable description of one planning
problem: world size, obstacles, start, goal. It is deliberately NOT a function
that builds an environment -- `build_env` is the single place that touches
`GridEnvironment`. Two reasons this matters:

  1. The random generator (PART 2) must emit the SAME type, so the harness can
     treat hand-designed and random scenarios uniformly.
  2. Keeping scenarios as data (no live env, no planner) makes them trivially
     serialisable and comparable, and keeps this module free of any dependency
     on the planners it will be used to benchmark.

Every hand-designed scenario below shares the COMMON conditions
(150x150mm world, 0.5mm resolution, kappa=1/50). The planner-facing knobs
(margin, goal_tolerance, step_dt, edge_velocity, budget) live in the harness's
config, NOT here -- a scenario is geometry, not planner tuning. They are echoed
in COMMON_CONDITIONS for reference and so the verify script can pin them.

MIXED REPORTING STRUCTURE (deliberate -- the benchmark write-up should say so
rather than leave a reader wondering why one row has no variance): `open`,
`constrained_passage` and `cluttered` sit in a regime where kinodynamic
succeeds MOST-but-not-always (or, for `open`, always -- it is the efficiency
baseline), so they yield success RATES and cost/iteration distributions.
`target_behind` is instead a PASS/FAIL scenario, tuned to the symmetric point
where kinodynamic is a stable 0/10 and vanilla a stable 10/10; its row is a
binary by design, not a degenerate rate. The two kinds are both meaningful,
but they are read differently -- see each scenario's comment for why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from needlesim.environments.grid_environment import GridEnvironment
from needlesim.models.unicycle_needle import State

# --- shared conditions, identical for every hand-designed scenario ----------
# Geometry only. Planner-facing knobs (margin=2.0, goal_tolerance=3.0,
# step_dt=0.05, edge_velocity=5.0, kappa=1/50) are echoed here for reference
# but belong to the harness config, not the scenario data.
WIDTH = 150.0
HEIGHT = 150.0
RESOLUTION = 0.5

COMMON_CONDITIONS = dict(
    width=WIDTH,
    height=HEIGHT,
    resolution=RESOLUTION,
    kappa=1.0 / 50.0,
    margin=2.0,
    goal_tolerance=3.0,
    step_dt=0.05,
    edge_velocity=5.0,
)


# --- obstacle primitives ----------------------------------------------------


@dataclass(frozen=True)
class Circle:
    """A filled circular obstacle (mm)."""

    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Rect:
    """A filled axis-aligned rectangular obstacle (mm). No rotation -- that is
    a deliberate accepted limitation of GridEnvironment.add_rect."""

    x0: float
    y0: float
    x1: float
    y1: float


Obstacle = Circle | Rect


# --- the scenario ------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One planning problem, as plain data.

    `obstacles` is a tuple (frozen -> hashable -> the whole Scenario is
    hashable and immutable) of Circle | Rect. `start` and `goal` are poses.
    `description` is one line stating what the scenario is meant to test.
    """

    name: str
    width: float
    height: float
    resolution: float
    obstacles: tuple[Obstacle, ...]
    start: State
    goal: State
    description: str


def build_env(scenario: Scenario) -> GridEnvironment:
    """Turn a Scenario into a baked GridEnvironment.

    The ONLY function in the benchmark package that touches GridEnvironment.
    Paints each obstacle, then bakes the SDF once.
    """
    env = GridEnvironment(
        width=scenario.width,
        height=scenario.height,
        resolution=scenario.resolution,
    )
    for obs in scenario.obstacles:
        if isinstance(obs, Circle):
            env.add_circle(obs.cx, obs.cy, obs.r)
        elif isinstance(obs, Rect):
            env.add_rect(obs.x0, obs.y0, obs.x1, obs.y1)
        else:  # pragma: no cover - guards against a new obstacle type
            raise TypeError(f"unknown obstacle type: {type(obs).__name__}")
    env.bake()
    return env


# --- the four hand-designed scenarios ---------------------------------------
#
# All 150x150mm @ 0.5mm. Circle radii kept in the anatomically-plausible
# 5-25mm band; rectangles may be elongated (vessels in-plane read as strips).
# The gap width / detour offset / clutter spacing are DESIGN VARIABLES to be
# tuned from the verify_scenarios table -- the values here are starting points,
# not claims of correct difficulty.

_UP = math.pi / 2

# 1. open -- single central obstacle, clear corridors either side. The
#    efficiency baseline. Exact values reused from verify_common_scenario.py.
OPEN = Scenario(
    name="open",
    width=WIDTH,
    height=HEIGHT,
    resolution=RESOLUTION,
    obstacles=(Circle(75.0, 75.0, 20.0),),
    start=State(30.0, 20.0, _UP),
    goal=State(30.0, 130.0, _UP),
    description="single central obstacle, clear corridors -- efficiency baseline",
)

# 2. constrained_passage -- a wall with a 16mm gap the needle must thread. Two
#    elongated rects leave a gap centred at x=75; the start is offset to x=41
#    so the needle cannot just drive straight up through it -- at R=50mm it
#    must curve over AND arrive heading-aligned to pass, whereas VanillaRRT
#    teleports its heading and threads trivially (10/10 at every setting).
#
#    TUNED (fine sweep, see experiments/results/scenario_tuning/): the binding
#    difficulty is APPROACH ANGLE (start_x), not gap width -- a 26mm gap fails
#    at an unfavourable offset and a 12mm gap succeeds at a favourable one. The
#    kinodynamic success band spans start_x 38-44 (70% / 60% / 90%), so
#    start_x=41 sits in the plateau interior rather than on the sharp lower
#    edge at start_x 35-38 (0% -> 70% over 3mm). Gap width only modulates
#    ITERATION cost within the solvable region (2301 -> 6482 iters as the gap
#    narrows 26 -> 12mm), which is why 16mm was chosen over something wider:
#    narrow enough to be iteration-expensive, wide enough to stay on the
#    plateau. Kinodynamic ~6/10 here; a rate-and-distribution scenario.
_GAP_HALF = 8.0  # half the 16mm wall-to-wall gap; gap centred at x=75
CONSTRAINED_PASSAGE = Scenario(
    name="constrained_passage",
    width=WIDTH,
    height=HEIGHT,
    resolution=RESOLUTION,
    obstacles=(
        Rect(0.0, 72.0, 75.0 - _GAP_HALF, 78.0),
        Rect(75.0 + _GAP_HALF, 72.0, 150.0, 78.0),
    ),
    start=State(41.0, 20.0, _UP),
    goal=State(75.0, 130.0, _UP),
    description=(
        "16mm gap approached off-axis (start_x=41) -- curvature must align "
        "before the wall; rate scenario, kino ~6/10, vanilla 10/10"
    ),
)

# 3. target_behind -- start heading straight at an obstacle; goal placed
#    directly BEHIND it (symmetric to the start about the obstacle centre), so
#    the needle must commit to a detour rather than driving at it. Tests
#    lookahead vs greed.
#
#    A DELIBERATE PASS/FAIL SCENARIO, not tuned into the 50-90% band -- and the
#    reason is recorded here so the binary row is not read as a mistake. The
#    fine sweep (experiments/results/scenario_tuning/) found the difficulty
#    peak is a ~4-degree 0%-success CORE (goal angles 88/90/92 deg all 0/10),
#    and the only in-band cells (86 deg at 70%, 94 deg at 90%) are each
#    isolated, flanked by 100% and 0% neighbours. Tuning to a middle would put
#    the scenario on a 2-degree knife edge where a re-seed or a 1mm nudge
#    swings it to 0% or 100%. The symmetric point is stable and defensible
#    instead: kinodynamic 0/10, vanilla 10/10.
#
#    MECHANISM (confirmed by the path figures): kinodynamic commits to whichever
#    side the goal leans and swings ~30mm clear of the obstacle -- right at
#    84 deg, left at 94 deg -- even when the goal is only 4mm off centre. At
#    exact symmetry neither route dominates, the tree's effort splits between
#    the two equally-costly detours, and nothing completes in budget. Vanilla
#    solves it regardless because it teleports its heading and never commits.
#
#    Goal (75, 133) is the 90-degree position on the ~58.3mm radius the earlier
#    goal sat on -- directly above the obstacle centre (75, 75), mirror of the
#    start (75, 20).
TARGET_BEHIND = Scenario(
    name="target_behind",
    width=WIDTH,
    height=HEIGHT,
    resolution=RESOLUTION,
    obstacles=(Circle(75.0, 75.0, 20.0),),
    start=State(75.0, 20.0, _UP),
    goal=State(75.0, 133.0, _UP),
    description=(
        "goal directly behind a head-on obstacle (symmetric) -- PASS/FAIL "
        "scenario: kino 0/10 (effort splits between two equal detours), "
        "vanilla 10/10"
    ),
)

# 4. cluttered -- 6 small scattered obstacles, the most anatomically
#    representative case (multiple vessels). Radii 9-13mm; centres spaced so no
#    two merge into one effective blob. Start/goal on a diagonal so the path
#    must weave rather than run one clear corridor.
CLUTTERED = Scenario(
    name="cluttered",
    width=WIDTH,
    height=HEIGHT,
    resolution=RESOLUTION,
    obstacles=(
        Circle(52.0, 55.0, 12.0),
        Circle(98.0, 50.0, 10.0),
        Circle(60.0, 98.0, 11.0),
        Circle(105.0, 100.0, 13.0),
        Circle(40.0, 118.0, 9.0),
        Circle(120.0, 78.0, 10.0),
    ),
    start=State(30.0, 20.0, _UP),
    goal=State(120.0, 135.0, _UP),
    description="6 scattered small obstacles -- weave through clutter",
)


HAND_DESIGNED: tuple[Scenario, ...] = (
    OPEN,
    CONSTRAINED_PASSAGE,
    TARGET_BEHIND,
    CLUTTERED,
)

# Name -> scenario, for the harness and verify script to look up by name.
SCENARIOS: dict[str, Scenario] = {s.name: s for s in HAND_DESIGNED}

"""Enlarged-scale (SECONDARY experiment) scenarios: the four hand-designed
scenarios geometrically scaled x10/3 (150x150mm -> 500x500mm).

WHY THIS EXISTS (and why it is SEPARATE from scenarios.py): the primary
benchmark excludes RRTStar because at realistic curvature (R = 1/kappa = 50mm)
in the 150mm anatomical workspace it cannot connect poses -- a recorded finding
(see docs/roadmap.md, Task 3.6). This experiment tests whether that exclusion
is a consequence of SCALE rather than of the algorithm being useless: enlarge
the scene x10/3 while keeping kappa fixed (R=50mm becomes 10 turning radii
across the workspace, versus 3 at primary scale), and see whether all three
planners function. Its results must NEVER be pooled with the primary ones --
different workspace, different question -- hence a separate module and a
separate CSV.

WHAT SCALES AND WHAT DOES NOT (the one asymmetry that matters):
  - SCALED x10/3: workspace, obstacle geometry, start, goal, and (in the
    harness) goal_tolerance. The whole scene grows uniformly.
  - resolution SCALED x10/3 too (0.5 -> 1.667mm): this keeps the grid at
    300x300 cells, so bake time and memory match the primary benchmark exactly
    rather than exploding to 1000x1000 at a fixed 0.5mm. Coarser resolution
    makes the is_arc_free spacing invariant EASIER, not harder
    (edge_velocity*step_dt = 0.25 <= 0.5*1.667 = 0.833), and the collision
    granularity stays a fixed fraction of scene features, so geometric fidelity
    relative to the obstacles is preserved.
  - NOT scaled: margin (2.0mm). It is the needle's physical width, which does
    not grow with the workspace. This is the single quantity that must stay
    fixed; it lives in the harness config, echoed in SCALED_COMMON_CONDITIONS.
  - NOT scaled: kappa (1/50) and n_steps_per_extend (edge length v*dt*n = 5mm).
    The needle is unchanged; only the scene grows. Keeping edge length AND
    turning radius fixed preserves the edge/R = 0.1 design ratio the roadmap is
    explicit about -- scaling n_steps_per_extend would change the planner's
    character mid-experiment.

The scenarios are produced by a pure geometric transform of the primary
HAND_DESIGNED tuple, so the two sets stay definitionally in lockstep: if a
primary scenario is retuned, its scaled twin follows automatically.
"""

from __future__ import annotations

from needlesim.benchmark.scenarios import (
    HAND_DESIGNED,
    Circle,
    Obstacle,
    Rect,
    Scenario,
)
from needlesim.models.unicycle_needle import State

# 150mm -> 500mm. Kept as a fraction (not 3.333) so scaled values are exact
# rationals of the originals rather than carrying rounding.
SCALE = 10.0 / 3.0


def _scale_state(s: State) -> State:
    """Scale position; heading is an angle and does NOT scale."""
    return State(s.x * SCALE, s.y * SCALE, s.theta)


def _scale_obstacle(o: Obstacle) -> Obstacle:
    if isinstance(o, Circle):
        return Circle(o.cx * SCALE, o.cy * SCALE, o.r * SCALE)
    if isinstance(o, Rect):
        return Rect(o.x0 * SCALE, o.y0 * SCALE, o.x1 * SCALE, o.y1 * SCALE)
    raise TypeError(f"unknown obstacle type: {type(o).__name__}")


def scale_scenario(sc: Scenario) -> Scenario:
    """Uniformly scale one Scenario's geometry x SCALE. Name is preserved (the
    scaled twin keeps its identity); description is prefixed so a stray row can
    never be mistaken for a primary-scale one."""
    return Scenario(
        name=sc.name,
        width=sc.width * SCALE,
        height=sc.height * SCALE,
        resolution=sc.resolution * SCALE,
        obstacles=tuple(_scale_obstacle(o) for o in sc.obstacles),
        start=_scale_state(sc.start),
        goal=_scale_state(sc.goal),
        description=f"SCALED x10/3: {sc.description}",
    )


# The four scaled scenarios, in the same order as HAND_DESIGNED.
SCALED_HAND_DESIGNED: tuple[Scenario, ...] = tuple(
    scale_scenario(sc) for sc in HAND_DESIGNED
)

SCALED_SCENARIOS: dict[str, Scenario] = {s.name: s for s in SCALED_HAND_DESIGNED}

# Planner-facing knobs for the enlarged scale, echoed for reference (they live
# in the harness config, not the scenario data -- a scenario is geometry). Only
# goal_tolerance scales; margin explicitly does not.
SCALED_COMMON_CONDITIONS = dict(
    kappa=1.0 / 50.0,
    margin=2.0,  # NOT scaled -- needle physical width
    goal_tolerance=3.0 * SCALE,  # 10.0mm -- scaled so the goal region keeps its
    # relative size to the scene
    step_dt=0.05,
    edge_velocity=5.0,
)

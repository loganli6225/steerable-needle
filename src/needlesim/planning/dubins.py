"""CCC Dubins steering for the bevel-tip needle (no straight segments).

THIS IS YOUR TASK 3.5 FILE. Interface and the three test classes are laid out;
the geometry is yours to implement (marked IMPLEMENT ME). Per the working
agreement, the Dubins math is load-bearing and yours; the tests are scaffolding
you may lean on.

WHY CCC-ONLY
------------
A bevel-tip needle always curves at +/- kappa; it cannot go straight. The
classical Dubins result says shortest curvature-bounded paths are CSC or CCC;
dropping the straight (S) segment leaves the CCC family: three arcs at radius
R = 1/kappa, alternating sense -- RLR or LRL. This is the documented model for
2D steerable-needle planning.

There is NO external reference solver you can diff against, so correctness
rests entirely on the three analytic test classes in tests/test_dubins.py.
The strongest of these rolls the returned controls through the Task 1 `step`
and checks the needle actually lands at the target: your trusted integrator is
the oracle for the untrusted geometry.

KEY GEOMETRIC FACTS (build tests around these)
----------------------------------------------
- R = 1/kappa is the fixed arc radius for every segment.
- An L arc turns left (b such that theta increases); an R arc turns right.
- A CCC path exists ONLY when the two poses are close enough: the distance
  between the relevant turning centres must be < 4R. Beyond that, return None
  -- this is not an edge case, it is central. RRT* handles None by not
  creating that edge.
- The middle arc bends opposite to the outer two. Its angle is fixed by the
  geometry once the outer arcs are chosen.

OUTPUT
------
A DubinsPath (below) carrying the CONTROL SEQUENCE (list[Control] for `step`)
plus the total length. Control sequence chosen over a geometric description so
execution (test 3) and RRT* edge-building are trivial: just feed the controls
to step/is_arc_free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from needlesim.models.unicycle_needle import Control, NeedleParams, State


@dataclass
class DubinsPath:
    """Result of a CCC steering query.

    controls: the per-step Control sequence that drives `step` from the start
              pose to (approximately) the goal pose. Feeding these to the Task 1
              model must land at the goal -- that is test 3.
    length:   total arc length [mm] (sum of the three arc lengths). This is what
              RRT* minimises.
    word:     "RLR" or "LRL" -- which CCC type won. Handy for debugging/plots.
    """

    controls: list[tuple[Control, float]]
    length: float
    word: str


def dubins_ccc(
    start: State,
    goal: State,
    params: NeedleParams,
    v: float,
    dt: float,
) -> DubinsPath | None:
    """Shortest CCC (arc-arc-arc) path from `start` to `goal`, or None.

    IMPLEMENT ME. Return None if no CCC path connects the poses (centres >= 4R
    apart, or the configuration is otherwise infeasible).

    Args:
        start, goal: oriented poses (x, y, theta).
        params: needle params; R = 1/params.kappa is the arc radius.
        v: insertion speed used to synthesise controls [mm/s].
        dt: per-step timestep; the control sequence is discretised at this dt,
            consistent with `step`. (Arc of angle phi at radius R has length
            R*phi; number of steps ~ R*phi / (v*dt).)

    Suggested structure:
        1. R = 1/params.kappa.
        2. Compute the two candidate words (RLR, LRL). For each:
             a. locate the outer turning centres from start and goal poses,
             b. check the 4R reachability condition; skip if it fails,
             c. solve for the three arc angles (closed form),
             d. total length = R * (phi1 + phi2 + phi3).
        3. Pick the shorter valid word. If neither is valid, return None.
        4. Discretise the chosen word into a Control sequence: for each arc,
           emit ceil(R*phi / (v*dt)) steps of Control(v, b=+1 or -1) with the
           sign matching L vs R. (Mind the LAST step of each arc: rounding up
           slightly overshoots; decide whether to trim dt on the final step or
           accept sub-step error. Document your choice -- test 3's tolerance
           depends on it.)

    Return DubinsPath(controls, length, word) or None.

    NOTE the b-sign convention: confirm against your model which sign of b makes
    theta INCREASE (left turn) vs decrease (right turn). Getting this backwards
    produces a path that is the mirror image of the correct one -- it will look
    plausible and fail test 3.
    """
    R = 1 / params.kappa
    candidates = []
    for arc_direction in [False, True]:
        cx_start, cy_start = _turning_center(pose=start, left=arc_direction, R=R)
        cx_goal, cy_goal = _turning_center(pose=goal, left=arc_direction, R=R)
        d = math.sqrt((cx_goal - cx_start) ** 2 + (cy_goal - cy_start) ** 2)
        if d >= 4 * R or d <= 1e-9:
            continue
        ux, uy = (cx_goal - cx_start) / d, (cy_goal - cy_start) / d
        Mx, My = cx_start + ux * (d / 2), cy_start + uy * (d / 2)
        h = math.sqrt(max(0.0, (2 * R) ** 2 - (d / 2) ** 2))
        for side in [1, -1]:
            C2x, C2y = Mx + side * h * (-uy), My + side * h * (ux)
            t1 = ((cx_start + C2x) / 2, (cy_start + C2y) / 2)
            t2 = ((C2x + cx_goal) / 2, (C2y + cy_goal) / 2)
            phi1 = _arc_angle(
                (cx_start, cy_start), (start.x, start.y), t1, left=arc_direction
            )
            phi2 = _arc_angle((C2x, C2y), t1, t2, left=not arc_direction)
            phi3 = _arc_angle(
                (cx_goal, cy_goal), t2, (goal.x, goal.y), left=arc_direction
            )
            length = R * (phi1 + phi2 + phi3)
            controls = []
            for phi, direction in [
                (phi1, arc_direction),
                (phi2, not arc_direction),
                (phi3, arc_direction),
            ]:
                exact_len = R * phi
                n = max(1, math.ceil(exact_len / (v * dt)))
                full = n - 1
                last_dt = (exact_len - full * v * dt) / v
                seg = [(Control(v, b=+1 if direction else -1), dt)] * full
                seg.append((Control(v, b=+1 if direction else -1), last_dt))
                controls += seg
            candidates.append(
                DubinsPath(
                    controls=controls,
                    length=length,
                    word="LRL" if arc_direction else "RLR",
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.length)


# Optional helpers you will likely want (implement as needed):


def _turning_center(pose: State, left: bool, R: float) -> tuple[float, float]:
    """Centre of the radius-R circle the needle rides for an L (or R) turn
    from `pose`. Left turn: centre is 90deg to the LEFT of heading. IMPLEMENT."""
    if left:
        direction = pose.theta + (math.pi / 2)
    else:
        direction = pose.theta - (math.pi / 2)
    cx = pose.x + (R * math.cos(direction))
    cy = pose.y + (R * math.sin(direction))
    return cx, cy


def _arc_angle(
    center: tuple[float, float],
    p_from: tuple[float, float],
    p_to: tuple[float, float],
    left: bool,
) -> float:
    """Swept angle [radians, >= 0] of an arc around `center`, from `p_from`
    to `p_to`, turning left (CCW) or right (CW).

    Both points lie on the circle. Returns the POSITIVE angle swept in the
    given turn direction -- always in [0, 2*pi). Multiply by R to get arc
    length.

    Method: bearing of each point from the centre via atan2, take the
    difference, then normalise into [0, 2*pi) IN THE TURN DIRECTION:
      - left (CCW): angle increases, so wrap (a1 - a0) into [0, 2*pi)
      - right (CW): angle decreases, so wrap (a0 - a1) into [0, 2*pi)
    """
    angle_1 = math.atan2(p_from[1] - center[1], p_from[0] - center[0])
    angle_2 = math.atan2(p_to[1] - center[1], p_to[0] - center[0])
    d_theta = angle_2 - angle_1
    if left:
        d_theta = d_theta % (2 * math.pi)
    else:
        d_theta = (-d_theta) % (2 * math.pi)
    return d_theta

"""Dubins steering for the bevel-tip needle: CCC (arc-only) and full CSC+CCC.

TASK 3.6. The CCC half is DONE (yours, from Task 3.5). What is new here is the
CSC family -- the four words with a STRAIGHT middle segment -- and a combined
`dubins_full` that returns the shortest over all six words.

WHY STRAIGHT SEGMENTS, AFTER ALL
--------------------------------
Task 3.5 deliberately excluded them: a bevel-tip needle always curves at
+/-kappa, so CCC (three arcs) looked like the honest model. At the fictional
curvature used for early testing (R=5mm in a 100mm world) that worked.

At REALISTIC curvature it does not. The literature puts bevel-tip radii of
curvature at roughly 40-170mm (optimised tip designs reach <50mm; plain bevel
tips are often ~160mm or negligible). At R=50mm in a 150mm workspace, every
CCC connection between two nearby poses is a giant loop comparable to the
whole world, so nearly all of them collide.

The needle CAN travel effectively straight, by DUTY CYCLING: alternating
b=+1/-1 gives kappa_eff = kappa*(2p-1), so p=0.5 is a straight centerline.
This is a real, physically available capability, and it is already verified in
this codebase (tests/test_needle_model.py :: test_duty_cycle_scales_curvature).
Excluding S segments made the PLANNER more restrictive than the HARDWARE, so
CSC is the honest connection primitive to expose.

WHAT THE S SEGMENTS DID AND DID NOT FIX (measured -- Task 3.6 finding)
---------------------------------------------------------------------
They fixed REACHABILITY, not TRACTABILITY. CSC makes almost any two poses
connectable (dubins_full rarely returns None), but it did NOT rescue RRT*:
on the common scenario (R=50mm, 150x150mm world, r=20mm obstacle, margin 2mm)
RRT* built ~155 nodes in 10,000 iterations (~98.5% sample rejection) and
failed to reach the goal -- IDENTICAL with CCC-only and with full Dubins
(verified: scripts/verify_common_scenario.py). The word family was never the
binding constraint: at R=50mm even a full-Dubins connection between two poses
with mismatched headings needs most of a 314mm turning circle to fix heading,
and those loops sweep the workspace and collide. That is a STRUCTURAL limit of
RRT*'s exact-pose-to-pose requirement at anatomical scale, not a word-family
gap -- see docs/roadmap.md (Task 3.6) for the full finding and the revised
benchmark scope (RRT* moves to a SECONDARY, enlarged-workspace role).

MODELLING ASSUMPTION TO STATE IN THE WRITEUP
--------------------------------------------
An S segment represents duty-cycled motion at p=0.5, in the idealised
(infinite cycling frequency) limit. The true tip path is a zigzag of tiny
alternating arcs that oscillates about the straight centerline; amplitude
shrinks with cycling frequency. Planning at the centerline level is standard;
the sub-millimetre execution oscillation, and the slightly wider corridor it
sweeps, are deferred to the control layer (and partly absorbed by `margin`).

WHAT LIVES HERE
---------------
    dubins_ccc(...)   CCC only (RLR, LRL). Now a thin wrapper -- KEPT so the
                      CCC-only vs full-Dubins comparison stays runnable. That
                      comparison is a genuine result: CCC-only fails at
                      realistic curvature.
    dubins_full(...)  All six words. This is what the planners should use.

VERIFIED GEOMETRIC FACTS (checked numerically; build on these)
--------------------------------------------------------------
- CSC words: LSL, RSR (both outer arcs the same sense) and LSR, RSL (opposite
  senses).
- SAME-SENSE (LSL / RSR) uses the EXTERNAL tangent between the two turning
  circles. Straight length = d, the centre-to-centre distance. Always exists
  (no reachability condition).
- OPPOSITE-SENSE (LSR / RSL) uses the INTERNAL (crossing) tangent. It exists
  only when d >= 2R; straight length = sqrt(d^2 - 4R^2). Return no candidate
  when d < 2R.
- Unlike CCC, CSC has NO 4R limit -- distant poses are always connectable via
  a long S. Consequence: dubins_full almost never returns None. This fixes
  REACHABILITY (the None-rate), NOT RRT*'s sample rejection -- see the
  measured Task 3.6 finding above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from needlesim.models.unicycle_needle import Control, NeedleParams, State


@dataclass
class DubinsPath:
    """Result of a steering query.

    controls: per-step (Control, dt) pairs driving `step` from start to goal.
    length:   total path length [mm] -- what RRT* minimises.
    word:     one of RLR, LRL (CCC) or LSL, RSR, LSR, RSL (CSC).
    """

    controls: list[tuple[Control, float]]
    length: float
    word: str


# ---------------------------------------------------------------------------
# EXISTING, DONE: helpers from Task 3.5. Unchanged -- CSC reuses both.
# ---------------------------------------------------------------------------


def _turning_center(pose: State, left: bool, R: float) -> tuple[float, float]:
    """Centre of the radius-R circle the needle rides for an L (or R) turn."""
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
    """Swept angle [rad, in [0, 2pi)] around `center`, from `p_from` to `p_to`,
    turning left (CCW) or right (CW)."""
    angle_1 = math.atan2(p_from[1] - center[1], p_from[0] - center[0])
    angle_2 = math.atan2(p_to[1] - center[1], p_to[0] - center[0])
    d_theta = angle_2 - angle_1
    if left:
        d_theta = d_theta % (2 * math.pi)
    else:
        d_theta = (-d_theta) % (2 * math.pi)
    return d_theta


# ---------------------------------------------------------------------------
# NEW: discretisation helpers. Factored out because CSC needs to emit STRAIGHT
# runs too, and both families share the exact-length trimming rule.
# ---------------------------------------------------------------------------


def _discretise_arc(phi: float, R: float, left: bool, v: float, dt: float) -> list:
    """(Control, dt) pairs tracing an arc of angle `phi` at radius R.

    IMPLEMENT ME (mostly a lift from your CCC code). Same exact-landing rule
    as Task 3.5: n-1 steps at full dt plus one trimmed final step, so the arc
    length is exact rather than ceil-overshooting. Reuse that logic verbatim.
    """
    exact_len = R * phi
    if exact_len <= 0:
        return []
    n = max(1, math.ceil(exact_len / (v * dt)))
    full = n - 1
    last_dt = (exact_len - full * v * dt) / v
    seg = [(Control(v, b=+1 if left else -1), dt)] * full
    seg.append((Control(v, b=+1 if left else -1), last_dt))
    return seg


def _discretise_straight(length: float, v: float, dt: float) -> list:
    """(Control, dt) pairs tracing a STRAIGHT run of `length` mm.

    IMPLEMENT ME. This is the duty-cycle idealisation made concrete.

    Decide and DOCUMENT which representation you emit, because the choice has
    consequences the writeup should own:

    (a) IDEALISED CENTERLINE -- alternate b=+1/-1 in equal measure at the
        step level (p=0.5), so the modelled path wiggles at the discretisation
        scale but its net heading change is zero. Feeding these to `step`
        reproduces the straight centerline to within the wiggle amplitude.
        Honest about the mechanism; introduces small position error vs a true
        line.

    (b) EXPLICIT ZERO-CURVATURE -- emit controls that `step` integrates as a
        true straight line (e.g. by handing this segment kappa=0 params, the
        same trick VanillaRRT uses). Exactly straight; hides the mechanism.

    Whichever you choose, the LENGTH accounting must stay exact (same trimmed
    final step rule as the arcs), and `test_controls_land_at_goal` will hold
    you to it: rolling the returned controls through the Task 1 model must
    land at the goal. Option (a) will need a looser tolerance than (b); if you
    pick (a), work out the wiggle amplitude and set the tolerance from it
    rather than by trial and error.
    """
    if length <= 0:
        return []
    n = math.ceil(length / (v * dt))
    if n % 2 == 0:
        n += 1
    true_dt = length / (v * n)
    first_half_seg = [(Control(v, b=-1), true_dt/2)]
    middel_segs = [(Control(v, b=+1), true_dt), (Control(v, b=-1), true_dt)] * ((n-1)//2)
    last_half_seg = [(Control(v, b=+1), true_dt/2)]
    seg = first_half_seg + middel_segs + last_half_seg
    return seg


# ---------------------------------------------------------------------------
# NEW: the CSC family. THIS IS THE LOAD-BEARING GEOMETRY -- yours.
# ---------------------------------------------------------------------------


def _csc_candidate(
    start: State,
    goal: State,
    R: float,
    first_left: bool,
    last_left: bool,
    v: float,
    dt: float,
) -> DubinsPath | None:
    """One CSC word: arc -> straight -> arc. Returns None if infeasible.

    IMPLEMENT ME. `first_left`/`last_left` select the word:
        (True,  True)  -> LSL      (False, False) -> RSR
        (True,  False) -> LSR      (False, True)  -> RSL

    Structure (all four verified numerically, see module docstring):

    1. c1 = _turning_center(start, first_left, R)
       c3 = _turning_center(goal,  last_left,  R)
       d     = distance(c1, c3)
       theta = atan2(c3.y - c1.y, c3.x - c1.x)

    2. SAME SENSE (first_left == last_left) -- external tangent:
         The straight segment is parallel to the c1->c3 line, offset
         perpendicular by R. Both tangent points use the SAME offset
         direction from their respective centres:
             off = theta - pi/2   for L arcs
             off = theta + pi/2   for R arcs
             t1 = c1 + R*(cos off, sin off)
             t2 = c3 + R*(cos off, sin off)
             straight = d
         Always feasible.

       OPPOSITE SENSE (first_left != last_left) -- internal tangent:
             if d < 2R: return None            <-- the only CSC infeasibility
             straight = sqrt(d^2 - 4R^2)
             alpha    = atan2(2R, straight)
             base = theta + alpha  if first arc is L, else theta - alpha
             off  = base - pi/2    if first arc is L, else base + pi/2
             t1 = c1 + R*(cos off, sin off)
             t2 = c3 - R*(cos off, sin off)     <-- note the MINUS: the second
                                                    tangent point is on the
                                                    opposite side, which is
                                                    what "crossing" means
       Sanity check while implementing: |t1 - t2| must equal `straight`, and
       both tangent points must sit exactly R from their own centre. Assert
       these once during development -- they catch sign errors instantly.

    3. phi1 = _arc_angle(c1, start_xy, t1, first_left)
       phi3 = _arc_angle(c3, t2, goal_xy,  last_left)
       length = R*(phi1 + phi3) + straight

    4. controls = _discretise_arc(phi1, ...) + _discretise_straight(straight,
       ...) + _discretise_arc(phi3, ...)

    5. Return DubinsPath(controls, length, word) with word built from the two
       flags, e.g. ("L" if first_left else "R") + "S" + ("L" if last_left
       else "R").
    """
    c1_x, c1_y = _turning_center(start, first_left, R)
    c3_x, c3_y = _turning_center(goal, last_left, R)
    d = math.sqrt((c1_x - c3_x)**2 + (c1_y - c3_y)**2)
    theta = math.atan2(c3_y - c1_y, c3_x - c1_x)

    if first_left == last_left:
        off = theta - math.pi/2 if first_left else theta + math.pi/2
        t1_x, t1_y = c1_x + R*(math.cos(off)), c1_y + R*(math.sin(off))
        t2_x, t2_y = c3_x + R*(math.cos(off)), c3_y + R*(math.sin(off))
        straight = d
    else:
        if d < 2 * R:
            return None
        straight = math.sqrt(d*d - 4*R*R)
        alpha = math.atan2(2*R, straight)
        base = theta + alpha if first_left else theta - alpha
        off = base - math.pi/2 if first_left else base + math.pi/2
        t1_x, t1_y = c1_x + R*math.cos(off), c1_y + R*math.sin(off)
        t2_x, t2_y = c3_x - R*math.cos(off), c3_y - R*math.sin(off)

    phi1 = _arc_angle((c1_x, c1_y), (start.x, start.y), (t1_x, t1_y), first_left)
    phi3 = _arc_angle((c3_x, c3_y), (t2_x, t2_y), (goal.x, goal.y), last_left)
    length = R*(phi1 + phi3) + straight

    controls = _discretise_arc(phi1, R, first_left, v, dt) + _discretise_straight(straight, v, dt) + _discretise_arc(phi3, R, last_left, v, dt)
    word = ("L" if first_left else "R") + "S" + ("L" if last_left else "R")
    return DubinsPath(controls=controls, length=length, word=word)

# ---------------------------------------------------------------------------
# The two public entry points.
# ---------------------------------------------------------------------------


def dubins_ccc(
    start: State, goal: State, params: NeedleParams, v: float, dt: float
) -> DubinsPath | None:
    """Shortest CCC (arc-arc-arc) path, or None. ARC-ONLY -- no straight.

    KEPT DELIBERATELY, not dead code: `dubins_full` vs `dubins_ccc` is a
    runnable comparison, and the finding is that BOTH collapse identically at
    realistic curvature (~98.5% RRT* sample rejection at R=50mm in a 150mm
    world, same node count either way) -- reproducing that equality is the
    point, since it shows the word family was never the binding constraint.

    KEEP YOUR TASK 3.5 BODY HERE UNCHANGED. If you factored the discretisation
    into _discretise_arc, call it here too so both families trim identically.
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


def dubins_full(
    start: State, goal: State, params: NeedleParams, v: float, dt: float
) -> DubinsPath | None:
    """Shortest path over ALL SIX words: LSL, RSR, LSR, RSL, RLR, LRL.

    IMPLEMENT ME (small, once the pieces above exist):
        1. candidates = []
        2. for (first_left, last_left) in the four CSC combinations:
               c = _csc_candidate(...); if c: candidates.append(c)
        3. c = dubins_ccc(...); if c: candidates.append(c)
        4. return min(candidates, key=length) or None if empty.

    NOTE: with CSC available this essentially never returns None -- any two
    poses connect via turn/straight/turn. The planners' None-handling paths
    stay correct but become rare. That fixes REACHABILITY only: measured, it
    did NOT lift RRT*'s sample acceptance at realistic curvature (identical
    ~98.5% rejection with CCC-only and full Dubins -- Task 3.6 finding, see
    the module docstring).

    Do NOT assume CCC is always beaten by CSC. For poses close together
    relative to R, a CCC path is often shorter -- that is exactly the regime
    CCC exists for. Always compare all six.
    """
    candidates = []
    R = 1 / params.kappa
    for first_left in [True, False]:
        for last_left in [True, False]:
            c_csc = _csc_candidate(start, goal, R, first_left, last_left, v, dt)
            if c_csc:
                candidates.append(c_csc)
    c_ccc = dubins_ccc(start, goal, params, v, dt)
    if c_ccc:
        candidates.append(c_ccc)
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.length)
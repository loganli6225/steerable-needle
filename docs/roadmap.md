# Roadmap

Why the phases are ordered the way they are. This file holds only the
sequencing rationale — see `README.md` for scope and framing,
`docs/architecture.md` for the module map and interface discipline, and
`docs/research_question.md` for the claim this is all pointed at.

Read this before proposing that a phase be skipped, merged, or reordered.

---

## Phase sequence

| Phase | Deliverable | Adds |
|---|---|---|
| **Task 1** | 2D bevel-tip kinematics + visualizer | the model everything else calls |
| **Task 2** | obstacle maps + collision checking | the environment |
| **Task 3** | RRT → kinodynamic RRT → RRT* | planning |
| **Task 3.5** | Dubins CCC exact steering | the connect-exactly primitive RRT* rewiring needs |
| **Phase 3** | EKF + particle filter, closed-loop replanning | estimation under uncertainty |
| **Phase 4** | learned deflection model / learned sampling | learning, and state-dependent kappa |

The ordering principle throughout: **each step introduces exactly one new
source of bugs.** When something breaks, the list of candidate causes should
be short. This is worth more than moving fast, because a wrong result that
looks plausible costs more than a slow one that doesn't.

---

## Why the planner sequence is RRT → kinodynamic RRT → RRT*

RRT* is the best of the three. We are not going to build it first. Four
reasons, in descending order of importance:

**1. Each step isolates one unknown.**
Vanilla RRT on a holonomic point exercises the environment, the collision
checker, and the sampling loop using a planner simple enough to be obviously
correct. If it fails, the bug is in the environment or the collision checker
— not the planner. Kinodynamic RRT then adds the curvature constraint on top
of a substrate already known to work. RRT* adds rewiring and a cost metric on
top of a kinodynamic planner already known to work.

Build RRT* first and a failure has four plausible causes at once: the
environment, the collision checker, the steering function, or the rewiring
logic. Debugging that is not a shortcut.

**2. The progression is a result, not scaffolding.**
This project's contribution is head-to-head benchmarking of methods inside one
consistent framework (see `docs/research_question.md`). Vanilla RRT and
kinodynamic RRT are not stepping stones to be discarded once RRT* works —
they are **baselines the benchmark needs**. Success rate, path cost, and
compute time *across* the three is the deliverable. Deleting the earlier
planners deletes the result.

**3. RRT* structurally depends on what the earlier ones build.**
RRT* needs (a) a steering function that connects two states exactly, for
rewiring, and (b) a cost metric. On a nonholonomic curvature-constrained
system, exact steering between two poses is a nontrivial subproblem in its own
right — Dubins-like, and easy to get subtly wrong. Solving it on top of a
working kinodynamic `extend` is far easier than solving it and the planner
simultaneously.

**4. Vanilla RRT's failure is itself evidence.**
A vanilla RRT that ignores the curvature constraint will produce paths the
needle physically cannot follow. Demonstrating that explicitly — planned path
versus executed path, diverging — motivates the kinodynamic version with data
instead of assertion. That comparison is a figure worth having, not a warm-up
exercise.

---

## What each phase must leave room for

Constraints that are cheap now and expensive to retrofit. These exist because
of decisions made in earlier phases; do not quietly drop them.

- **Task 2 must not touch Task 1.** If adding environments requires editing
  the needle model, the interface is wrong. The model does not know obstacles
  exist.
- **Collision checking is a query, not a property of the path.** The planner
  asks the environment; the environment answers. Keep them separable — the
  benchmark needs to swap environments under a fixed planner and vice versa.
- **The planner consumes `model_needle`, never `true_needle`.** The simulator
  owns ground truth. A planner that can see the true parameters silently
  invalidates every model-mismatch result downstream. This is the single
  easiest way to ruin the project without noticing.
- **Phase 4 makes kappa a function of position.** `_time_deriv` currently
  ignores `x` and `y` because kappa is constant. That will change. Anything
  that hardcodes constant curvature — in the planner, the filter, or the
  steering function — becomes a Phase 4 rewrite. Prefer passing `params`
  through to assuming `params.kappa` is a scalar forever.
- **Estimation needs full state traces, not endpoints.** Phase 3 compares
  filters; that comparison needs per-step logs. Anything that discards
  intermediate states to save memory forecloses it.

---

## Status

- **Task 1: complete.** RK4 model implemented and verified against the
  analytic solution (4th-order convergence confirmed). Acceptance tests
  rewritten to actually discriminate — see
  `tests/test_needle_model.py`.
- **Task 2: complete.** Grid environment with a baked signed distance field:
  O(1) collision (`is_free`) and clearance queries, and swept-arc checking
  (`is_arc_free`) that rolls the Task 1 model forward and asserts sample
  spacing ≤ half a cell. Verified against an analytic circle (clearance = d−r,
  resolution-convergence) and eyeballed via an SDF heatmap with straddling
  arcs. See `tests/test_grid_environment.py` and
  `scripts/eyeball_grid_environment.py`.
- **Task 3: vanilla + kinodynamic RRT done; RRT* next.** Both planners
  benchmarked on the same two scenarios (open single-circle, narrow doorway).
  Kinodynamic needs ~3-4x vanilla's nodes/iterations (steering instead of
  teleporting heading) but still finishes in well under 10% of its iteration
  budget on both. Vanilla's path is a jagged sequence of straight cuts a real
  needle cannot follow; kinodynamic's is a single smooth curvature-respecting
  arc — the before/after this phase was meant to produce (see "Vanilla RRT's
  failure is itself evidence" above). Figures: `docs/figures/rrt_vanilla_tree*.png`,
  `docs/figures/rrt_kinodynamic_tree*.png`.

  Two decisions baked into the current planner, worth recording because
  they're easy to "fix" by accident later:
  - **`nearest` uses position-only distance on purpose.** `_theta_weight` in
    `RRT.__init__` is `0.0`. Heading was originally weighted at `(1/kappa)**2`,
    but combined with a sampler that (at the time) always drew `theta=0`, it
    made `nearest` optimize for heading match over position and the tree
    collapsed into a thin fan instead of exploring — see the old failure
    plots this replaced. The sampler now draws `theta` uniformly, which
    removed the pathology, and position-only `nearest` is sufficient for both
    benchmark scenarios (neither has a passage tight enough that heading
    mismatch matters for connecting). Revisit non-zero heading weight if a
    future scenario has a gap comparable to the turning radius `1/kappa` —
    that's where picking the nearest-by-position node with the wrong heading
    starts producing edges that can't actually connect.
  - **The vanilla (straight-line) planner is retained, not deleted**, even
    though the kinodynamic planner supersedes it for actual use. Per the
    reasons above ("The progression is a result, not scaffolding"), vanilla
    is a baseline the benchmark needs, and its jagged-path failure mode is
    the comparison figure. The vanilla code path lives commented out in
    `RRT.__init__`/`extend` in `src/needlesim/planning/rrt.py` and is
    reproducible via `scripts/demo_rrt.py`.
- **Task 3.5: complete.** CCC (arc-arc-arc) Dubins steering for the bevel-tip
  needle — `dubins_ccc` in `src/needlesim/planning/dubins.py`. CCC-only, not
  the classical CSC/CCC pair: a bevel-tip needle always curves at ±kappa and
  cannot go straight, so the S segment is dropped and only RLR/LRL remain.
  Reachability is exact, not heuristic: a CCC path exists only when the two
  outer turning centres are < 4R apart, and that boundary is treated as
  central, not an edge case — beyond it `dubins_ccc` returns `None` cleanly
  (verified it doesn't raise, including fuzzed near the 4R boundary), which
  is what lets RRT* skip an edge instead of crashing. Correctness rests on
  the three analytic test classes in `tests/test_dubins.py` since there's no
  external reference solver; the strongest of the three round-trips the
  returned controls through the Task 1 `step` and checks the needle actually
  lands on the goal.

  Discretization is now exact, not deferred: each arc's final control step
  carries a trimmed `dt` (`DubinsPath.controls` is `list[tuple[Control,
  float]]`, executed via `rollout_variable`) so the executed path lands on
  the goal to floating-point precision instead of accumulating the
  `ceil()`-rounding overshoot of naively discretizing every step at a fixed
  `dt`. That overshoot was real, not just a theoretical worst case — fuzzing
  20k random reachable start/goal pairs under the naive scheme put ~10% of
  them over test 3's tolerance (worst case 1.29mm against a 0.5mm bound); the
  trimmed-final-step fix brings the same sweep's worst case down to
  floating-point noise (~1e-9mm). Not yet wired into RRT* rewiring — that
  integration is what "RRT* next" above still refers to.

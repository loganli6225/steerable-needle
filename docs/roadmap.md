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
- **Task 3: next.** RRT → kinodynamic RRT → RRT*, in that order (see the
  sequencing rationale above). First decision: the tree/node representation
  and how the planner samples poses, connects them via the needle's
  curvature-constrained `extend`, and calls `is_arc_free` for edge validity.

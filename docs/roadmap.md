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
| **Task 3** | vanilla RRT → kinodynamic RRT | planning |
| **Task 3.5** | Dubins CCC exact steering | the connect-exactly primitive RRT* rewiring needs |
| **Task 3.6** | full Dubins (CSC + CCC) steering | straight segments; and the RRT*-doesn't-scale finding |
| **Task 4** | RRT* with Dubins steering, length-only cost ("Phase A") | optimality: choose-parent + rewire |
| **Refactor** | step 1: shared `PlannerBase`; step 2: KD-tree spatial queries | benchmark-valid shared scaffolding |
| **Phase B** | clearance-weighted edge cost for RRT* | cost-aware planning (trades strict optimality for safety) |
| **Phase 3** | EKF + particle filter, closed-loop replanning | estimation under uncertainty |
| **Phase 4** | learned deflection model / learned sampling | learning, and state-dependent kappa |

Numbering note (as-executed history, per git): RRT* was originally folded
into Task 3 here, but Task 3.5 was deliberately executed in the MIDDLE of
that — after kinodynamic RRT, before RRT* — because RRT* structurally
depends on exact steering (reason 3 below). RRT* then landed as its own
**Task 4**. The refactor and Phase B were added after Task 4, once the
head-to-head benchmark became the next deliverable: the refactor exists so
the three planners share provably identical scaffolding before any numbers
are compared. "Phase A/B" name the two cost functions in
`rrt_star.py::edge_cost`, not project phases 3/4.

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
- **Task 3: complete (vanilla + kinodynamic RRT; RRT* became Task 4).** Both planners
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
  - **`nearest` uses position-only distance on purpose.** The metric is now
    the single `PlannerBase.distance` in `src/needlesim/planning/base.py`
    (the old `_theta_weight` knob is deleted; its design history is preserved
    as a comment on the method). Heading was originally weighted at
    `(1/kappa)**2`, but combined with a sampler that (at the time) always
    drew `theta=0`, it made `nearest` optimize for heading match over
    position and the tree collapsed into a thin fan instead of exploring —
    see the old failure plots this replaced. The sampler now draws `theta`
    uniformly, which removed the pathology, and position-only `nearest` is
    sufficient for both benchmark scenarios (neither has a passage tight
    enough that heading mismatch matters for connecting). Revisit non-zero
    heading weight if a future scenario has a gap comparable to the turning
    radius `1/kappa` — that's where picking the nearest-by-position node
    with the wrong heading starts producing edges that can't actually
    connect. If reintroduced, it goes in as a `KinodynamicRRT.distance`
    override, not a change to the shared base metric.
  - **The vanilla (straight-line) planner is retained, not deleted**, even
    though the kinodynamic planner supersedes it for actual use. Per the
    reasons above ("The progression is a result, not scaffolding"), vanilla
    is a baseline the benchmark needs, and its jagged-path failure mode is
    the comparison figure. Since the step-1 refactor (below) it is a
    first-class `VanillaRRT` class in `src/needlesim/planning/rrt.py` — no
    longer commented-out code — runnable by constructing `VanillaRRT` in
    place of `KinodynamicRRT` in `scripts/demo_rrt.py`.
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
  floating-point noise (~1e-9mm). Now wired into RRT* (Task 4): `steer`
  wraps `dubins_ccc` + a collision check, and is the one connection
  primitive both choose-parent and rewire use.
- **Task 4: RRT* complete (Phase A, length-only cost).** Built directly on
  the Task 3.5 primitive, per reason 3 above. The two tunable design
  choices are isolated in module-level functions, not scattered:
  `edge_cost` (currently pure `path.length`; Phase B swaps in a clearance
  penalty, A/B-able because `clearance_weight=0` reproduces Phase A) and
  `rewire_radius` (shrinking `r(n) = gamma * (log n / n)^(1/d)` with `d=3`
  — the Dubins config-space dimension, not the holonomic 2 — capped at
  `max_radius`). Target headings are derived nearest-node-toward-sample
  instead of sampled raw, because ~87% of raw sampled headings were
  CCC-unreachable. Two recorded simplifications: rewire does not propagate
  cost discounts to a re-parented node's descendants (paths stay valid,
  costs can go stale-high; documented in `rewire`), and BECAUSE costs can
  be stale, the reported best cost is recomputed from the reconstructed
  edges rather than read from `cost_from_start`. Contract, cost-consistency,
  and more-iterations-no-worse tests in `tests/test_rrt_star.py` (full
  planning budgets on purpose; originally ~16 min, ~1 min since the steer
  split below); eyeball plot via `scripts/eyeball_rrt_star.py`. Phase B not
  started.
- **Planner refactor, step 1 of 2 (pre–Phase B): complete.** The three
  planners now run on provably identical scaffolding, which is what makes
  their head-to-head benchmark valid (see "The progression is a result, not
  scaffolding" above). `PlannerBase` in `src/needlesim/planning/base.py` is
  the ONLY definition of `sample` / `nearest` / `near_indices` / `distance` /
  `reached_goal`, and now owns the spacing-invariant assert (which RRT*
  previously lacked). Hierarchy: `RRT` holds the shared loop with `extend`
  abstract; `VanillaRRT` (revived from commented-out code) and
  `KinodynamicRRT` differ ONLY in `extend`; `RRTStar` is a sibling of `RRT`,
  sharing only the base primitives, because its loop is structurally
  different (choose-parent + rewire + full-budget best-tracking).
  Behavior-preserving: full suite green (35 passed) with zero assertion
  changes — RRT*'s per-seed RNG sequence changed (the shared `sample` draws
  theta; the old RRT* sample didn't), but no seeded test flipped. Two
  hazards were recorded for the benchmark phase: `RRTConfig` and
  `RRTStarConfig` still default to DIFFERENT margins (2.0 vs 0.0) to
  preserve historical behavior — the harness must set margin explicitly and
  identically for all planners; and `nearest`/`near_indices` were still
  deliberate O(n) linear scans (resolved by step 2, below). Step 2's
  acceptance spec, `tests/test_kdtree.py`, was written and committed FIRST,
  independently: brute-force-equivalence tests for `nearest`/`near_indices`,
  including a straggler test that queries immediately after every node add —
  trivially green against the linear scans (which ARE the oracle), existing
  to catch the KD-tree's silent failure modes once it landed.
- **Planner refactor, step 2 of 2 (KD-tree): complete — and it produced a
  negative result worth keeping.** `nearest`/`near_indices` in `PlannerBase`
  are now backed by a cached `scipy.spatial.cKDTree` over (x, y) — matching
  the position-only base metric — rebuilt every K=50 node additions, with a
  linear scan of post-rebuild stragglers folded into every query. Staleness
  is guarded by three triggers (list identity, so a reused planner
  instance's second `plan()` never queries the previous run's tree;
  shrinkage; K new nodes) and rests on a documented append-only invariant:
  node positions never mutate — rewire touches parent/cost, never state.
  `near_indices` sorts the tree's hits before appending stragglers because
  `query_ball_point` doesn't sort single-point queries and choose-parent/
  rewire iterate the neighbourhood in order — index order is part of
  behavior preservation. All three gates passed: owner spec 4/4 untouched,
  full suite 41 passed with zero edits to existing tests, and
  `scripts/time_rrt_star.py` reproduced IDENTICAL node counts
  (331/1201/2530) and success flags. Two findings:
  - **The baseline's super-linear s/iter was misattributed — the O(n) scans
    were never the bottleneck.** Measured: the scans cost ~1.6s of the
    117.8s 3000-iter baseline row; profiling shows ~99% of runtime is
    `steer` → `rollout_variable` → `step` — Dubins edge collision rollouts
    in choose-parent/rewire. Their per-iteration count grows with
    neighbourhood cardinality, ~n^(1/3) since `rewire_radius` shrinks as
    (log n / n)^(1/3): measured 12.3 → 26.6 → 41.0 steer-calls/iter across
    the three timing rows, tracking s/iter almost exactly. The KD-tree made
    the queries themselves 42× faster — on ~1.4% of runtime — leaving totals
    statistically unchanged (6.99 / 42.95 / 125.54 s vs baseline
    6.67 / 40.75 / 117.83). It stays in because it's correct,
    behavior-identical, and removes the term that WOULD eventually dominate
    at much larger n (scan cost grows ~n per iteration vs ~n^(1/3) for
    steering). The feasibility problem itself was solved by the steer split
    (next entry).
  - **The obvious reuse test was vacuous for the guard it targeted.**
    Mutation-testing showed a plan()-twice test alone cannot catch a missing
    identity guard: the second call's fresh node list starts SHORTER than
    the cached count, so the shrinkage guard fires first and masks it.
    `tests/test_kdtree_reuse.py` therefore carries a second test swapping in
    a same-LENGTH different list and querying exact node positions — the one
    case only the identity guard covers. Both mutants (identity-only
    removed; identity+shrinkage removed) verified killed.
- **Steer split (the actual performance fix): complete.** `RRTStar.steer`
  fused two operations of wildly different cost: `dubins_ccc` (closed-form
  geometry, cheap) and the collision rollout (hundreds of RK4 steps, ~30×
  dearer) — and both RRT* steps paid the expensive half for every neighbour,
  then discarded almost all of it. `steer` is now split into the geometry
  call plus `_edge_collision_free`, composed cheap-first: `choose_parent`
  computes geometry + cost for ALL neighbours, sorts by cost (stable sort,
  so exact ties keep the old first-wins order), and rolls out in ascending
  order taking the first collision-free candidate; `rewire` compares cost
  BEFORE rolling out, so only candidates that would actually re-parent pay
  for a rollout. Verified NODE-EXACT against the pre-change code across 3
  runs (500 iters × seeds 1, 3; 1500 × seed 1): parent arrays, per-node
  `cost_from_start` to 1e-12, states bit-for-bit, and best_cost all
  identical — the reordering changes which conjunct is evaluated first,
  never the outcome, and touches no RNG. Numbers: 3000-iteration timing run
  117.8s → 11.2s (~10×); `rollout_variable` calls 35,221 → 2,855 (~1.9
  rollouts/iteration) while `dubins_ccc` calls stayed at 39,937 — exactly
  the split's predicted signature; full test suite 16:59 → 1:24. Benchmark
  feasibility: a 3-planner × 30-seed sweep at 3000 iterations is now ~17
  min of RRT* compute, versus ~3 hours before. `steer` itself is retained
  as the composed convenience (docstring updated to say so).

  **Lesson, recorded so it sticks: profile before optimising.** The KD-tree
  was built on an unprofiled assumption about where the time was going and
  bought ~1.4%; profiling afterwards found the real bottleneck and bought
  ~10×.

  Future work, briefly: (a) s/iter still grows ~2.5× across iteration
  counts — the ~n^(1/3) neighbourhood term now rides on the cheap
  `dubins_ccc` call rather than the rollout; vectorising the rollout in
  `models/` is the next lever if ever needed, NOT currently required;
  (b) Phase B's clearance-weighted cost interacts with the cheap-first
  ordering — see the lower-bound-prune comments in `rrt_star.py`'s `rewire`
  and `choose_parent`.
- **Task 3.6: full Dubins (CSC + CCC) steering — complete, and it produced
  a scope-changing negative result.** `dubins.py` gained the four CSC words
  (LSL, RSR, LSR, RSL) alongside the Task 3.5 CCC pair, plus `dubins_full`
  (shortest of all six). `dubins_ccc` is retained unchanged and
  `RRTStarConfig.use_full_dubins` (default True) switches between them, so
  CCC-only stays reproducible rather than being deleted. The S segment is the
  duty-cycle idealisation (`kappa_eff = kappa*(2p-1)`, so p=0.5 is a straight
  centerline) made concrete in `_discretise_straight`: alternating b=+/-1
  with half-steps at each end. Measured at v=5, dt=0.05, kappa=1/50 over a
  100mm segment: max perpendicular deviation 0.0003mm, endpoint shortfall
  0.0001mm, heading error exactly 0, arc length exact -- 0.015% of the 2mm
  planning margin, effectively free. (The naive +1/-1 alternation without the
  end half-steps bows one-sided at 0.25mm, ~800x worse; the half-steps centre
  the oscillation on the line.)

  **The finding (a result, not a footnote -- it changes the benchmark's
  scope): full Dubins fixed reachability but NOT tractability.** On the
  physically-grounded common scenario (R = 1/kappa = 50mm -- the optimistic
  end of the literature's 40-170mm range -- 150x150mm workspace, r=20mm
  obstacle, margin 2mm), RRT* built 155 nodes in 10,000 iterations (~98.5%
  sample rejection) and failed to reach the goal -- IDENTICAL node count and
  failure with CCC-only and with full Dubins (verified:
  `scripts/verify_common_scenario.py`). The word family was never the binding
  constraint. Root cause, measured directly: local pose-to-pose connections
  10-25mm apart at R=50mm cost a median of ~330mm (matched headings 332mm; the
  planner's derived nearest-node-toward-sample headings similar; random
  headings ~0% usable), because the needle's minimum turning circle is 314mm
  in circumference, so small heading corrections require most of a loop --
  and those loops sweep the whole workspace and collide. Only ~3-5% of local
  connections come in under 60mm. By contrast KinodynamicRRT succeeded on the
  same scenario (1883 nodes, 3005 iterations) and VanillaRRT trivially (103
  nodes, 120 iterations).

  This is a STRUCTURAL mismatch, not an implementation problem. RRT* requires
  exact pose-to-pose connection because rewiring is defined in terms of it --
  to ask whether node X is cheaper through the new node you must connect
  new-node to X exactly. Kinodynamic RRT never makes that demand: it advances
  the needle forward under b=+/-1 and accepts where it lands. Real needle
  insertion is monotonic, forward-directed, and roughly heading-aligned, so
  the arbitrary pose-to-pose connections RRT* needs do not arise clinically.
  RRT* therefore requires a capability the task does not need and the
  hardware cannot provide at anatomical scale.

  Benchmark scope, revised (future work, not a started task): PRIMARY is
  VanillaRRT vs KinodynamicRRT on the 150mm anatomically-scaled scenario at
  realistic curvature, with the RRT* limitation above reported as a finding;
  SECONDARY is RRT* at an enlarged (~500mm) workspace, where the turning
  radius is small relative to scene features, to demonstrate it functions
  when the scaling permits and to compare path quality against the others
  there. Open question (not a task): the current 150mm scenario is solved by
  VanillaRRT in ~120 iterations, so it is too easy to discriminate planners
  on efficiency alone. A constrained-passage scenario (the Task 3 doorway
  shape) is likely needed so vanilla's speed advantage does not dominate the
  table and the path-feasibility metric has something to bite on.

  Lesson, recorded so it sticks: this is the third time curvature-vs-scene
  scaling has been the hidden cause of a planner failure (RRT* looping at
  R=20/50 in Task 4, the 8mm doorway, now this). The governing ratios: edge
  length must be small relative to turning radius, and turning radius small
  relative to scene features. Violate either and the planner degrades in a
  way that looks like a bug.
- **Benchmark: complete (VanillaRRT vs KinodynamicRRT; RRT\* excluded per
  Task 3.6).** The PRIMARY benchmark from Task 3.6's revised scope is built,
  run, and written up. Full numbers and the four figure findings live in
  `docs/benchmark_results.md`; this is the one-paragraph pointer.

  Structure (`src/needlesim/benchmark/`): `scenarios.py` holds `Scenario` as
  plain frozen data + `build_env` (the only thing touching `GridEnvironment`),
  so hand-designed and random scenarios are one type the harness treats
  uniformly. Four hand-designed scenarios, two of them tuned from a fine
  difficulty sweep (`experiments/results/scenario_tuning/`, gitignored):
  `open` (baseline, both 30/30), `constrained_passage` (16mm gap off-axis,
  Kino 19/30 — a rate scenario), `target_behind` (goal symmetric behind the
  obstacle, Kino 0/30 — a deliberate PASS/FAIL, because the in-band region is
  a 2° knife edge), `cluttered` (Kino 29/30). This RESOLVES Task 3.6's open
  question — `constrained_passage` is exactly the doorway-shape scenario that
  gives vanilla's speed advantage something to bite on. `random_scenarios.py`
  is a seeded generator (30 scenarios persisted to the tracked
  `experiments/random_scenarios.json`, byte-reproducible) with the
  distribution stated in its docstring; overlaps permitted, no solvability
  screening.

  `harness.py` runs the 840-run grid (4×2×30 + 30×2×10), one CSV row per run,
  appended live and resumable; metrics computed by rolling controls through
  the ONE real model (κ=1/50): path_cost, endpoint_error, and heading
  discontinuity. `run_benchmark.py` → `experiments/results/benchmark_raw.csv`
  (gitignored, regenerable); `analyze_benchmark.py` → the tables;
  `plot_benchmark_figures.py` → `docs/figures/benchmark_*.png` (tracked; the
  near-median-cost representative seed per cell).

  VanillaRRT's execution metric was REVISED 2026-08-16 (`benchmark/
  vanilla_tracker.py`): its raw stored controls (one `Control(v,b=+1)`/edge)
  just trace a radius-1/κ circle, so the old `endpoint_error_mm` (133–230mm)
  measured a storage convention, not path quality. It is replaced, vanilla-only,
  by an open-loop segment-following tracker that reads the planned path as a
  reference polyline and derives feasible curved controls — giving
  tracked_endpoint, max_crosstrack, and (the headline) tracked_collides.
  Kino/RRT\* are untouched: their controls are model-generated, so their
  planned==executed and endpoint_error_mm is already exact. The benchmarks were
  fully re-run; every non-vanilla-execution column reproduced EXACTLY (success,
  cost, iters, heading disc, kino/RRT\* endpoint_error all byte-identical).

  **The result — feasibility, not just optimality.** VanillaRRT is fast
  (~0.05s) but its paths are DISCONTINUOUS (heading disc order-1 rad at 27–37
  nodes) and, even under the tracker's most charitable feasible execution,
  COLLIDE with a critical structure in 24/30–30/30 runs (91% of the random set)
  while straying 40–101mm from their own planned polyline. KinodynamicRRT is
  ~35–50× slower and loses some scenarios to the curvature constraint, but its
  paths are continuous (heading disc **exactly 0.000** across all 325 successful
  runs), executable as generated, land 1.8–2.3mm off, and are actually SHORTER.
  Random set: both 26/30, vanilla-only 4/30, kino-only 0, neither 0 (the last is
  seed-specific — Vanilla solved all 30, so the unsolvable tail did not appear at
  this generator seed). No true-vs-model mismatch is introduced yet; Vanilla's
  tracked collisions are purely its point-robot planning cheat. That mismatch
  axis is Phase 3.

  Sharper finding from the tracker (worth keeping): vanilla's path is not merely
  infeasible, it is not even INFORMATIVE. Its 5mm edges are 5.7° of turn at
  R=50mm and the zigzag flips the required-turn sign every 2–5 edges, so an
  open-loop follower duty-cycles ±5.7° by accident (κ_eff ≈ 0) and traces a
  near-straight line no matter how the b-signs are chosen — a 90° corner needs
  ~16 consecutive same-sign edges and no run is ever that long. So the tracker's
  sign decisions are almost irrelevant: at this segment length / turning radius
  the executable steering information simply isn't recoverable from node
  positions, which is the geometric flip-side of the Task 3 note that vanilla's
  steering lives entirely in the per-node headings it synthesises and discards.
  Derivation + measured numbers in `docs/benchmark_results.md`.
- **Secondary benchmark (all three planners at enlarged scale): complete — and
  it FALSIFIES the strong hypothesis it was built to test.** Full numbers,
  calibration, and figures-of-the-argument live in
  `docs/benchmark_scaled_results.md`; this is the one-paragraph pointer. Task
  3.6 deferred a SECONDARY experiment: rerun RRT* at an enlarged (~500mm)
  workspace where R=50mm is small relative to scene features, to show its
  primary-scale exclusion is a consequence of SCALE, not a broken planner. The
  strong version of that — "where all three function, RRT*'s optimisation buys
  BETTER paths than kinodynamic's first-found path" — is the one I tested, so
  all three planners run, not just RRT*.

  Structure (`src/needlesim/benchmark/`): `scaled_scenarios.py` is a pure ×10/3
  geometric transform of the primary `HAND_DESIGNED` four (workspace, obstacles,
  start/goal, goal_tolerance, and resolution all scale; **margin and kappa and
  n_steps_per_extend do NOT** — the needle is unchanged, only the scene grows,
  preserving the edge/R=0.1 ratio). `harness_scaled.py` reuses the primary
  metric code for the RRT family wholesale and adds three RRT*-only adapters (its
  result carries `control_dt_pairs`/`best_cost`, a different shape); separate CSV
  (`benchmark_scaled_raw.csv`), never pooled with the primary. Two calibrated,
  reported-not-equalised budgets: RRT family 20000, RRT* 5000 (cost plateaus by
  5000). One required finding from calibration: **RRT*'s rewire radius is in mm,
  so gamma must scale ×10/3 to ~133 — at the default 40 its neighbourhoods
  shrink below the enlarged scene and it fails EVERY run.**

  **The result.** RRT* is implementable and correct at scale (continuous,
  on-target paths — heading disc 0.000 across all 16 successful runs, endpoint
  error within tol), so the primary exclusion is genuinely about scale. But it
  is **4.4–5.7× LONGER and 40–263× slower** than kinodynamic on every scenario
  where both succeed (open +443%, constrained +568%, target_behind +502%;
  cluttered RRT\* 0/10 so no comparison), because enlargement let its
  minimum-turning-circle loops FIT without colliding but did nothing to shorten
  them. RRT*'s reliability also collapses with clutter (9/10 → 5/10 → 2/10 →
  **0/10**) while **kinodynamic is 10/10 on all four**. Notable side-result: at
  this scale kinodynamic SOLVES `target_behind` 10/10 (0/30 at primary scale) —
  the symmetric split-effort failure was scale-dependent — so target_behind is
  no longer a discriminating PASS/FAIL here. Net: the finding SHARPENS the
  primary result — RRT* is unsuited on two independent grounds (can't connect at
  anatomical scale; dominated by the turning-circle cost floor even when it
  can), not merely excluded.
- **Max-edge-length (steering-horizon) experiment: complete — answer is "both,
  scenario-dependent," and it confirms the loops are structural.** Full numbers
  and figures in `docs/max_edge_length_experiment.md`; one-paragraph pointer
  here. Tests whether rejecting Dubins edges longer than a threshold X (below the
  ~314mm loop signature) helps RRT* (short-edge tree, lower cost) or starves it
  (loops are the only way to reconcile arbitrary sampled headings, so forbidding
  them makes pairs unconnectable). Source change is minimal and gated:
  `RRTStarConfig.max_edge_length: float = inf` (default preserves behaviour —
  verified EXACT, scaled `open` seed 5 reproduces nodes=2203, cost=2340.1289…),
  checked in `choose_parent`/`rewire`/`steer` after `dubins_full` and before the
  rollout (cheap-first); two new tests. The `choose_parent` rejection rate is the
  unifying diagnostic (measured via a counting subclass in the experiment script,
  no source change). **Result:** crossing below 314mm spikes rejection ~57% →
  ~97% and collapses the tree ~14× (the predicted starvation MECHANISM is real),
  but on permissive `open` the short-edge survivors chain into DIRECT paths at
  ~392mm — matching kinodynamic's ~405mm (5.7× cheaper than unconstrained RRT*)
  with success preserved (4/5); the figures show the loops genuinely DISAPPEAR,
  not truncate. On harder `constrained_passage` the SAME cap starves to 0/5.
  Net: does NOT rescue RRT* — where capping helps it does so by making RRT*
  connect only short heading-aligned edges (i.e. becoming kinodynamic-like, and
  tying its cost), and where arbitrary-pose connection is actually needed it
  kills the planner. The secondary experiment's turning-circle-floor finding
  STANDS, sharpened: the loops are the price of exact arbitrary-heading
  connection, not wasteful detours to cap away.

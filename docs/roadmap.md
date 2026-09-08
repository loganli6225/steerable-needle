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
- **The tissue field's layer edges are in ABSOLUTE mm, and do not scale.**
  `TissueField` (Phase 4b) places its layers at fixed depths (`y = 45..70` for
  the prostate path) in a 150mm workspace. `scaled_scenarios.py` enlarges the
  world ×10/3 while deliberately holding kappa (and n_steps_per_extend) fixed —
  so if 4b/4c is ever run at the scaled (500mm) scale, those layers would
  collapse into a thin band near the entry point and the field would be
  geometric nonsense. This is a DECISION required before that combination, not
  a bug to fix now: either scale the layer edges with the workspace (keeping
  the field's relative structure) or keep them absolute (a shallow field in a
  deep world) — whichever the experiment intends. Flagged here so it is not
  discovered by a silently wrong run.

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
- **Phase 3: EKF for tip-state estimation — complete (EKF only; see scope
  note).** First phase where the `true_needle` vs `model_needle` split, kept
  clean since Task 1, does real work: the simulator advances the needle with
  TRUE kappa, the filter predicts with MODEL kappa, and the gap is the thing
  measured. `NeedleEKF` in `src/needlesim/estimation/ekf.py` maintains a
  Gaussian belief over (x, y, theta) with POSITION-ONLY, INTERMITTENT
  measurements (imaging every 20 sim steps ≈ 1 Hz against a 20 Hz sim). Heading
  is never observed; it is recovered through the position-heading covariance
  correlation that `predict` builds up (verified nonzero in
  `test_predict_builds_position_heading_correlation`, and that the Kalman gain's
  bottom row acts on it in `test_update_corrects_heading_indirectly`). 13 tests
  in `tests/test_ekf.py`, all green; 8-panel matched-vs-mismatch diagnostic via
  `scripts/eyeball_ekf.py`.

  Correctness rests on the Jacobian test, same strategy as the Dubins geometry:
  the hand-derived analytic Jacobian is checked against finite differences of
  the real `step`, not a reference solver we don't have. The Jacobian is the
  Euler-form linearisation while `step` is RK4, so agreement is deliberately
  APPROXIMATE — measured ~6e-4 across headings at v=5, kappa=1/50, dt=0.05,
  which is the right order for a first-order approximation of a fourth-order
  integrator over a small dt and is far below the process noise. Behaviour
  under matched params: clean covariance sawtooth (grow over the 20 predict-only
  steps, collapse at each update), position error <2mm, zero-mean innovation.
  Under a 2x kappa mismatch (true 1/50, model 1/25): the filter still TRACKS
  (position error bounded ~5mm, not diverging — measurements keep pulling it
  back) but its innovation goes systematically biased and heading drifts to
  ~0.47 rad. That bias is what model mismatch looks like from inside the filter,
  and it is the signal Phase 4's learned model would consume.

  Two caveats recorded here so a future phase doesn't trip on them:

  - **The innovation-mean caveat (a Phase 4 diagnostic-design point).** Mean
    innovation is a valid bias summary only on a CONSTANT-sign arc — which is
    what `test_tracks_under_model_mismatch` uses (all b=+1), so its
    `|mean| > 1.0mm` assertion is sound. But on an S-curve the bias REVERSES
    with the turn and partially cancels in the mean, so mean innovation
    UNDERSTATES the effect. This is visible in `eyeball_ekf.py`: the S-curve
    mismatch panel flips innovation sign at the b=+1→b=-1 handover (~step 300)
    and the printed mean (≈ x=-0.14, y=+0.62) is smaller than either half's
    per-segment bias. For Phase 4, prefer MEAN ABSOLUTE innovation, or
    innovation correlated against the control `b`, as the bias summary — mean
    innovation alone is misleading on any trajectory that changes turn sign.

  - **The Jacobian tripwire (leave the failing test failing).**
    `test_jacobian_structure` asserts theta's row is exactly [0, 0, 1], which
    holds ONLY at constant kappa (theta_dot = v*kappa*b is state-independent).
    Phase 4 makes kappa position-dependent, at which point theta_dot depends on
    x and y, the bottom row gains terms, and THIS TEST WILL FAIL. That failure
    is the intended signal that the Jacobian needs its new terms — do not delete
    or weaken the assertion to make it pass; add the terms. The `jacobian`
    docstring's "WHEN THIS BREAKS" note is the paired reminder in the source.

  Scope note: the roadmap's Phase 3 line reads "EKF + particle filter,
  closed-loop replanning." The particle filter is DELIBERATELY declined, not
  skipped — the model is smooth, low-dimensional and only mildly nonlinear,
  the regime where linearisation works and a PF's multimodal advantage doesn't
  materialise (rationale in the `ekf.py` DESIGN CHOICES docstring; the honest
  limitation — the EKF's unimodal-Gaussian assumption would fail under genuine
  tip-location ambiguity, which continuous position measurement prevents — is
  recorded there too). Closed-loop replanning (feeding the belief back to a
  planner and replanning on drift) is built next, as Phase 3.5.
- **Phase 3.5: closed-loop control (plan / execute / estimate / replan) —
  complete.** Closes the loop the Phase 3 EKF left open. `run_closed_loop` in
  `src/needlesim/control/closed_loop.py` plans from the FILTER'S estimate (never
  truth), executes a prefix, updates the EKF on intermittent position
  measurements, and replans from the estimate when cross-track drift from the
  planned polyline exceeds a threshold; `run_open_loop` is the explicit baseline
  (plan once, execute blind). The `true_needle` vs `model_needle` split does the
  work here for the fourth place in the codebase: the simulator steps with TRUE
  kappa, the filter and planner both use MODEL kappa, and closed-loop is the only
  mechanism by which that model error gets CORRECTED rather than ACCUMULATED.
  Sweep + per-seed tables via `scripts/sweep_closed_loop.py`; terminal-approach
  diagnosis via `scripts/diagnose_replans.py`. Tests in
  `tests/test_closed_loop.py` (the dedicated headline test
  `test_closed_loop_beats_open_loop_under_mismatch` is left an unfilled stub and
  skipped on purpose — the result is CONDITIONAL, so a single assert-cl-beats-ol
  would be false on `open`; the claim lives in the sweep tables below, not a
  one-pair assertion).

  **The result is CONDITIONAL — "closed-loop improves accuracy" would be an
  overclaim.** Measured over 5 seeds × 4 model curvatures (true 1/50); medians in
  mm:

  - `constrained_passage` (precision required): open-loop degrades with mismatch
    (6.5 → 11.9 → 18.6 at model 1/40, 1/33, 1/25); closed-loop beats it at every
    level (3.5 → 9.1 → 4.4) and wins 5/5 seeds at 1/25. This is where replanning
    earns its ~1.6s cost.
  - `open` (open-loop already within tolerance): open-loop is essentially immune
    (2.9 at every level); closed-loop matches it at low mismatch and DEGRADES at
    1/25 (median 6.0, winning only 2/5). Where open-loop already lands inside the
    goal tolerance, replanning is an unnecessary intervention that can only add
    variance — a reading the data support, not merely allow: every closed-loop
    degradation on `open` traces to a replan that fired when open-loop needed
    none.

  So closed-loop recovers most of the accuracy lost to curvature error WHERE
  PRECISION IS REQUIRED, and slightly degrades it where open-loop already
  suffices. The value is conditional on the scenario.

  **The terminal-approach guard, derived from a diagnosed failure.** Diagnosed
  on `open`, seed 1, model 1/25 (`scripts/diagnose_replans.py`): a replan firing
  15.6mm from the goal with 0.89 rad of heading offset returned a 399mm plan —
  25x the direct distance — whose loop drove the needle off the workspace.
  Mechanism: at radius R = 1/kappa, correcting a heading error of `off` radians
  costs R·off of arc; at R=25mm, 0.89 rad needs ~22mm of travel against 15.6mm
  remaining, so the correction is geometrically impossible and the planner can
  only answer with a loop (both curvatures loop from those poses; the true-kappa
  planner failed outright at 20,000 iterations). The guard suppresses replanning
  when `R · heading_offset > distance_to_goal`, computed with MODEL kappa — using
  truth there would be the same cheat as planning from truth and would invalidate
  the result. After it: `off_map` terminations vanished from `open` entirely, and
  `constrained_passage` at 1/25 improved from 3/5 to 5/5 wins.

  **This is the FOURTH appearance of the curvature-vs-scale constraint** — after
  RRT* unable to connect at anatomical scale (Task 3.6), the 8mm doorway, and
  kinodynamic's 5mm edge at R=5mm. The governing ratio, the through-line of the
  project: the turning radius must be small relative to the distances over which
  corrections are required. Terminal-approach replanning violates it because the
  distance remaining shrinks toward zero as the goal nears while the heading
  error need not — so near the goal even a modest heading offset is
  uncorrectable, and the planner's only honest answer is a loop the guard must
  forbid.

  **Two instrumentation bugs fixed, both of which changed the numbers.** (a)
  Out-of-bounds was being counted as a collision: `env.clearance` returns -1.0
  outside the world (a documented Task 2 TODO), so `is_free` reported False and
  leaving the arena registered as an obstacle strike. Now separated — `collided`
  is guarded by an in-bounds check and out-of-bounds terminates the run with
  reason `left_bounds`. This removed all three "collisions" on `open` at 1/25;
  they were never obstacle contacts. (b) A per-run `termination_reason` was added,
  because the error number alone conflates "stopped at 58mm having given up"
  (`replan_cap`) with "wandered to 58mm because it would not" (`left_bounds`).

  **One residual, documented not chased.** `constrained_passage` at model 1/33
  still produces `off_map` on 2 of 5 seeds (worst 107.2mm), while 1/40 and 1/25
  are clean. The guard's arithmetic is a LOWER bound — it checks that the heading
  correction fits in the remaining distance but ignores that the needle must also
  COVER that distance while turning — so at this one level the correction is
  marginally "possible" by the guard, replanning proceeds, and still returns a
  loop. A known limitation of the guard, flagged rather than fixed.

  Note on the sweep script: an earlier version of `sweep_closed_loop.py` had a
  variable-scoping bug (`for SCENARIO in ...` shadowed a function-local while
  `make_env` and the start/goal read the module global), so it ran
  `constrained_passage` twice and its "open" table was a mislabelled duplicate.
  The `open` findings above were verified against a corrected run and the script
  now takes the scenario explicitly; the constrained numbers were unaffected.
- **Phase 4a: joint state-parameter estimation (kappa in the filter state) —
  complete (estimator half AND feedback loop).**
  `AugmentedNeedleEKF` in `src/needlesim/estimation/ekf_augmented.py` extends
  the Phase 3 filter to a state of `(x, y, theta, log kappa)`, so curvature is
  estimated online from the same position-only intermittent measurements
  rather than assumed.

  **This is classical estimation, not machine learning, and that is the point.**
  4a is recursive/joint state-parameter estimation: one unknown scalar, estimated
  live from a signal the filter already computes — no dataset, no model class,
  no train/test split. An EKF augmentation is the RIGHT tool for a single
  well-posed parameter; reaching for a learned model here would be worse
  engineering. The machine learning is 4c, where kappa varies with POSITION and
  must be learned from trajectory data — and 4a is 4c's BASELINE, the reason it
  comes first: "does learning a spatial field beat optimally estimating a single
  number?" cannot be asked without the number-estimating version to compare
  against.

  **Log parameterisation, chosen over clamping.** The fourth state is
  log(kappa), not kappa, because kappa must stay positive: `dubins_full`
  computes R = 1/kappa, so a zero/negative estimate breaks the planner this
  phase exists to feed. Clamping would work but stops it being strictly a
  Kalman filter — the covariance is computed by the standard update and knows
  nothing about the constraint, so P[3,3] could report high confidence in a
  value the clamp is holding. Log space avoids that: any real state is valid,
  kappa = exp(state) is positive by construction, no clamp (verified: no
  clamp/clip/min/max anywhere in `update`, and `test_kappa_stays_positive_
  without_clamping` drives log-kappa hard toward the zero boundary — down to
  ~1/5790 — and it stays positive). The uncertainty band is therefore
  MULTIPLICATIVE in kappa-space (`kappa * exp(+/- sd_log)`); plotting
  `kappa +/- sqrt(variance)` would be wrong and could dip negative.

  **The result: kappa converges from a 2x wrong prior.** Starting the filter
  believing 1/25 while the world runs at 1/50 (the same mismatch Phase 3.5
  measured), over 1200 steps: alternating-b leaves 5.4% of the initial error
  (final 1/47.4), the pure arc 2.1% (final 1/49.0) — ~95% recovered using only
  noisy position measurements, curvature never observed directly. The mechanism
  is Phase 3's heading-recovery correlation chain one level deeper
  (kappa -> theta -> position, built in the covariance by `predict`, so H's zero
  fourth column does not prevent correction; asserted in
  `test_predict_builds_kappa_position_correlation`). The reported uncertainty is
  HONEST, not just converging: the +/-1 sigma band narrows as evidence
  accumulates (log-sd 0.65 -> 0.11 / 0.05) and contains the true value on
  63%/82% of steps — near the ~68% a calibrated 1-sigma band should, i.e. not
  overconfident. Regenerated (numbers + band-coverage check + figure) by
  `scripts/kappa_convergence.py`.

  **A corrected assumption (recorded, not buried).** The module docstring
  originally asserted as fact that kappa is only observable while TURNING, so
  pure constant-b arcs make it poorly identifiable (a wrong kappa and a wrong
  initial heading producing similar traces). The measurement DISAGREED: the pure
  arc converged BETTER (2.1% vs 5.4%). The ambiguity requires the HEADING prior
  to be genuinely uncertain; this filter starts with a confident heading prior
  (sigma 0.05 rad) and a bad kappa prior, so heading is pinned and position
  evidence flows into kappa, while flipping b partially CANCELS the accumulating
  position error that carries the signal. Recorded in the docstring rather than
  asserted as a test — one seed at one parameter set corrects an assumption but
  does not establish a property — and the obsolete
  `test_kappa_less_observable_on_a_pure_arc` (and its helper) were deleted.

  **A second correction, of my own verification (the instructive one).** The
  Jacobian's fourth column carries a chain-rule factor of kappa plus two
  second-order position terms (d(x')/d(log k), d(y')/d(log k)); the terms ARE
  correct and cut the FOURTH-COLUMN residual against finite differences from
  ~4e-4 to ~2e-6. But an earlier writeup claimed omitting them raises the
  WHOLE-MATRIX error to 3.1e-2, "caught" by the test's 5e-3 tolerance. That did
  not reproduce: the matrix max stays ~6e-4 with or without the terms, because
  it lives in the theta column (the Euler-vs-RK4 pose residual), which these
  terms do not touch — so the whole-matrix bar never guarded them, and the test
  passed identically with them deleted. The 3.1e-2 figure belonged to a
  PLAIN-kappa parameterisation (where d(theta')/d(kappa)=v*b*dt=0.25 makes the
  column dominate the matrix max) and was carried over unchanged after the
  switch to log-kappa. Fixed: `test_augmented_jacobian_matches_finite_
  differences` now asserts on the fourth column directly at 1e-5 (mutation-
  checked — zeroing the terms passes the matrix bar but fails the column bar at
  every theta), and both docstrings are corrected.

  **The through-line lesson: re-measure after the thing being measured
  changes.** This is the second time a number of mine survived a context change
  (the first: the KD-tree's "O(n) scans are the bottleneck" assumption, refuted
  by profiling after the fact). The 3.1e-2 was true before the log
  reparameterisation and false after it, and it propagated into two docstrings
  and this file before verification caught it. A measured number is bound to the
  setup that produced it.

  **The feedback loop is now built — and the estimate is consumed.**
  `run_closed_loop_adaptive` in `src/needlesim/control/adaptive_loop.py`
  rebuilds the planner from `ekf.params` at each replan, closing the gap the
  estimator half left open. It extends Phase 3.5's two conditions to five:
  beyond open-loop (1) and drift-triggered replanning with a FIXED wrong kappa
  (2), it provides drift- (3), kappa-change- (4), and union-triggered (5)
  replanning, all from the LEARNED kappa.

  **The result: fixing the model beats merely re-aiming, and the win is in the
  TAIL.** Over five seeds on `constrained_passage` at 2x mismatch (true 1/50,
  initial guess 1/25):

  | condition | median | range |
  |---|---|---|
  | open-loop (Phase 3.5) | 18.6mm | — |
  | closed-loop, FIXED wrong kappa (Phase 3.5) | 4.4mm | 2.8–16.1 |
  | closed-loop, LEARNED kappa (this phase) | 3.0mm | 2.8–6.1 |

  The median improvement over fixed-kappa is modest — 3.0mm is close to the
  3.0mm goal-tolerance floor, so there is little headroom left once replanning
  from a better pose has done its work. The clinically meaningful effect is the
  TAIL: fixed-kappa's worst seed lands 16.1mm out, learned-kappa's 6.1mm. A
  method that usually works and occasionally misses by 16mm is not deployable;
  one bounded at 6mm is a different proposition. On `open`, where open-loop is
  already adequate (median 2.9mm) and replanning of any kind is an intervention
  without a problem, learning does not make the degradation worse than
  fixed-kappa (learned 4.8mm vs fixed 6.0mm median) — a deliberately weak,
  one-directional claim, since asserting more of a five-seed difference would
  overstate it. Both directions are pinned in `tests/test_adaptive_loop.py`
  (median-plus-worst-case under constraint; no-harm in the open).

  **The union trigger is near-redundant with kappa-change, and that is the
  result.** A union trigger (replan on drift OR kappa-change) fires almost
  identically to the kappa-change trigger alone (9/10 runs, 5 seeds x 2
  scenarios). Model-triggered replanning refreshes the path before cross-track
  drift can accumulate, so the drift arm is largely inert once kappa-replanning
  is active. Drift, which Phase 3.5 treated as the primary replanning signal, is
  therefore mostly a SYMPTOM of model error rather than an independent one —
  which also partly explains why replanning-on-drift helped in Phase 3.5 but
  only partially: it was treating the symptom. (The sole exception across the 10
  runs, open seed 1, adds one drift replan at step 381 that helps, 3.9mm ->
  3.0mm — so the drift arm is near-inert, not entirely so.)

  **The methodological lesson (a third instance of the re-measure theme).** The
  union was predicted to "fire roughly the sum of conditions 3 and 4," inferred
  from the two triggers' firing patterns measured in INDEPENDENT single-trigger
  runs (kappa-change ~21-81, drift ~101-281, near-disjoint). That inference is
  wrong, and the combined run exposes why: the firing pattern of a combined
  trigger cannot be predicted from the firing patterns of its components run
  independently, because each component alters the trajectory the others
  observe. This joins the KD-tree "O(n) scans are the bottleneck" profiling
  assumption and the stale plain-kappa 3.1e-2 Jacobian figure as a third case of
  the same theme — re-measure after the thing being measured changes.

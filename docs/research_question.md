# Research question (living document)

Written before knowing the answer, revised as I learn. Purpose: keep the
build honest and pointed at a claim, not just "a system that runs."

## Working question

> Within a single, consistent simulation framework, how do classical and
> learned methods for steerable-needle navigation trade off against each
> other, and where do they break down under model mismatch?

## Why it might matter

The needle-steering literature is fragmented: each paper uses its own
simulator, tissue model, and metrics, so cross-method comparison is nearly
impossible. A unified, reproducible testbed that benchmarks methods head to
head on image-derived environments could surface where each method actually
fails, not where its originating paper implied.

## What the classical comparison found (primary + secondary benchmarks)

The classical half of the question is answered; the learned half (below) is not
yet begun. Full detail lives in `docs/benchmark_results.md` (primary, 150mm
anatomical scale, 840 runs) and `docs/benchmark_scaled_results.md` (secondary,
500mm); the claims, briefly:

**VanillaRRT vs KinodynamicRRT — the trade is feasibility, not merely
optimality.** Vanilla is 100% successful and ~30× faster, but its paths carry
27–37 heading discontinuities each, and even under the *most charitable*
feasible execution — an open-loop tracker that reinterprets the planned path as
followable curved controls (`benchmark/vanilla_tracker.py`; vanilla-only, since
the other planners' controls are already model-generated and exact) — they
**collide with a critical structure in 24/30 to 30/30 runs** (91% across the
random set) and stray 40–101mm from their own planned polyline. Kinodynamic is
slower and sometimes fails, but its paths are continuous (heading discontinuity
exactly 0.000 across all 325 successful runs), are executable as generated,
land 1.8–2.3mm from target, and are also *shorter* (114 vs 138mm on `open`).
Vanilla's sole genuine advantage is wall-clock, bought with paths that are
longer, discontinuous, and unexecutable by a continuous vehicle.

A sharper form of the same point: vanilla's path is not merely infeasible, it is
not even *informative*. Its 5mm edges are 5.7° of turn at R=50mm, and the
zigzag flips the required turn sign every 2–5 edges — so an open-loop follower
duty-cycles ±5.7° *by accident* (κ_eff ≈ 0) and traces a near-straight line
regardless of how the b-signs are chosen; a 90° corner would need ~16 consecutive
same-sign edges and no run is ever that long. The tracker tries to recover
vanilla's steering from node *positions* and the geometry forbids it: at this
segment length and turning radius, the executable steering information simply
isn't in the path (full derivation and measured numbers in
`docs/benchmark_results.md`).

(An earlier version of this write-up cited a 133–231mm vanilla "endpoint error."
That number rolled vanilla's *raw stored controls* — one `Control(v, b=+1)` per
edge — which just trace a radius-1/κ circle, so it measured a storage convention,
not path quality. It was replaced on 2026-08-16 by the tracker metrics above,
which give vanilla's path the fairest feasible reading; the endpoint distances
dropped substantially, but the argument never rested on them — it rests on
`tracked_collides` and the heading discontinuities, both unaffected.)

**RRTStar is excluded on three measured grounds, not omitted.**
1. It cannot connect at anatomical scale — 155 nodes in 10,000 iterations
   (~98.5% sample rejection), because rewiring requires exact pose-to-pose
   connection and at R=50mm connecting poses 10–25mm apart costs a median
   ~458mm (only 3–4% come in under 60mm; the minimum turning circle is ~314mm
   around).
2. Even where it *can* connect (secondary experiment, 500mm workspace) its cost
   is dominated by that turning-circle floor: 4.4–5.7× longer than kinodynamic
   and 40–263× slower. Enlarging the workspace let the loops fit without
   colliding, but did nothing to shorten them.
3. Its reliability collapses with clutter (9/10 → 5/10 → 2/10 → 0/10 across the
   four scaled scenarios) while kinodynamic holds 10/10 — and clutter is the
   more anatomically representative case.

**The secondary experiment falsified its own hypothesis, and that is part of
the result.** It set out to show that RRT*'s optimisation would buy *better*
paths in a regime where it functions. It does not — see ground 2. A hypothesis
stated before the data and then refuted by it is recorded here deliberately,
not omitted; the falsification is what sharpens the exclusion from "can't run
here" to "wouldn't help even if it could."

**The obvious fix for ground 2 was tested, and it does not rescue RRT*.** The
turning-circle loops are heading reconciliation, so the standard remedy is a
max-edge-length (steering-horizon) constraint: reject any Dubins edge longer
than a threshold below the ~314mm loop. Sweeping that threshold on scaled `open`
(see `docs/max_edge_length_experiment.md`) shows the loops are *structural*, not
wasteful detours to cap away — crossing below 314mm spikes the `choose_parent`
rejection rate ~57% → ~97% and collapses the tree ~14×, exactly the predicted
starvation. Where capping nonetheless "helps" (permissive `open`: cost drops
5.7× to ~392mm, matching kinodynamic, with the loops visibly *gone*, not
truncated), it does so only by forcing RRT* to connect short, near-heading-
aligned edges — i.e. by making it behave like kinodynamic RRT, at kinodynamic's
cost, while still paying RRT*'s rewiring overhead. On the harder
`constrained_passage`, where arbitrary-pose connection is actually needed, the
same cap starves the planner to 0/5. So the constraint buys a *tie* on the easy
scenario and an outright *loss* on the harder one — it removes the exact-
connection capability that was RRT*'s only reason for being here, confirming
ground 2 rather than overturning it.

**One qualification, so a primary result is not overclaimed.** `target_behind`
fails kinodynamic 0/30 at the primary scale — effort splits symmetrically
between two equal detours, neither completing in budget — but is solved 10/10
at 500mm. That failure is therefore *scale-dependent*: it appears when the
workspace is tight relative to the turning radius, and is not a permanent
limitation of sampling planners at geometric symmetry.

**A practical asymmetry worth recording.** RRT* required a scale-dependent
parameter change (rewire-radius gamma ×10/3 → 133) to function at 500mm at all;
vanilla and kinodynamic ran unchanged. The rewire radius is a distance in mm,
so it must track the workspace — one more tuning surface, and it sits alongside
the cost finding as a reason the forward, monotonic kinodynamic planner is the
better fit for a forward, monotonic insertion problem.

## What closing the loop found (Phase 3.5, estimation half)

The EKF (Phase 3) is the first place the `true_needle` vs `model_needle` split
does real work — under a 2x kappa mismatch (true 1/50, model 1/25) the filter
tracks with bounded error (~3.6mm measured, not diverging) while its innovation
goes systematically biased (mean ~1–1.7mm per axis, against ~0.15mm matched),
the signal Phase 4's learned model would consume. Phase 3.5 feeds that estimate
back to the planner: plan from the estimate, execute a prefix, update the EKF,
replan when cross-track drift exceeds a threshold. Full detail in
`docs/roadmap.md`; the claims, briefly:

**Closed-loop replanning helps under constraint and slightly hurts in open
space — the value is CONDITIONAL, and "it improves accuracy" would overclaim.**
Over 5 seeds × 4 model curvatures (medians, mm): on `constrained_passage`, where
precision is required, open-loop degrades with mismatch (6.5 → 11.9 → 18.6 at
model 1/40, 1/33, 1/25) while closed-loop beats it at every level (3.5 → 9.1 →
4.4) and wins 5/5 seeds at 1/25. On `open`, open-loop is essentially immune
(2.9mm at every level) and closed-loop matches it at low mismatch but degrades
at 1/25 (6.0mm, winning only 2/5): where open-loop already lands inside the goal
tolerance, replanning is an unnecessary intervention that only adds variance.
So closed-loop recovers most of the accuracy lost to curvature error where it is
needed, and mildly costs where it is not.

**The curvature-vs-scale constraint appears a FOURTH time, in the guard.** A
replan firing 15.6mm from the goal at 0.89 rad of heading offset returned a
399mm plan (25x direct) whose loop left the workspace: at R = 1/kappa, a heading
correction costs R·off of arc, and near the goal the distance remaining shrinks
below that, so the correction is geometrically impossible and the planner can
only loop. The guard suppresses replanning when `R · heading_offset >
distance_to_goal`, computed with MODEL kappa (using truth would be the same
cheat as planning from truth). This is the same governing ratio behind RRT*'s
failure to connect at anatomical scale, the 8mm doorway, and kinodynamic's 5mm
edge at R=5mm: the turning radius must be small relative to the distances over
which corrections are required. It is the through-line of the project.

(These are genuine `true`-vs-`model` results — the simulator steps true kappa,
the filter and planner use model kappa — not single-model artifacts.)

## What estimating kappa online found (Phase 4a, estimator half)

Phase 3.5 replanned from a better *position* but kept carrying the same wrong
*belief* — every new plan used the same fixed model kappa. Phase 4a puts kappa
in the filter state: `AugmentedNeedleEKF` estimates `(x, y, theta, log kappa)`,
correcting curvature from the same position-only measurements. Full detail in
`docs/roadmap.md`; the claims, briefly:

**This is classical estimation, and that is the right tool — not a concession.**
4a is recursive joint state-parameter estimation of a single well-posed scalar:
no dataset, no model class, no train/test split. The machine learning is 4c
(kappa varying with *position*, learned from trajectory data); 4a is its
baseline, and the interesting question — "does learning a spatial field beat
optimally estimating a single number?" — cannot be posed without it.

**Kappa converges from a 2x wrong prior, and the uncertainty is honest.** From a
prior of 1/25 against a true 1/50, over 1200 steps, the estimate recovers ~95% of
the error (5.4% remaining alternating-b, 2.1% pure arc) using only noisy position
measurements, with curvature never directly observed — the same covariance
correlation chain that recovered heading in Phase 3, one level deeper
(kappa → theta → position). The +/-1 sigma band narrows with evidence and
contains the truth on 63–82% of steps, near the ~68% of a calibrated band, so
the filter is not merely converging but reporting honest confidence. Kept
positive by construction via a log parameterisation (no clamp), so the estimate
is safe to hand to a planner that computes R = 1/kappa. Regenerable end to end by
`scripts/kappa_convergence.py`, including the band-coverage check.

**A hypothesis measurement refuted (recorded, per the honesty policy above).**
The classical intuition — kappa is only identifiable while *turning*, so pure
arcs are poorly observable — was asserted as fact in the module docstring and
then contradicted: the pure arc converged *better*. The ambiguity requires an
uncertain *heading* prior; this filter's is confident, so heading is pinned and
position evidence flows into kappa. Recorded, not asserted as a test (one seed
corrects an assumption, it does not establish a property). A second correction
was to my own verification of it — a "3.1e-2" Jacobian error figure that turned
out to belong to a pre-log-space parameterisation and no longer applied after
the state changed to log(kappa); the test now guards the terms it claimed to
(fourth-column assertion) and the docstrings are fixed. Both are the same
lesson: re-measure after the thing being measured changes.

**Feeding the estimate back beats merely re-aiming — and the win is in the
tail.** `run_closed_loop_adaptive` (`src/needlesim/control/adaptive_loop.py`)
rebuilds the planner from `ekf.params` at each replan, so the plan tracks the
learned curvature rather than the original wrong guess. Over five seeds on
`constrained_passage` at 2x mismatch (medians): open-loop 18.6mm, closed-loop
with a FIXED wrong kappa 4.4mm (worst seed 16.1mm), closed-loop with LEARNED
kappa 3.0mm (worst seed 6.1mm). The median gain over fixed-kappa is small —
3.0mm sits near the 3.0mm goal tolerance — but the tail collapses from 16.1mm to
6.1mm, which is the clinically meaningful half: a method bounded at 6mm is
deployable where one that occasionally misses by 16mm is not. On `open`, where
open-loop already suffices and replanning is an intervention without a problem,
learning does not worsen the degradation relative to fixed-kappa. Both
directions are test-pinned (`tests/test_adaptive_loop.py`).

A trigger sub-finding worth keeping: replanning on EITHER cross-track drift or a
material change in the kappa belief (a union) fires almost identically to the
kappa-change trigger alone (9/10 runs). Model-triggered replanning refreshes the
path before drift can accumulate, so drift is mostly a SYMPTOM of model error
rather than an independent signal — which partly explains why Phase 3.5's
drift-triggered replanning helped but only partially: it was treating the
symptom. (These estimation-half results are all genuine `true`-vs-`model`: the
simulator steps true kappa, the filter and planner use the learned belief.)

## What the field baseline found (Phase 4c baseline, before any learned model)

Phase 4b gave the SIMULATOR a depth-dependent curvature field (four layers,
fat/muscle/capsule/gland, R = 34/22/15/28mm) while the planner and filters still
believe a single scalar. Before building anything that learns the field, the
five-way comparison was run against the field-carrying simulator with a scalar
belief of 1/29 (the thickness-weighted mean of the layer radii over the 150mm
workspace — an uninformed average prior, not a fitted value). This is the number
4c must beat. Full detail in `docs/roadmap.md`; the claims, briefly (all genuine
`true`-vs-`model`: the field drives the sim, the scalar belief drives the
planner and filters):

**Spatial structure hurts even when the average is right.** Open-loop final
error — the mismatch with no correction — jumps from 2.8mm against a 2x CONSTANT
kappa mismatch to 26.5mm against a FIELD of the same average curvature on the
`open` scenario (~9x worse, though the belief is correct on average); on
`constrained_passage` open-loop was already ~18mm under the constant error and
stays ~20mm. So it is the spatial variation, not the average error, that does
the damage.

**Learning a SCALAR against a field is destabilising — the Phase 4a ranking
reverses.** In the constant world, replanning from a learned kappa was the safe
best condition. Against the field, the learned-kappa and union triggers produce
catastrophic off-map failures on `constrained_passage` (up to 130mm, via runaway
replanning — one run fires 16 replans over a trajectory 3x normal length),
because the scalar estimate chases the moving field and the kappa-change trigger
fires endlessly. Drift-triggered re-aiming stays bounded, and FIXED closed-loop
(no learning) is the most robust of all (worst 4.5mm) — so against a field,
fixed >= learned-scalar, the opposite of Phase 4a. Feeding a scalar estimate of
a spatial field into a model-triggered replanner is a hazard, and 4a's "union ==
kappa-change" near-identity does not survive the field (another re-measure-after-
the-setup-changes instance).

**The highest-contrast layer is effectively unobservable, and process noise
cannot fix it.** The 5mm capsule (R=15) gets ~1 position measurement per
insertion (exactly 1 per seed on the clean `open` path), so the scalar estimate
never resolves the capsule spike — it settles near the average, lags the field,
and peaks ~30mm too deep. Sweeping the log-kappa process noise across three
orders of magnitude changes the field-tracking error by essentially nothing (it
floors at ~20%, what a fixed-at-average estimate gives); the binding constraint
is OBSERVABILITY (position-only, intermittent, one capsule sample per
insertion), not the filter's willingness to move. This directly constrains 4c:
a single insertion carries almost no capsule evidence, so imaging must densify
or many insertions must be pooled, and the model must widen its uncertainty
where data is absent rather than confidently interpolate — which is the argument
that motivates 4d. The bar 4c must clear is therefore fixed-closed-loop's ~3mm
WITHOUT the runaway tail, not open-loop's ~20-26mm. Regenerable end to end by
`scripts/four_c_baseline.py`.

## Falsifiable sub-claims to test later

The classical comparison above is settled. These remain open — all on the
learned / estimation half (Phases 3–4), and all still pending model mismatch:

- [ ] Uncertainty-aware planning (using a learned deflection model's
      calibrated variance) beats point-estimate planning under model
      mismatch. Ablation must show the *uncertainty* does the work, not just
      added conservatism.
- [ ] EKF vs particle filter for tip tracking: which wins, and under what
      measurement noise / dropout regimes? (EKF built — Phase 3; PF
      deliberately declined for now, see roadmap. The head-to-head this
      sub-claim asks for is still open.)
- [ ] Learned sampling distributions speed up RRT without hurting success
      rate.

## Honesty notes

- The core problem is ~20 years old (Webster, Cowan, Alterovitz, Okamura...).
  This is integration + rigorous benchmarking, NOT a novel algorithm. Pitch
  accordingly.
- Sim-only work faces one reflexive reviewer question: does it transfer to
  tissue? Keep the true/model split clean so mismatch results are credible.

## Status

Classical planning comparison **complete** — the primary and secondary
benchmarks above answer the classical side of the working question (and
falsified one hypothesis along the way). Delivered: the needle model, grid
environment, the three planners (vanilla/kinodynamic/RRT*), full Dubins
steering, the shared-scaffolding refactor, and both benchmarks; see
`docs/roadmap.md` for the as-executed history. **Complete:** Phase 3 estimation
(the EKF, `src/needlesim/estimation/ekf.py`) and Phase 3.5 closed-loop control
(`src/needlesim/control/closed_loop.py`) — the first places the `true_needle` vs
`model_needle` split does real work: the simulator steps with true kappa, the
filter predicts with model kappa, and under a 2x mismatch the filter tracks
(bounded ~3.6mm) while its innovation goes systematically biased (the Phase 4
signal). Closing the loop on that estimate helps conditionally — recovering
accuracy where precision is required (`constrained_passage`, 5/5 seeds at 2x
mismatch, 18.6 → 4.4mm) and slightly degrading it in open space where open-loop
already suffices (see the section above). The particle filter is deliberately
declined (see roadmap). **Phase 4a (kappa in the filter state) — complete:** the
augmented EKF (`src/needlesim/estimation/ekf_augmented.py`) recovers kappa from a
2x wrong prior to within 2–5% with an honest uncertainty band, and
`run_closed_loop_adaptive` (`src/needlesim/control/adaptive_loop.py`) feeds that
estimate back — rebuilding the planner from the learned kappa at each replan.
Under constraint that beats a fixed wrong kappa mainly in the tail (worst seed
16.1 → 6.1mm; median 4.4 → 3.0mm), and does no harm in the open. This is
classical parameter estimation and the baseline for 4c's learned
(position-dependent) kappa. **Phase 4b (depth-dependent tissue field in the
simulator) — complete** (`src/needlesim/models/tissue_field.py`), and its
**4c baseline is now run:** against the field with a scalar belief of 1/29
everywhere, open-loop degrades ~9x more than under a constant 2x mismatch on
`open` (2.8 → 26.5mm), scalar learning becomes DESTABILISING (catastrophic
off-map failures via runaway replanning, reversing 4a's fixed-vs-learned
ranking), and the highest-contrast capsule layer is effectively unobservable
(~1 measurement per insertion, unfixable by process noise) — see the section
above. That is the number 4c must beat. **Not yet begun:** Phase 4c itself (the
learned spatial kappa field / learned sampling). Note that with the EKF and
closed loop, endpoint/estimate errors under mismatch are now genuine `true` vs
`model` results, not one-shared-model artifacts; the planning "endpoint error"
figures above remain single-model planning artifacts.

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
27–37 heading discontinuities each and, executed under real curvature, land
133–231mm from target in a 150mm workspace — on `cluttered`, past the workspace
diagonal. Kinodynamic is slower and sometimes fails, but its paths are
continuous (heading discontinuity exactly 0.000 across all 325 successful runs),
land 1.8–2.3mm from target, and are also *shorter* (114 vs 138mm on `open`).
Vanilla's sole genuine advantage is wall-clock, bought with paths that are
longer, discontinuous, and unexecutable by a continuous vehicle.

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

## Falsifiable sub-claims to test later

The classical comparison above is settled. These remain open — all on the
learned / estimation half (Phases 3–4), and all still pending model mismatch:

- [ ] Uncertainty-aware planning (using a learned deflection model's
      calibrated variance) beats point-estimate planning under model
      mismatch. Ablation must show the *uncertainty* does the work, not just
      added conservatism.
- [ ] EKF vs particle filter for tip tracking: which wins, and under what
      measurement noise / dropout regimes?
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
`docs/roadmap.md` for the as-executed history. **Not yet begun:** Phase 3
(EKF / particle filter, closed-loop replanning) and Phase 4 (learned deflection
model, learned sampling) — which is also where the first genuine `true_needle`
vs `model_needle` mismatch enters. Until then, every "endpoint error" above is a
planning artifact under one shared model, not tissue mismatch.

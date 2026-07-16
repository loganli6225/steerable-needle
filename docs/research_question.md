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

## Falsifiable sub-claims to test later

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

Phase 0: scaffolding. Task 1 (needle model) in progress.

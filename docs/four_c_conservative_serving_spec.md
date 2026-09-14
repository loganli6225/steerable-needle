# Phase 4c — conservative field-serving specification (FROZEN before Gate 3)

**Frozen 2026-09-13, before any Gate 3 code exists.** This is the
anti-circularity commitment. Every parameter below is set from a data-side
adequacy standard or a stated safety posture, **not** from Gate 3 outcomes, so
Gate 3 grades a fixed mechanism instead of one tuned to pass it. If any number
here is later changed *in response to a Gate 3 collision count*, the gate stops
testing the mechanism and starts defining it — the same inverse crime that made
Gates 1–2 certify only self-consistency, moved up one level into the tuning.

## What this is

Turns the fit — a learned mean curvature `k_hat(y)` plus data-side coverage
signals — into a single scalar `k_eff(y)` served to the planner through the
step-0 lookup table. The planner gets a good model where data supports it and a
*collision-safe* model where it does not, **without ever consulting the
posterior variance**, which step 2 proved worthless (the capsule miscalibration
is an irreducible ~5% bias in the dangerous under-curvature direction, not a
width problem: measured `scatter/sigma < 1`, `|bias|` 6–11x the scatter and
flat as N goes 40→320).

## Invariants (non-negotiable)

1. **Never consult the posterior sigma.** It is a biased point with a shrinking
   band; it lies exactly in the capsule.
2. **Fail toward higher curvature, never toward the prior.** The scalar prior
   (R29) is a ~2x under-estimate of the capsule (R15) and under-curvature is the
   unrecoverable direction. Reverting-to-prior fails toward danger; this design
   fails away from it.
3. **Serve a scalar.** `k_eff(y)` is tabulated in the existing lookup table;
   variance never reaches the planner. Step-0 plumbing is unchanged. (Variance
   reaching the planner is Phase 4d, out of scope.)
4. **No hard switches.** `k_eff(y)` must be at least as smooth as the field
   itself (>= the 2mm transition scale), or it breaks the planner's arc
   smoothness and the EKF Jacobian. Blend, never step.

## Mechanism

### Coverage signal — two-part, and the second part closes the density blind spot

A depth `y` is declared **well-observed** only if BOTH hold:

- **(i) Depth density** `cov_d(y)` = pooled measurements within +/- w of depth y
  is at least `tau_d`. Honest under the matched model.
- **(ii) Residual whiteness in x** `cov_r(y)` = the fit residuals at depth y are
  consistent with iid noise rather than **correlated with x**.

Part (ii) exists because part (i) has a blind spot: depth density is honest only
in the coordinate it bins. Under a lateral (x) dependence the model does not
represent, `cov_d` happily reports "well-observed" while `k_hat(y)` is averaging
over an unmodeled x-variation — so the mean is served confidently where it is
wrong, and nothing fires. But that same unmodeled x-dependence makes the
residuals at that depth **structured in x** rather than white. So (ii) detects
exactly what (i) is blind to, as a data-side check, without pretending the
`k(y)` model handles x. If (ii) fails pervasively, that is not a serving-
parameter problem — it is a **model-class verdict** (`k(y)` is too narrow),
which Gate 3 case (b) is designed to surface and price.

### Served value

Let `s(y)` in [0,1] be a smooth well-observed score (1 where both signals pass
comfortably, 0 where either fails, blended over a transition >= 2mm). Then:

```
k_eff(y) = s(y) * k_hat(y)  +  (1 - s(y)) * k_cons(y)
k_cons(y) = min( k_hat(y) * (1 + m),  k_max )
```

Where covered, `k_eff = k_hat` (the good ~95%-of-recovery estimate, served
untouched). Where uncovered, or where residuals go non-white, `k_eff` inflates
toward higher curvature, capped at `k_max`. The capsule, given characterization
dwell arcs, is covered — so it is **served, not the fallback's target**; the
fail-safe fires in genuinely uncovered depths and in any depth (ii) flags.

## Frozen parameters and their justifications

| param | value | justification (must survive "did you pick this because it worked?") |
|---|---|---|
| `tau_d` | ~3 measurements / mm of depth | coverage-adequacy standard: below this a 2mm feature has < ~6 samples, the resolution floor the whole capsule arithmetic rests on. Data-side, independent of any outcome. |
| `w` | 2–3 mm | the field's own transition scale; a narrower window aliases the boundary, a wider one smears layers. |
| `tau_r` | residual–x correlation test at alpha = 0.05 | a standard iid-noise test at a conventional level, chosen before any Gate 3 data exists. |
| `m` | **0.20 (20%)** | **asymmetry of failure, not symmetric error.** Under-curving is unrecoverable (planner plans a straighter pass, the needle over-shoots into structure); over-curving is recoverable (tighter assumed turn -> wider berth -> drift-replan corrects it). The costs are not symmetric, so the margin is not set as if they were. 20% > the ~5% known step-2 bias plus headroom. It applies ONLY where coverage fails, so it costs nothing in well-observed regions; the price is paid only in depths the characterization did not reach, where conservatism is wanted anyway. Checked: at `k_max = R15`, `k_hat*1.2` stays below the cap in every non-capsule tissue (fat R34->R28, muscle R22->R18, gland R28->R23), so 20% is a moderate tightening, not a wild one, and saturates only in the capsule. |
| `k_max` | **R = 15 mm (k = 1/15 ~ 0.0667)** | **physical curvature bound.** R15 is the minimum turning radius an optimized bevel-tip needle achieves in the literature; assuming anything tighter is assuming a needle that does not exist. At the cap the conservative branch therefore assumes the tightest turn *physically possible*, so it **can never under-curve relative to any real truth** — the unrecoverable direction is structurally excluded at the cap. This holds regardless of the field's actual values. NON-LOAD-BEARING COINCIDENCE, recorded so it is not mistaken for validation: R15 happens to equal this field's capsule truth (4b built the capsule at the literature-tight value), so inflation in the capsule lands near truth — but the cap would be equally justified if the capsule were R20, because the justification is the physical bound, not this field's capsule outcome. |
| blend width | 2–3 mm | invariant 4 (>= field transition scale). |

Of these, `m` and `k_max` are safety-posture choices (not derivations); the rest
follow from data-side adequacy. All frozen as of the date above.

## Held-out discipline (the anti-circularity commitment)

- The parameters above are frozen **before** Gate 3 code exists.
- Gate 3's verdict is reported on a **held-out misspecification**: develop /
  sanity-check against case (c) (the mild Gaussian bump, which step 2 predicts
  is near-inert), and hold out cases (a) (sub-length-scale capsule) and (b)
  (lateral x-tilt) — the real threats — for the verdict. (b) is the one that
  tests the coverage signal's own foundation via part (ii); (a) is the one that
  amplifies the known dangerous-direction bias.
- Two metrics, kept separate, job-2 deciding: **job 1** (recovery bias per
  misspecification) is diagnostic; **job 2** (collision + endpoint on a
  capsule-relevant held-out scenario, conservative-served field in the planner)
  decides go/no-go. A large recovery bias that serving covers is fine; a small
  one that tips a collision is not.

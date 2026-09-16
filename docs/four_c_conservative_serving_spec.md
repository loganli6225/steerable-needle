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
   itself, or it breaks the planner's arc smoothness and the EKF Jacobian.
   Blend, never step. PRECISELY (settled in task 1): "as smooth as the field"
   means the served field's smoothing SCALE >= the field's own. The field's
   sigmoid scale is `transition_mm/4 = 0.5mm` (full transition width ~2mm); the
   blend applies a Gaussian with `BLEND_SIGMA_MM = 1.0mm` (FWHM ~2.4mm) -- 2x
   the field's scale, and FWHM above the 2mm width -- so k_eff is smoother than
   the field by both measures. The invariant is about scale >= 0.5mm-field-scale,
   NOT "sigma >= 2mm".

## Mechanism

### Coverage signal — two-part, and the second part closes the density blind spot

A depth `y` is declared **well-observed** only if BOTH hold:

- **(i) Depth density** `cov_d(y)` = pooled measurements within +/- w of depth y
  is at least `tau_d`. Honest under the matched model.
- **(ii) Residual whiteness in x** `cov_r(y)` = the fit residuals at depth y are
  consistent with iid noise rather than **correlated with x**, judged by EFFECT
  SIZE (|r| >= 0.30, with significance p < 0.05 as a necessary secondary gate),
  not by significance alone — see the amendment note below.

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
| `tau_r` | residual–x correlation: EFFECT-SIZE floor \|r\|>=0.30 (ramp to 0.45), AND significance p<0.05 | see the amendment note; a pure-significance test is sample-size-dependent and cries wolf at large N. |
| `m` | **0.20 (20%)** | **asymmetry of failure, not symmetric error.** Under-curving is unrecoverable (planner plans a straighter pass, the needle over-shoots into structure); over-curving is recoverable (tighter assumed turn -> wider berth -> drift-replan corrects it). The costs are not symmetric, so the margin is not set as if they were. 20% > the ~5% known step-2 bias plus headroom. It applies ONLY where coverage fails, so it costs nothing in well-observed regions; the price is paid only in depths the characterization did not reach, where conservatism is wanted anyway. Checked: at `k_max = R15`, `k_hat*1.2` stays below the cap in every non-capsule tissue (fat R34->R28, muscle R22->R18, gland R28->R23), so 20% is a moderate tightening, not a wild one, and saturates only in the capsule. |
| `k_max` | **R = 15 mm (k = 1/15 ~ 0.0667)** | **physical curvature bound.** R15 is the minimum turning radius an optimized bevel-tip needle achieves in the literature; assuming anything tighter is assuming a needle that does not exist. At the cap the conservative branch therefore assumes the tightest turn *physically possible*, so it **can never under-curve relative to any real truth** — the unrecoverable direction is structurally excluded at the cap. This holds regardless of the field's actual values. NON-LOAD-BEARING COINCIDENCE, recorded so it is not mistaken for validation: R15 happens to equal this field's capsule truth (4b built the capsule at the literature-tight value), so inflation in the capsule lands near truth — but the cap would be equally justified if the capsule were R20, because the justification is the physical bound, not this field's capsule outcome. |
| blend width | Gaussian sigma = 1.0mm (FWHM ~2.4mm) | invariant 4: >= the field's own sigmoid scale (0.5mm); 2x on scale, FWHM above the 2mm transition width. |

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

## Amendments (post-freeze, from task-1 verification on matched + generic fixtures)

Recorded transparently. Both were driven ONLY by the matched model and generic
fixtures during the task-1 build; NEITHER looked at held-out (a) or (b), so the
anti-circularity commitment stands.

**Amendment 1 (2026-09-14) — whiteness test: effect size, not significance.**
The original `tau_r` wording ("residual-x correlation test at alpha=0.05") was a
pure significance test, which is sample-size-dependent. Measured under the
matched model, the densely-sampled fat band (200-430 measurements/bin) produced
|r| ~ 0.19-0.26 — explaining < 7% of cross-track variance, a path-phase-
entanglement artifact, not real x-structure — yet that cleared p < 0.05 and the
test FIRED, marking a well-observed region uncovered and inflating it. That is
the cry-wolf failure this verification step existed to catch. The generic
x-gradient fixture gave |r| = 0.4-0.89, so there is a clean gap. The criterion
now gates on EFFECT SIZE: white if |r| <= 0.30, structured if |r| >= 0.45
(smooth ramp between), AND still requires p < 0.05 so a tiny noisy bin cannot
fire on a meaningless high |r|. The floor 0.30 = "correlations explaining
< ~9% of cross-track variance are noise," comfortably above the measured 0.26
null and below real structure; it sits in a genuine gap, not placed to make
anything pass. Implemented in `scripts/four_c_serving.py::residual_whiteness_score`.

Re-confirmed 2026-09-16 on the AUGMENTED (multi-depth) pool at task-2 scale
(N=130) -- the pool task 2 actually uses: null significant-|r| max 0.167
(combined-gate fire 3%, ~alpha) vs structure |r| median 0.81; the 0.30 floor
sits in the same clean gap there. The 0.26 / 0.4-0.89 numbers above are the
original pre-multi-depth calibration, retained for provenance. Two residual
notes: (i) there is NO multiple-comparison correction across bins -- at
N>=130 the null |r| is small enough (0.167) that the effect-size gate absorbs
the ~alpha chance-significant bins, but a SMALL-N pool can still clear both
gates by luck (an N=55 null check hit |r|=0.556 in 1 of 23 bins), so whiteness
MUST be computed at large N (>=130) in task 2, not on a reduced pool; (ii) the
gate fails OPEN on bins too sparse to test (< R_MIN_BIN measurements), which is
safe only because the density signal uses the SAME +/- DENS_W window and so
already flags those bins as low-coverage -- verified the two windows match.

**Amendment 2 (2026-09-14) — multi-depth-entry characterization arcs.** A
constant-b arc cannot climb from the perineal face (y=20) to the gland (y>70):
its turning circle (R <= 34mm, shrinking as it climbs) caps the reachable depth
at ~y63 (measured: vertical launch peaks at y50.8, off-vertical at y63.2; the
deepest arc in the revisit+dwell pool reaches y69.9). So "add depth-spanning
arcs" is not available under the frozen constant-b protocol. The fix keeps
constant-b and uses the phantom's characterization license (the same license
that permits the depth-revisiting arcs): place needles AT DEPTH in the gel
block. `Insertion` gained `start_y` (default 20.0, so every existing gate
insertion is byte-identical and the gates reproduce); `multi_depth_arcs` in
`four_c_serving.py` enters across start depths 30-115mm, bringing gland density
from ~0 to ~14/mm (> tau_d) while keeping constant-b (no deconvolution). Not
option (a) duty-cycling (breaks constant-b, inhomogeneous dataset) nor (c)
serve-gland-conservatively (the gland truth R28 ~ the R29 prior, so failing-high
there moves AWAY from truth in the terminal-approach region every job-2 path
ends in — a confound on the endpoint half of the verdict metric). Verified the
augmented pool does not weaken Gate-1 separation (separation and coverage are
independent; the revisit arcs still supply separation).
